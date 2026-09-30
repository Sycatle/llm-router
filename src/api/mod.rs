pub mod openai_compat;

use crate::router::RouterService;
use axum::routing::{get, post};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<RouterService>,
}

pub fn app(router: Arc<RouterService>) -> axum::Router {
    axum::Router::new()
        .route("/v1/chat/completions", post(openai_compat::chat_completions))
        .route("/v1/models", get(openai_compat::models))
        .route("/debug/routes", get(openai_compat::debug_routes))
        .route("/health", get(openai_compat::health))
        .with_state(AppState { router })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::router::classifier::{ClassifierError, ClassifyInput, RouterClassifier};
    use crate::router::signals::{ContextSize, RoutingSignals, TaskType};
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::response::IntoResponse;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    /// Real HTTP upstream speaking the OpenAI chat-completions protocol.
    async fn fake_openai(status: u16, tag: &'static str) -> String {
        let app = axum::Router::new().route(
            "/chat/completions",
            post(move |axum::Json(req): axum::Json<Value>| async move {
                if status != 200 {
                    return (StatusCode::from_u16(status).unwrap(), tag).into_response();
                }
                if req["stream"] == true {
                    let body = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{tag}\"}}}}]}}\n\ndata: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":3,\"completion_tokens\":2}}}}\n\ndata: [DONE]\n\n"
                    );
                    return ([("content-type", "text/event-stream")], body).into_response();
                }
                axum::Json(json!({"model": req["model"], "tag": tag, "usage": {"prompt_tokens": 3, "completion_tokens": 2}})).into_response()
            }),
        );
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    struct FixedClassifier(Option<RoutingSignals>);
    #[async_trait]
    impl RouterClassifier for FixedClassifier {
        async fn classify(&self, _: &ClassifyInput) -> Result<RoutingSignals, ClassifierError> {
            self.0.clone().ok_or(ClassifierError("jev down".into()))
        }
    }

    fn trivial() -> RoutingSignals {
        RoutingSignals {
            task_type: TaskType::SmallEdit,
            complexity: 0.0,
            reasoning: 0.0,
            tool_intensity: 0.0,
            latency_sensitivity: 0.3,
            ambiguity: 0.0,
            context_size: ContextSize::Small,
            confidence: 0.9,
        }
    }

    fn config(fast_url: &str, std_url: &str) -> Config {
        toml::from_str(&format!(
            r#"
            [storage]
            db_path = ":memory:"
            [providers.cheap]
            kind = "openai"
            base_url = "{fast_url}"
            [providers.big]
            kind = "openai"
            base_url = "{std_url}"
            [models."cheap/m"]
            upstream = "cheap-up"
            context_window = 100000
            [models."big/m"]
            upstream = "big-up"
            context_window = 100000
            [tiers.fast]
            models = ["cheap/m"]
            [tiers.standard]
            models = ["big/m"]
            "#
        ))
        .unwrap()
    }

    async fn post_chat(app: &axum::Router, body: Value) -> (StatusCode, axum::http::HeaderMap, String) {
        let resp = app
            .clone()
            .oneshot(Request::post("/v1/chat/completions").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let (status, headers) = (resp.status(), resp.headers().clone());
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, headers, String::from_utf8_lossy(&bytes).into())
    }

    fn chat(model: &str, stream: bool) -> Value {
        json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "fix this typo"}]})
    }

    #[tokio::test]
    async fn routes_trivial_request_to_fast_model_and_logs_decision() {
        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let app = app(svc.clone());
        let (st, h, body) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(h["x-router-tier"], "fast");
        assert_eq!(h["x-router-model"], "cheap/m");
        assert!(body.contains("cheap-up") && body.contains("\"tag\":\"A\""));
        let rec = &svc.metrics.recent(1)[0];
        assert_eq!(rec["selected_model"], "cheap/m");
        assert_eq!(rec["tokens_input"], 3);
        assert_eq!(rec["success"], true);
    }

    #[tokio::test]
    async fn provider_5xx_falls_back_to_next_tier_and_degrades_model() {
        let (a, b) = (fake_openai(500, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let app = app(svc.clone());
        let (st, h, body) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(h["x-router-model"], "big/m");
        assert!(body.contains("\"tag\":\"B\""));
        let rec = &svc.metrics.recent(1)[0];
        assert_eq!(rec["fallback_used"], true);
        assert!(svc.tracker.degraded().contains("cheap/m"));
    }

    #[tokio::test]
    async fn rate_limited_model_is_skipped_on_next_request() {
        let (a, b) = (fake_openai(429, "slow down").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let app = app(svc.clone());
        let (_, h, _) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(h["x-router-model"], "big/m");
        assert!(svc.tracker.degraded().contains("cheap/m"));
        let (_, h, _) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(h["x-router-model"], "big/m");
        let rec = &svc.metrics.recent(1)[0];
        assert!(rec["attempts"].as_array().unwrap().is_empty(), "limited model must not be retried");
    }

    #[tokio::test]
    async fn quota_exhaustion_disables_the_whole_provider() {
        let (a, b) = (fake_openai(429, "insufficient_quota").await, fake_openai(200, "B").await);
        let mut cfg = config(&a, &b);
        cfg.models.insert("cheap/n".into(), crate::config::ModelConfig { upstream: "n".into(), context_window: 100_000, tools: true, price_in: 0.0, price_out: 0.0 });
        cfg.tiers.insert("fast".into(), crate::config::TierConfig { models: vec!["cheap/m".into(), "cheap/n".into()] });
        let svc = RouterService::build(&cfg, Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let app = app(svc.clone());
        let (_, h, _) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(h["x-router-model"], "big/m");
        assert!(svc.tracker.degraded().contains("cheap"), "provider-wide cooldown");
        let (_, h, _) = post_chat(&app, chat("auto", false)).await;
        assert_eq!(h["x-router-model"], "big/m", "sibling model of the exhausted provider is not tried first");
        assert!(svc.metrics.recent(1)[0]["attempts"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn jev_down_routes_to_standard() {
        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(None))).unwrap();
        let (st, h, _) = post_chat(&app(svc.clone()), chat("auto", false)).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(h["x-router-tier"], "standard");
        assert!(svc.metrics.recent(1)[0]["jev_error"].is_string());
    }

    #[tokio::test]
    async fn all_candidates_failing_returns_502_and_forced_model_is_honoured() {
        let (a, b) = (fake_openai(500, "A").await, fake_openai(503, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let (st, _, _) = post_chat(&app(svc.clone()), chat("auto", false)).await;
        assert_eq!(st, StatusCode::BAD_GATEWAY);

        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let (_, h, _) = post_chat(&app(svc), chat("big/m", false)).await;
        assert_eq!(h["x-router-model"], "big/m");
    }

    #[tokio::test]
    async fn streams_sse_through_and_records_usage_on_completion() {
        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let (st, h, body) = post_chat(&app(svc.clone()), chat("auto", true)).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(h["content-type"], "text/event-stream");
        assert!(body.contains("\"content\":\"A\"") && body.trim_end().ends_with("data: [DONE]"));
        let rec = &svc.metrics.recent(1)[0];
        assert_eq!(rec["success"], true);
        assert_eq!(rec["tokens_output"], 2);
    }

    #[tokio::test]
    async fn context_too_large_for_every_model_is_a_400() {
        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let big = "x".repeat(600_000);
        let (st, _, _) = post_chat(&app(svc), json!({"model": "auto", "messages": [{"role": "user", "content": big}]})).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn models_and_debug_endpoints_respond() {
        let (a, b) = (fake_openai(200, "A").await, fake_openai(200, "B").await);
        let svc = RouterService::build(&config(&a, &b), Arc::new(FixedClassifier(Some(trivial())))).unwrap();
        let app = app(svc);
        let r = app.clone().oneshot(Request::get("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
        let v: Value = serde_json::from_slice(&axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert!(v["data"].as_array().unwrap().iter().any(|m| m["id"] == "auto"));
        let r = app.oneshot(Request::get("/debug/routes").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
    }
}
