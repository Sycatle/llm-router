use super::AppState;
use crate::providers::ChatRequest;
use crate::router::signals::Mode;
use crate::router::{RouteOverrides, Routed};
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::HashMap;

fn error_response(status: u16, message: &str) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    (code, Json(json!({"error": {"message": message, "type": "router_error", "code": status}}))).into_response()
}

fn overrides(headers: &HeaderMap) -> RouteOverrides {
    let get = |n: &str| headers.get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    RouteOverrides {
        session_id: get("x-session-id").or_else(|| get("x-opencode-session")).or_else(|| get("x-session-affinity")),
        min_tier: get("x-router-min-tier").and_then(|t| match Mode::parse(&t) {
            Some(Mode::Tier(t)) => Some(t),
            _ => None,
        }),
    }
}

pub async fn chat_completions(State(st): State<AppState>, headers: HeaderMap, Json(req): Json<ChatRequest>) -> Response {
    tracing::debug!(headers = ?headers.keys().map(|k| k.as_str()).collect::<Vec<_>>(), model = %req.model, stream = req.stream, "incoming");
    let out = match st.router.handle(req, overrides(&headers)).await {
        Ok(o) => o,
        Err(e) => return error_response(e.status, &e.message),
    };
    let mut resp = match out.body {
        Routed::Json(v) => Json(v).into_response(),
        Routed::Stream(s) => {
            let body = Body::from_stream(futures::StreamExt::map(s, |r| r.map_err(|e| std::io::Error::other(e.to_string()))));
            let mut r = Response::new(body);
            r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            r
        }
    };
    let h = resp.headers_mut();
    for (k, v) in [("x-router-model", out.model), ("x-router-tier", out.tier.as_str().to_string()), ("x-router-request-id", out.request_id)] {
        if let Ok(v) = HeaderValue::from_str(&v) {
            h.insert(k, v);
        }
    }
    resp
}

pub async fn models(State(st): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = st.router.virtual_models().into_iter().map(|id| json!({"id": id, "object": "model", "owned_by": "router"})).collect();
    Json(json!({"object": "list", "data": data}))
}

pub async fn debug_routes(State(st): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Json<Value> {
    let limit = q.get("limit").and_then(|l| l.parse().ok()).unwrap_or(20usize).min(500);
    Json(Value::Array(st.router.metrics.recent(limit)))
}

pub async fn health() -> &'static str {
    "ok"
}
