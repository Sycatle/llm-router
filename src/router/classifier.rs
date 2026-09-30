use super::signals::{ContextSize, RoutingSignals, TaskType};
use crate::config::JevConfig;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

/// Minimal view of a request handed to a classifier (already truncated / summarized).
#[derive(Debug, Clone, Default)]
pub struct ClassifyInput {
    pub last_user: String,
    pub first_user: Option<String>,
    pub n_messages: usize,
    pub tool_names: Vec<String>,
    pub est_tokens: u64,
}

#[derive(Debug)]
pub struct ClassifierError(pub String);

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Anything that can turn a request into routing signals. Must never name a model.
#[async_trait]
pub trait RouterClassifier: Send + Sync {
    async fn classify(&self, input: &ClassifyInput) -> Result<RoutingSignals, ClassifierError>;
}

pub struct JevClassifier {
    http: reqwest::Client,
    cfg: JevConfig,
    api_key: Option<String>,
}

const LEVELS: usize = 4;

impl JevClassifier {
    pub fn new(cfg: JevConfig) -> Self {
        let api_key = std::env::var(&cfg.api_key_env).ok().filter(|k| !k.is_empty());
        Self { http: reqwest::Client::new(), cfg, api_key }
    }

    pub fn has_key(&self) -> bool {
        self.api_key.is_some()
    }

    fn state(input: &ClassifyInput) -> String {
        let mut s = String::new();
        s.push_str("Coding-agent request to route to the cheapest sufficient LLM.\n");
        if let Some(first) = &input.first_user {
            s.push_str(&format!("Conversation opened with: {}\n", truncate(first, 500)));
        }
        s.push_str(&format!(
            "Messages so far: {}. Approx context tokens: {}. Tools available: {}.\n",
            input.n_messages,
            input.est_tokens,
            if input.tool_names.is_empty() { "none".into() } else { input.tool_names.iter().take(20).cloned().collect::<Vec<_>>().join(", ") }
        ));
        s.push_str(&format!("Latest user message:\n{}", truncate(&input.last_user, 3000)));
        s
    }

    fn score_question(instructions: &str, levels: [&str; LEVELS]) -> Value {
        json!({"type": "score", "instructions": instructions, "criteria": levels})
    }

    pub fn build_request(&self, input: &ClassifyInput) -> Value {
        json!({
            "state": Self::state(input),
            "model": self.cfg.model,
            "questions": {
                "task_type": {
                    "type": "choice",
                    "instructions": "What kind of software engineering task is the latest user message asking for?",
                    "criteria": {
                        "question": "Simple question or explanation, no code change",
                        "repo_search": "Find or read something in the repository",
                        "small_edit": "Typo, rename, tiny local edit, trivial generation, rewording",
                        "feature": "Implement a standard feature or production code",
                        "debugging": "Diagnose and fix a bug or failure",
                        "refactor": "Restructure existing code",
                        "architecture": "System design, cross-cutting analysis of a whole codebase",
                        "tests": "Write or fix tests",
                        "docs": "Write or update documentation",
                        "other": "None of the above"
                    }
                },
                "complexity": Self::score_question("Overall difficulty of the task", ["Trivial", "Routine", "Hard", "Very hard / massive scope"]),
                "reasoning": Self::score_question("Amount of deep reasoning required", ["None", "Some", "Substantial multi-step", "Maximal, subtle, e.g. concurrency or distributed systems"]),
                "tool_intensity": Self::score_question("Expected number of tool calls (file reads, edits, commands)", ["None", "A few", "Many", "Dozens across many files"]),
                "latency_sensitivity": Self::score_question("How much the user needs a fast answer", ["Can wait", "Normal", "Wants it quick", "Interactive, instant"]),
                "ambiguity": {"type": "noul", "instructions": "The request is ambiguous, underspecified or open-ended"}
            }
        })
    }

    pub fn parse_response(body: &Value, context_size: ContextSize) -> Result<RoutingSignals, ClassifierError> {
        #[derive(Deserialize)]
        struct Answer {
            choice: Option<String>,
            score: Option<f64>,
            noul: Option<f64>,
            confidence: Option<f64>,
        }
        #[derive(Deserialize)]
        struct Resp {
            answers: HashMap<String, Answer>,
        }
        let resp: Resp = serde_json::from_value(body.clone()).map_err(|e| ClassifierError(format!("bad Jev response: {e}")))?;
        let get = |k: &str| resp.answers.get(k).ok_or_else(|| ClassifierError(format!("Jev answer '{k}' missing")));
        let level = |k: &str| -> Result<f64, ClassifierError> {
            let s = get(k)?.score.ok_or_else(|| ClassifierError(format!("Jev answer '{k}' has no score")))?;
            Ok((s / (LEVELS - 1) as f64).clamp(0.0, 1.0))
        };
        let task = get("task_type")?;
        let task_type = match task.choice.as_deref() {
            Some("question") => TaskType::Question,
            Some("repo_search") => TaskType::RepoSearch,
            Some("small_edit") => TaskType::SmallEdit,
            Some("feature") => TaskType::Feature,
            Some("debugging") => TaskType::Debugging,
            Some("refactor") => TaskType::Refactor,
            Some("architecture") => TaskType::Architecture,
            Some("tests") => TaskType::Tests,
            Some("docs") => TaskType::Docs,
            _ => TaskType::Other,
        };
        let confs: Vec<f64> = resp.answers.values().filter_map(|a| a.confidence).collect();
        let confidence = if confs.is_empty() { 0.5 } else { confs.iter().sum::<f64>() / confs.len() as f64 };
        Ok(RoutingSignals {
            task_type,
            complexity: level("complexity")?,
            reasoning: level("reasoning")?,
            tool_intensity: level("tool_intensity")?,
            latency_sensitivity: level("latency_sensitivity")?,
            ambiguity: get("ambiguity")?.noul.unwrap_or(0.0).clamp(0.0, 1.0),
            context_size,
            confidence,
        })
    }
}

#[async_trait]
impl RouterClassifier for JevClassifier {
    async fn classify(&self, input: &ClassifyInput) -> Result<RoutingSignals, ClassifierError> {
        let key = self.api_key.as_ref().ok_or_else(|| ClassifierError(format!("{} is not set", self.cfg.api_key_env)))?;
        let resp = self
            .http
            .post(format!("{}/v1/systemone", self.cfg.api_base.trim_end_matches('/')))
            .bearer_auth(key)
            .timeout(Duration::from_millis(self.cfg.timeout_ms))
            .json(&self.build_request(input))
            .send()
            .await
            .map_err(|e| ClassifierError(format!("Jev request failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(ClassifierError(format!("Jev returned HTTP {}", resp.status())));
        }
        let body: Value = resp.json().await.map_err(|e| ClassifierError(format!("Jev body: {e}")))?;
        Self::parse_response(&body, ContextSize::from_tokens(input.est_tokens))
    }
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Json, Router};

    async fn fake_jev(body: Value, status: u16) -> String {
        let app = Router::new().route(
            "/v1/systemone",
            post(move |Json(req): Json<Value>| {
                let body = body.clone();
                async move {
                    assert!(req["questions"]["task_type"]["criteria"].is_object());
                    (axum::http::StatusCode::from_u16(status).unwrap(), Json(body))
                }
            }),
        );
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn classifier(base: String, key: bool) -> JevClassifier {
        let cfg = JevConfig { api_base: base, api_key_env: "TEST_JEV_KEY_UNUSED".into(), model: "jev-latest".into(), timeout_ms: 500 };
        JevClassifier { http: reqwest::Client::new(), cfg, api_key: key.then(|| "k".to_string()) }
    }

    fn canned() -> Value {
        json!({"model": "jev-1", "answers": {
            "task_type": {"type": "choice", "choice": "debugging", "confidence": 0.9, "probabilities": {}},
            "complexity": {"type": "score", "score": 2.0, "confidence": 0.8, "probabilities": {}},
            "reasoning": {"type": "score", "score": 3.0, "confidence": 1.0, "probabilities": {}},
            "tool_intensity": {"type": "score", "score": 1.0, "confidence": 0.9, "probabilities": {}},
            "latency_sensitivity": {"type": "score", "score": 0.0, "confidence": 0.9, "probabilities": {}},
            "ambiguity": {"type": "noul", "noul": 0.25}
        }})
    }

    #[tokio::test]
    async fn parses_real_http_response_into_signals() {
        let c = classifier(fake_jev(canned(), 200).await, true);
        let s = c.classify(&ClassifyInput { last_user: "find the deadlock".into(), est_tokens: 50_000, ..Default::default() }).await.unwrap();
        assert_eq!(s.task_type, TaskType::Debugging);
        assert!((s.complexity - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.reasoning, 1.0);
        assert_eq!(s.context_size, ContextSize::Large);
        assert!((s.confidence - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn http_error_missing_key_and_unreachable_are_errors() {
        let c = classifier(fake_jev(json!({}), 500).await, true);
        assert!(c.classify(&ClassifyInput::default()).await.is_err());
        let c = classifier(fake_jev(canned(), 200).await, false);
        assert!(c.classify(&ClassifyInput::default()).await.is_err());
        let c = classifier("http://127.0.0.1:1".into(), true);
        assert!(c.classify(&ClassifyInput::default()).await.is_err());
    }
}
