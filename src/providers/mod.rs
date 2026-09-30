pub mod anthropic;
pub mod openai;

use crate::config::{Catalog, Config, ModelSpec};
use crate::metrics::{HealthTracker, LatencyStats};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Internal request format: OpenAI chat-completions. Unknown fields are preserved in `extra`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    pub fn est_tokens(&self) -> u64 {
        let chars = serde_json::to_string(&self.messages).map_or(0, |s| s.len())
            + self.tools.as_ref().and_then(|t| serde_json::to_string(t).ok()).map_or(0, |s| s.len());
        (chars / 4) as u64
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .flatten()
            .filter_map(|t| t["function"]["name"].as_str().map(String::from))
            .collect()
    }
}

/// Text of an OpenAI message content (string or array of parts).
pub fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
}
pub type UsageCell = Arc<Mutex<Usage>>;

pub enum ProviderOutput {
    /// Complete OpenAI chat.completion body.
    Json { body: Value, usage: Usage },
    /// OpenAI SSE bytes (`data: {...}\n\n` ... `data: [DONE]\n\n`).
    Stream { stream: BoxStream<'static, Result<Bytes, ProviderError>>, usage: UsageCell },
}

#[derive(Debug, Clone)]
pub struct ProviderError {
    pub status: Option<u16>,
    pub message: String,
    /// From the `Retry-After` header, when the provider sent one.
    pub retry_after: Option<Duration>,
}

/// How a failed attempt must be treated by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// The request itself is wrong (400/413/422): no cooldown, other models may still accept it.
    BadRequest,
    /// Network error, 5xx, auth: short exponential cooldown on the model.
    Transient,
    /// Rate limit (429/529): the model is skipped until the window reopens.
    RateLimited(Duration),
    /// Quota / credits / usage cap exhausted: the whole provider is skipped.
    QuotaExhausted(Duration),
}

const DEFAULT_RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(60);
const DEFAULT_QUOTA_COOLDOWN: Duration = Duration::from_secs(15 * 60);
const MAX_COOLDOWN: Duration = Duration::from_secs(6 * 3600);

impl ProviderError {
    pub fn new(status: Option<u16>, message: impl Into<String>) -> Self {
        Self { status, message: message.into(), retry_after: None }
    }

    pub fn classify(&self) -> Failure {
        let cap = |d: Duration| d.min(MAX_COOLDOWN);
        let msg = self.message.to_ascii_lowercase();
        let quota_text = [
            "insufficient_quota", "exceeded your current quota", "quota exceeded", "out of extra usage",
            "usage limit", "credit balance", "insufficient credits", "billing_hard_limit",
        ]
        .iter()
        .any(|k| msg.contains(k));
        let quota_status = matches!(self.status, Some(400 | 402 | 403 | 429));
        if self.status == Some(402) || (quota_status && quota_text) {
            return Failure::QuotaExhausted(cap(self.retry_after.unwrap_or(DEFAULT_QUOTA_COOLDOWN)));
        }
        match self.status {
            Some(429 | 529) => Failure::RateLimited(cap(self.retry_after.unwrap_or(DEFAULT_RATE_LIMIT_COOLDOWN))),
            Some(400 | 413 | 422) => Failure::BadRequest,
            _ => Failure::Transient,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(s) => write!(f, "HTTP {s}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderHealth {
    Up,
    Down(String),
}

pub struct ProviderCore {
    pub id: String,
    pub specs: Vec<ModelSpec>,
    pub tracker: Arc<HealthTracker>,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn core(&self) -> &ProviderCore;
    /// Cheap local check (credentials present); no network.
    async fn health(&self) -> ProviderHealth;
    async fn send_request(&self, model: &ModelSpec, req: &ChatRequest) -> Result<ProviderOutput, ProviderError>;

    fn id(&self) -> &str {
        &self.core().id
    }
    fn models(&self) -> &[ModelSpec] {
        &self.core().specs
    }
    fn spec(&self, model_id: &str) -> Option<&ModelSpec> {
        self.models().iter().find(|m| m.id == model_id)
    }
    fn supports_tools(&self, model_id: &str) -> bool {
        self.spec(model_id).is_some_and(|m| m.tools)
    }
    fn context_window(&self, model_id: &str) -> Option<u32> {
        self.spec(model_id).map(|m| m.context_window)
    }
    /// USD, from the configured per-million-token prices.
    fn estimated_cost(&self, model_id: &str, input: u64, output: u64) -> Option<f64> {
        let m = self.spec(model_id)?;
        Some((input as f64 * m.price_in + output as f64 * m.price_out) / 1_000_000.0)
    }
    fn latency_metrics(&self, model_id: &str) -> Option<LatencyStats> {
        self.core().tracker.latency(model_id)
    }
}

pub fn build_providers(cfg: &Config, catalog: &Catalog, tracker: Arc<HealthTracker>) -> Result<HashMap<String, Arc<dyn LlmProvider>>, String> {
    let mut out: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
    for (name, pc) in &cfg.providers {
        let specs: Vec<ModelSpec> = catalog.models.values().filter(|m| &m.provider == name).cloned().collect();
        let core = ProviderCore { id: name.clone(), specs, tracker: tracker.clone() };
        let timeout = Duration::from_secs(cfg.server.request_timeout_secs);
        let p: Arc<dyn LlmProvider> = match pc.kind.as_str() {
            "anthropic" => Arc::new(anthropic::AnthropicProvider::new(core, pc, timeout)),
            "openai" => Arc::new(openai::OpenAiProvider::new(core, pc, timeout)),
            k => return Err(format!("provider '{name}': unknown kind '{k}'")),
        };
        out.insert(name.clone(), p);
    }
    Ok(out)
}

/// Turns an HTTP error response into a ProviderError (body truncated, Retry-After honoured).
pub async fn http_error(resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let body = resp.text().await.unwrap_or_default();
    let mut e = ProviderError::new(Some(status), crate::router::classifier::truncate(&body, 500));
    e.retry_after = retry_after;
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(status: u16, msg: &str) -> ProviderError {
        ProviderError::new(Some(status), msg)
    }

    #[test]
    fn classifies_limits_per_provider_wording() {
        assert_eq!(err(429, "Rate limit reached").classify(), Failure::RateLimited(Duration::from_secs(60)));
        assert_eq!(err(529, "overloaded").classify(), Failure::RateLimited(Duration::from_secs(60)));
        // OpenAI
        assert!(matches!(err(429, "{\"code\":\"insufficient_quota\"}").classify(), Failure::QuotaExhausted(_)));
        // Anthropic subscription cap comes as a 400
        assert!(matches!(err(400, "You're out of extra usage").classify(), Failure::QuotaExhausted(_)));
        assert!(matches!(err(402, "payment required").classify(), Failure::QuotaExhausted(_)));
        assert_eq!(err(400, "invalid tool schema").classify(), Failure::BadRequest);
        assert_eq!(err(503, "down").classify(), Failure::Transient);
        assert_eq!(ProviderError::new(None, "timeout").classify(), Failure::Transient);
    }

    #[test]
    fn retry_after_is_honoured_and_capped() {
        let mut e = err(429, "slow down");
        e.retry_after = Some(Duration::from_secs(7));
        assert_eq!(e.classify(), Failure::RateLimited(Duration::from_secs(7)));
        e.retry_after = Some(Duration::from_secs(999_999));
        assert_eq!(e.classify(), Failure::RateLimited(MAX_COOLDOWN));
    }
}
