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
}

impl ProviderError {
    pub fn new(status: Option<u16>, message: impl Into<String>) -> Self {
        Self { status, message: message.into() }
    }
    /// Cooldown to apply to the model; None when the error is the request's fault.
    pub fn cooldown(&self) -> Option<Duration> {
        match self.status {
            Some(400 | 413 | 422) => None,
            Some(429) => Some(Duration::from_secs(60)),
            _ => Some(Duration::from_secs(30)),
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

/// Turns an HTTP error response into a ProviderError (body truncated).
pub async fn http_error(resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    ProviderError::new(Some(status), crate::router::classifier::truncate(&body, 500))
}
