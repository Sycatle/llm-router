//! OpenAI-compatible upstream (OpenAI, Mistral, Ollama, llama.cpp, OpenRouter...): near pass-through.

use super::*;
use crate::config::ProviderConfig;
use futures::StreamExt;

pub struct OpenAiProvider {
    core: ProviderCore,
    http: reqwest::Client,
    base_url: String,
    api_key_env: Option<String>,
    timeout: Duration,
}

/// Splits a byte stream into complete SSE event blocks (separated by a blank line).
#[derive(Default)]
pub struct SseSplitter {
    buf: String,
}

impl SseSplitter {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.push_str(&String::from_utf8_lossy(chunk).replace('\r', ""));
        let mut out = vec![];
        while let Some(i) = self.buf.find("\n\n") {
            out.push(self.buf[..i].to_string());
            self.buf.drain(..i + 2);
        }
        out
    }
}

/// The `data:` payload of an SSE block.
pub fn sse_data(block: &str) -> Option<&str> {
    block.lines().find_map(|l| l.strip_prefix("data:")).map(str::trim)
}

impl OpenAiProvider {
    pub fn new(core: ProviderCore, pc: &ProviderConfig, timeout: Duration) -> Self {
        Self {
            core,
            http: reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().expect("http client"),
            base_url: pc.base_url.clone().unwrap_or_else(|| "https://api.openai.com/v1".into()).trim_end_matches('/').to_string(),
            api_key_env: pc.api_key_env.clone(),
            timeout,
        }
    }

    fn key(&self) -> Option<String> {
        self.api_key_env.as_ref().and_then(|e| std::env::var(e).ok()).filter(|k| !k.is_empty())
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn core(&self) -> &ProviderCore {
        &self.core
    }

    async fn health(&self) -> ProviderHealth {
        match &self.api_key_env {
            Some(env) if self.key().is_none() => ProviderHealth::Down(format!("{env} is not set")),
            _ => ProviderHealth::Up,
        }
    }

    async fn send_request(&self, model: &ModelSpec, req: &ChatRequest) -> Result<ProviderOutput, ProviderError> {
        let mut body = serde_json::to_value(req).map_err(|e| ProviderError::new(None, e.to_string()))?;
        body["model"] = Value::String(model.upstream.clone());
        if req.stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }
        let mut rb = self.http.post(format!("{}/chat/completions", self.base_url)).json(&body);
        if let Some(k) = self.key() {
            rb = rb.bearer_auth(k);
        }

        if !req.stream {
            let fut = async {
                let resp = rb.send().await.map_err(|e| ProviderError::new(None, e.to_string()))?;
                if !resp.status().is_success() {
                    return Err(http_error(resp).await);
                }
                resp.json::<Value>().await.map_err(|e| ProviderError::new(None, format!("invalid JSON: {e}")))
            };
            let body = tokio::time::timeout(self.timeout, fut).await.map_err(|_| ProviderError::new(None, "timeout"))??;
            let usage = Usage {
                input: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                output: body["usage"]["completion_tokens"].as_u64().unwrap_or(0),
            };
            return Ok(ProviderOutput::Json { body, usage });
        }

        // Streaming: the timeout covers connection + response headers; failures here can still fall back.
        let resp = tokio::time::timeout(self.timeout, rb.send())
            .await
            .map_err(|_| ProviderError::new(None, "timeout waiting for response headers"))?
            .map_err(|e| ProviderError::new(None, e.to_string()))?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }
        let usage: UsageCell = Arc::default();
        let cell = usage.clone();
        let mut splitter = SseSplitter::default();
        let stream = resp
            .bytes_stream()
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| ProviderError::new(None, format!("stream error: {e}")))?;
                for block in splitter.push(&chunk).iter().filter(|b| b.contains("\"usage\"")) {
                    if let Some(v) = sse_data(block).and_then(|d| serde_json::from_str::<Value>(d).ok())
                        && let Some(p) = v["usage"]["prompt_tokens"].as_u64() {
                            *cell.lock().unwrap() = Usage { input: p, output: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) };
                        }
                }
                Ok(chunk)
            })
            .boxed();
        Ok(ProviderOutput::Stream { stream, usage })
    }
}
