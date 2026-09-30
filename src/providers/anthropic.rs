//! Anthropic Messages API adapter: translates OpenAI chat-completions in both directions.

use super::openai::{sse_data, SseSplitter};
use super::*;
use crate::config::ProviderConfig;
use futures::StreamExt;
use serde_json::json;

const DEFAULT_MAX_TOKENS: u64 = 8192;

pub struct AnthropicProvider {
    core: ProviderCore,
    http: reqwest::Client,
    base_url: String,
    api_key_env: String,
    timeout: Duration,
}

impl AnthropicProvider {
    pub fn new(core: ProviderCore, pc: &ProviderConfig, timeout: Duration) -> Self {
        Self {
            core,
            http: reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().expect("http client"),
            base_url: pc.base_url.clone().unwrap_or_else(|| "https://api.anthropic.com".into()).trim_end_matches('/').to_string(),
            api_key_env: pc.api_key_env.clone().unwrap_or_else(|| "ANTHROPIC_API_KEY".into()),
            timeout,
        }
    }

    /// Returns the API key or the reason it is unusable.
    fn credential(&self) -> Result<String, String> {
        let env = &self.api_key_env;
        std::env::var(env).ok().filter(|k| !k.is_empty()).ok_or(format!("{env} is not set"))
    }
}

// ---------- request translation ----------

fn push_msg(msgs: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = msgs.last_mut()
        && last["role"] == role {
            last["content"].as_array_mut().unwrap().extend(blocks);
            return;
        }
    msgs.push(json!({"role": role, "content": blocks}));
}

fn text_block(t: &str) -> Option<Value> {
    (!t.is_empty()).then(|| json!({"type": "text", "text": t}))
}

fn user_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => text_block(s).into_iter().collect(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p["type"].as_str() {
                Some("text") => text_block(p["text"].as_str().unwrap_or("")),
                Some("image_url") => {
                    let url = p["image_url"]["url"].as_str().or(p["image_url"].as_str())?;
                    let source = match url.strip_prefix("data:").and_then(|r| r.split_once(";base64,")) {
                        Some((mime, data)) => json!({"type": "base64", "media_type": mime, "data": data}),
                        None => json!({"type": "url", "url": url}),
                    };
                    Some(json!({"type": "image", "source": source}))
                }
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

pub fn to_anthropic(req: &ChatRequest, upstream: &str) -> Value {
    let mut system: Vec<String> = vec![];
    let mut msgs: Vec<Value> = vec![];
    for m in &req.messages {
        match m["role"].as_str() {
            Some("system" | "developer") => system.push(content_text(&m["content"])),
            Some("user") => push_msg(&mut msgs, "user", user_blocks(&m["content"])),
            Some("assistant") => {
                let mut blocks: Vec<Value> = text_block(&content_text(&m["content"])).into_iter().collect();
                for tc in m["tool_calls"].as_array().into_iter().flatten() {
                    let args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                    let input: Value = serde_json::from_str(args).ok().filter(Value::is_object).unwrap_or(json!({}));
                    blocks.push(json!({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": input}));
                }
                push_msg(&mut msgs, "assistant", blocks);
            }
            Some("tool") => {
                let text = content_text(&m["content"]);
                let block = json!({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": text});
                push_msg(&mut msgs, "user", vec![block]);
            }
            _ => {}
        }
    }

    let max_tokens = req.extra.get("max_tokens").or(req.extra.get("max_completion_tokens")).and_then(Value::as_u64).unwrap_or(DEFAULT_MAX_TOKENS);
    let mut body = json!({"model": upstream, "max_tokens": max_tokens, "messages": msgs, "stream": req.stream});

    let system_text = system.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    if !system_text.is_empty() {
        body["system"] = Value::String(system_text);
    }

    let choice = req.extra.get("tool_choice");
    let tools_disabled = choice.is_some_and(|c| c == "none");
    if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty() && !tools_disabled) {
        body["tools"] = tools
            .iter()
            .map(|t| {
                let f = &t["function"];
                json!({"name": f["name"], "description": f["description"].as_str().unwrap_or(""),
                       "input_schema": if f["parameters"].is_object() { f["parameters"].clone() } else { json!({"type": "object", "properties": {}}) }})
            })
            .collect();
        match choice {
            Some(Value::String(s)) if s == "required" => body["tool_choice"] = json!({"type": "any"}),
            Some(Value::String(s)) if s == "auto" => body["tool_choice"] = json!({"type": "auto"}),
            Some(c) if c["function"]["name"].is_string() => body["tool_choice"] = json!({"type": "tool", "name": c["function"]["name"]}),
            _ => {}
        }
    }
    if let Some(t) = req.extra.get("temperature").filter(|t| t.is_number()) {
        body["temperature"] = t.clone();
    } else if let Some(p) = req.extra.get("top_p").filter(|p| p.is_number()) {
        body["top_p"] = p.clone();
    }
    match req.extra.get("stop") {
        Some(Value::String(s)) => body["stop_sequences"] = json!([s]),
        Some(Value::Array(a)) if !a.is_empty() => body["stop_sequences"] = Value::Array(a.clone()),
        _ => {}
    }
    body
}

// ---------- response translation ----------

fn finish_reason(stop: Option<&str>) -> &'static str {
    match stop {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        Some("refusal") => "content_filter",
        _ => "stop",
    }
}

fn usage_of(u: &Value) -> (u64, u64) {
    let input = ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"].iter().filter_map(|k| u[*k].as_u64()).sum();
    (input, u["output_tokens"].as_u64().unwrap_or(0))
}

pub fn from_anthropic(resp: &Value, model: &str) -> (Value, Usage) {
    let mut text = String::new();
    let mut tool_calls = vec![];
    for b in resp["content"].as_array().into_iter().flatten() {
        match b["type"].as_str() {
            Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
            Some("tool_use") => tool_calls.push(json!({"id": b["id"], "type": "function",
                "function": {"name": b["name"], "arguments": b["input"].to_string()}})),
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": if text.is_empty() && !tool_calls.is_empty() { Value::Null } else { Value::String(text) }});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let (input, output) = usage_of(&resp["usage"]);
    let body = json!({
        "id": resp["id"], "object": "chat.completion", "created": now_secs(), "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(resp["stop_reason"].as_str())}],
        "usage": {"prompt_tokens": input, "completion_tokens": output, "total_tokens": input + output}
    });
    (body, Usage { input, output })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Anthropic SSE events -> OpenAI chat.completion.chunk SSE blocks.
pub struct StreamTranslator {
    id: String,
    model: String,
    created: u64,
    tool_index: HashMap<u64, usize>,
    usage: UsageCell,
}

impl StreamTranslator {
    pub fn new(model: &str, usage: UsageCell) -> Self {
        Self { id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()), model: model.into(), created: now_secs(), tool_index: HashMap::new(), usage }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        let v = json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        format!("data: {v}\n\n")
    }

    pub fn handle(&mut self, ev: &Value) -> Result<Vec<String>, ProviderError> {
        Ok(match ev["type"].as_str() {
            Some("message_start") => {
                let (i, _) = usage_of(&ev["message"]["usage"]);
                self.usage.lock().unwrap().input = i;
                vec![self.chunk(json!({"role": "assistant", "content": ""}), None)]
            }
            Some("content_block_start") if ev["content_block"]["type"] == "tool_use" => {
                let idx = self.tool_index.len();
                self.tool_index.insert(ev["index"].as_u64().unwrap_or(0), idx);
                let cb = &ev["content_block"];
                vec![self.chunk(json!({"tool_calls": [{"index": idx, "id": cb["id"], "type": "function",
                    "function": {"name": cb["name"], "arguments": ""}}]}), None)]
            }
            Some("content_block_delta") => match ev["delta"]["type"].as_str() {
                Some("text_delta") => vec![self.chunk(json!({"content": ev["delta"]["text"]}), None)],
                Some("input_json_delta") => {
                    let Some(idx) = self.tool_index.get(&ev["index"].as_u64().unwrap_or(0)) else { return Ok(vec![]) };
                    vec![self.chunk(json!({"tool_calls": [{"index": idx, "function": {"arguments": ev["delta"]["partial_json"]}}]}), None)]
                }
                _ => vec![],
            },
            Some("message_delta") => {
                if let Some(o) = ev["usage"]["output_tokens"].as_u64() {
                    self.usage.lock().unwrap().output = o;
                }
                vec![self.chunk(json!({}), Some(finish_reason(ev["delta"]["stop_reason"].as_str())))]
            }
            Some("message_stop") => {
                let u = *self.usage.lock().unwrap();
                let v = json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                    "choices": [], "usage": {"prompt_tokens": u.input, "completion_tokens": u.output, "total_tokens": u.input + u.output}});
                vec![format!("data: {v}\n\n"), "data: [DONE]\n\n".into()]
            }
            Some("error") => {
                return Err(ProviderError::new(None, format!("upstream stream error: {}", ev["error"]["message"].as_str().unwrap_or("unknown"))));
            }
            _ => vec![],
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn core(&self) -> &ProviderCore {
        &self.core
    }

    async fn health(&self) -> ProviderHealth {
        match self.credential() {
            Ok(_) => ProviderHealth::Up,
            Err(e) => ProviderHealth::Down(e),
        }
    }

    async fn send_request(&self, model: &ModelSpec, req: &ChatRequest) -> Result<ProviderOutput, ProviderError> {
        let cred = self.credential().map_err(|e| ProviderError::new(Some(401), e))?;
        let body = to_anthropic(req, &model.upstream);
        let mut rb = self.http.post(format!("{}/v1/messages", self.base_url)).header("anthropic-version", "2023-06-01").json(&body);
        rb = rb.header("x-api-key", cred);

        if !req.stream {
            let fut = async {
                let resp = rb.send().await.map_err(|e| ProviderError::new(None, e.to_string()))?;
                if !resp.status().is_success() {
                    return Err(http_error(resp).await);
                }
                resp.json::<Value>().await.map_err(|e| ProviderError::new(None, format!("invalid JSON: {e}")))
            };
            let resp = tokio::time::timeout(self.timeout, fut).await.map_err(|_| ProviderError::new(None, "timeout"))??;
            let (body, usage) = from_anthropic(&resp, &model.upstream);
            return Ok(ProviderOutput::Json { body, usage });
        }

        let resp = tokio::time::timeout(self.timeout, rb.send())
            .await
            .map_err(|_| ProviderError::new(None, "timeout waiting for response headers"))?
            .map_err(|e| ProviderError::new(None, e.to_string()))?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }
        let usage: UsageCell = Arc::default();
        let mut tr = StreamTranslator::new(&model.upstream, usage.clone());
        let mut splitter = SseSplitter::default();
        let stream = resp
            .bytes_stream()
            .map(move |chunk| -> Vec<Result<Bytes, ProviderError>> {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => return vec![Err(ProviderError::new(None, format!("stream error: {e}")))],
                };
                let mut out = vec![];
                for block in splitter.push(&chunk) {
                    let Some(ev) = sse_data(&block).and_then(|d| serde_json::from_str::<Value>(d).ok()) else { continue };
                    match tr.handle(&ev) {
                        Ok(blocks) => out.extend(blocks.into_iter().map(|b| Ok(Bytes::from(b)))),
                        Err(e) => out.push(Err(e)),
                    }
                }
                out
            })
            .flat_map(futures::stream::iter)
            .boxed();
        Ok(ProviderOutput::Stream { stream, usage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(v: Value) -> ChatRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn translates_history_with_tool_calls_and_results() {
        let r = req(json!({
            "model": "auto", "stream": true, "max_tokens": 100, "temperature": 0.2,
            "tool_choice": "required",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "read a.rs and b.rs"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.rs\"}"}},
                    {"id": "c2", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"b.rs\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "A"},
                {"role": "tool", "tool_call_id": "c2", "content": "B"},
                {"role": "user", "content": [{"type": "text", "text": "thanks"}]}
            ],
            "tools": [{"type": "function", "function": {"name": "read", "description": "d", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        }));
        let b = to_anthropic(&r, "claude-x");
        assert_eq!(b["system"], "be brief");
        assert_eq!(b["max_tokens"], 100);
        assert_eq!(b["tool_choice"], json!({"type": "any"}));
        assert_eq!(b["tools"][0]["input_schema"]["properties"]["path"]["type"], "string");
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m.len(), 3, "tool results and following user text merge into one user turn");
        assert_eq!(m[1]["content"][1]["input"]["path"], "b.rs");
        assert_eq!(m[2]["content"][0]["type"], "tool_result");
        assert_eq!(m[2]["content"][1]["tool_use_id"], "c2");
        assert_eq!(m[2]["content"][2]["text"], "thanks");
    }

    #[test]
    fn converts_non_stream_response() {
        let (b, u) = from_anthropic(
            &json!({"id": "m1", "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "read", "input": {"p": 1}}]}),
            "claude-x",
        );
        assert_eq!(b["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"p\":1}");
        assert_eq!((u.input, u.output), (10, 5));
    }

    #[test]
    fn translates_stream_events() {
        let usage: UsageCell = Arc::default();
        let mut t = StreamTranslator::new("claude-x", usage.clone());
        let evs = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 7}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "read"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ];
        let out: Vec<String> = evs.iter().flat_map(|e| t.handle(e).unwrap()).collect();
        let joined = out.join("");
        assert!(joined.contains("\"content\":\"Hel\""));
        assert!(joined.contains("\"index\":0,\"type\":\"function\"") || joined.contains("\"name\":\"read\""));
        assert!(joined.contains("\"finish_reason\":\"tool_calls\""));
        assert!(joined.trim_end().ends_with("data: [DONE]"));
        let u = *usage.lock().unwrap();
        assert_eq!((u.input, u.output), (7, 9));
        assert!(t.handle(&json!({"type": "error", "error": {"message": "x"}})).is_err());
    }
}
