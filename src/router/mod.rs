pub mod classifier;
pub mod policy;
pub mod signals;

use crate::config::Config;
use crate::metrics::{DecisionRecord, HealthTracker, Metrics};
use crate::providers::{content_text, ChatRequest, LlmProvider, ProviderError, ProviderHealth, ProviderOutput, Usage, UsageCell};
use bytes::Bytes;
use classifier::{ClassifyInput, RouterClassifier};
use futures::stream::BoxStream;
use futures::Stream;
use policy::{Policy, PolicyInput, SessionView};
use serde_json::Value;
use signals::{ContextSize, Mode, RoutingSignals, Tier};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

pub struct RouterService {
    policy: Policy,
    default_mode: Mode,
    session_ttl: Duration,
    classifier: Arc<dyn RouterClassifier>,
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    sessions: Mutex<HashMap<String, Session>>,
    pub metrics: Arc<Metrics>,
    pub tracker: Arc<HealthTracker>,
}

struct Session {
    tier: Tier,
    model: String,
    history: Vec<Tier>,
    last_seen: Instant,
}

#[derive(Default)]
pub struct RouteOverrides {
    pub session_id: Option<String>,
    pub min_tier: Option<Tier>,
}

pub enum Routed {
    Json(Value),
    Stream(BoxStream<'static, Result<Bytes, ProviderError>>),
}

pub struct RouteOutcome {
    pub body: Routed,
    pub model: String,
    pub tier: Tier,
    pub request_id: String,
}

#[derive(Debug)]
pub struct RouterError {
    pub status: u16,
    pub message: String,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

impl RouterService {
    pub fn new(
        cfg: &Config,
        classifier: Arc<dyn RouterClassifier>,
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        metrics: Arc<Metrics>,
        tracker: Arc<HealthTracker>,
    ) -> Result<Self, String> {
        Ok(Self {
            policy: Policy::new(cfg.routing.clone(), cfg.catalog()?),
            default_mode: cfg.default_mode(),
            session_ttl: Duration::from_secs(cfg.routing.session_ttl_secs),
            classifier,
            providers,
            sessions: Mutex::default(),
            metrics,
            tracker,
        })
    }

    pub fn build(cfg: &Config, classifier: Arc<dyn RouterClassifier>) -> Result<Arc<Self>, String> {
        let tracker = Arc::new(HealthTracker::default());
        let catalog = cfg.catalog()?;
        let providers = crate::providers::build_providers(cfg, &catalog, tracker.clone())?;
        let metrics = Arc::new(Metrics::open(&cfg.storage.db_path).map_err(|e| format!("sqlite: {e}"))?);
        Ok(Arc::new(Self::new(cfg, classifier, providers, metrics, tracker)?))
    }

    pub async fn provider_health(&self) -> Vec<(String, ProviderHealth)> {
        let mut v = vec![];
        for (id, p) in &self.providers {
            for m in p.models() {
                tracing::info!(provider = p.id(), model = %m.id, tools = p.supports_tools(&m.id), context = ?p.context_window(&m.id), latency = ?p.latency_metrics(&m.id).map(|l| l.ema_ms), "model");
            }
            v.push((id.clone(), p.health().await));
        }
        v
    }

    pub fn virtual_models(&self) -> Vec<String> {
        let mut v: Vec<String> = ["auto", "auto-fast", "auto-standard", "auto-reasoning", "auto-frontier"].map(String::from).into();
        let mut real: Vec<String> = self.policy.catalog().models.keys().cloned().collect();
        real.sort();
        v.extend(real);
        v
    }

    fn mode_for(&self, model: &str) -> Mode {
        if let Some(m) = Mode::parse(model) {
            m
        } else if self.policy.catalog().models.contains_key(model) {
            Mode::ForceModel(model.to_string())
        } else {
            self.default_mode.clone()
        }
    }

    fn session_key(req: &ChatRequest) -> String {
        let find = |role: &str| req.messages.iter().find(|m| m["role"] == role).map(|m| content_text(&m["content"])).unwrap_or_default();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        find("system").chars().take(300).collect::<String>().hash(&mut h);
        find("user").hash(&mut h);
        format!("conv-{:016x}", h.finish())
    }

    /// Models whose provider is down or in cooldown.
    async fn degraded(&self) -> HashSet<String> {
        let mut set = self.tracker.degraded();
        for p in self.providers.values() {
            if let ProviderHealth::Down(_) = p.health().await {
                set.extend(p.models().iter().map(|m| m.id.clone()));
            }
        }
        set
    }

    pub async fn handle(&self, req: ChatRequest, ov: RouteOverrides) -> Result<RouteOutcome, RouterError> {
        let started = Instant::now();
        let request_id = format!("req-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
        let session_id = ov.session_id.clone().unwrap_or_else(|| Self::session_key(&req));
        let mode = self.mode_for(&req.model);
        let est_tokens = req.est_tokens();
        let continuation = req.messages.last().is_some_and(|m| m["role"] == "tool");

        let session = {
            let mut g = self.sessions.lock().unwrap();
            g.retain(|_, s| s.last_seen.elapsed() < self.session_ttl);
            g.get(&session_id).map(|s| SessionView { tier: s.tier, model: s.model.clone(), history: s.history.clone() })
        };

        let mut rec = DecisionRecord {
            request_id: request_id.clone(),
            session_id: session_id.clone(),
            ts_ms: now_ms(),
            mode: format!("{mode:?}"),
            previous_model: session.as_ref().map(|s| s.model.clone()),
            ..Default::default()
        };

        // 1. Signals (Auto mode only; skipped inside a tool loop of a known session).
        let mut signals: Option<RoutingSignals> = None;
        if mode == Mode::Auto && !(continuation && session.is_some()) {
            let last_user = req.messages.iter().rev().find(|m| m["role"] == "user").map(|m| content_text(&m["content"])).unwrap_or_default();
            let first_user = req.messages.iter().find(|m| m["role"] == "user").map(|m| content_text(&m["content"])).filter(|f| *f != last_user);
            let input = ClassifyInput { last_user, first_user, n_messages: req.messages.len(), tool_names: req.tool_names(), est_tokens };
            match self.classifier.classify(&input).await {
                Ok(s) => signals = Some(s),
                Err(e) => {
                    tracing::warn!(error = %e, "classifier failed; policy falls back");
                    rec.jev_error = Some(e.to_string());
                    signals = Some(RoutingSignals::unknown(ContextSize::from_tokens(est_tokens)));
                }
            }
        }
        if let Some(s) = &signals {
            rec.task_type = Some(format!("{:?}", s.task_type));
            rec.jev_scores = serde_json::to_value(s).ok();
        }

        // 2. Deterministic decision.
        let degraded = self.degraded().await;
        let decision = self.policy.decide(&PolicyInput {
            mode: &mode,
            signals: signals.as_ref(),
            continuation,
            session: session.as_ref(),
            est_tokens,
            needs_tools: req.tools.as_ref().is_some_and(|t| !t.is_empty()),
            min_tier: ov.min_tier,
            degraded: &degraded,
        });
        rec.selected_tier = decision.tier.as_str().into();
        rec.reason = decision.reason.clone();

        if decision.candidates.is_empty() {
            let message = format!("no configured model can serve this request (~{est_tokens} tokens, mode {mode:?})");
            rec.error = Some(message.clone());
            finish(&self.metrics, rec);
            return Err(RouterError { status: 400, message });
        }

        // 3. Try candidates in order; fallback is only possible before the first byte.
        let mut last_err = String::new();
        for (n, cand) in decision.candidates.iter().enumerate() {
            let Some(provider) = self.catalog_provider(&cand.model) else { continue };
            if let ProviderHealth::Down(why) = provider.health().await {
                rec.attempts.push(format!("{}: skipped ({why})", cand.model));
                continue;
            }
            let spec = provider.spec(&cand.model).expect("catalog consistent").clone();
            let t0 = Instant::now();
            match provider.send_request(&spec, &req).await {
                Ok(out) => {
                    self.tracker.record_success(&cand.model, t0.elapsed());
                    rec.selected_model = Some(cand.model.clone());
                    rec.selected_tier = cand.tier.as_str().into();
                    rec.fallback_used = n > 0;
                    if n > 0 {
                        rec.reason = format!("{}; fallback after: {}", rec.reason, rec.attempts.join(" | "));
                    }
                    self.remember(&session_id, decision.tier, &cand.model);
                    let body = match out {
                        ProviderOutput::Json { body, usage } => {
                            fill_outcome(&mut rec, &provider, &cand.model, usage, started, None);
                            finish(&self.metrics, rec);
                            Routed::Json(body)
                        }
                        ProviderOutput::Stream { stream, usage } => Routed::Stream(Box::pin(Tracked {
                            inner: stream,
                            guard: Some(FinishGuard { metrics: self.metrics.clone(), rec, provider, model: cand.model.clone(), usage, started, completed: false, error: None }),
                        })),
                    };
                    return Ok(RouteOutcome { body, model: cand.model.clone(), tier: cand.tier, request_id });
                }
                Err(e) => {
                    tracing::warn!(model = %cand.model, error = %e, "provider attempt failed");
                    self.tracker.record_failure(&cand.model, e.cooldown());
                    rec.attempts.push(format!("{}: {e}", cand.model));
                    last_err = e.to_string();
                }
            }
        }
        rec.error = Some(format!("all candidates failed: {}", rec.attempts.join(" | ")));
        finish(&self.metrics, rec);
        Err(RouterError { status: 502, message: format!("all candidate models failed; last error: {last_err}") })
    }

    fn catalog_provider(&self, model_id: &str) -> Option<Arc<dyn LlmProvider>> {
        let spec = self.policy.catalog().models.get(model_id)?;
        self.providers.get(&spec.provider).cloned()
    }

    fn remember(&self, session_id: &str, tier: Tier, model: &str) {
        let mut g = self.sessions.lock().unwrap();
        let s = g.entry(session_id.to_string()).or_insert_with(|| Session { tier, model: model.into(), history: vec![], last_seen: Instant::now() });
        s.tier = tier;
        s.model = model.into();
        s.history.push(tier);
        let excess = s.history.len().saturating_sub(3);
        s.history.drain(..excess);
        s.last_seen = Instant::now();
    }
}

fn fill_outcome(rec: &mut DecisionRecord, provider: &Arc<dyn LlmProvider>, model: &str, usage: Usage, started: Instant, error: Option<String>) {
    rec.latency_ms = Some(started.elapsed().as_millis() as u64);
    rec.tokens_input = Some(usage.input);
    rec.tokens_output = Some(usage.output);
    rec.estimated_cost = provider.estimated_cost(model, usage.input, usage.output);
    rec.success = error.is_none();
    rec.error = error;
}

fn finish(metrics: &Metrics, rec: DecisionRecord) {
    tracing::info!(
        target: "route",
        request_id = %rec.request_id, session_id = %rec.session_id, task_type = ?rec.task_type,
        previous_model = ?rec.previous_model, tier = %rec.selected_tier, model = ?rec.selected_model,
        reason = %rec.reason, latency_ms = ?rec.latency_ms, tokens_in = ?rec.tokens_input, tokens_out = ?rec.tokens_output,
        cost = ?rec.estimated_cost, success = rec.success, error = ?rec.error, fallback = rec.fallback_used,
        jev = %rec.jev_scores.as_ref().map(|v| v.to_string()).unwrap_or_default(),
        "routed"
    );
    metrics.insert(&rec);
}

/// Completes the decision record when a streamed response ends or is dropped (client cancellation).
struct FinishGuard {
    metrics: Arc<Metrics>,
    rec: DecisionRecord,
    provider: Arc<dyn LlmProvider>,
    model: String,
    usage: UsageCell,
    started: Instant,
    completed: bool,
    error: Option<String>,
}

impl Drop for FinishGuard {
    fn drop(&mut self) {
        let usage = *self.usage.lock().unwrap();
        let error = match (&self.error, self.completed) {
            (Some(e), _) => Some(e.clone()),
            (None, false) => Some("stream interrupted (client cancelled or upstream cut)".into()),
            (None, true) => None,
        };
        fill_outcome(&mut self.rec, &self.provider, &self.model, usage, self.started, error);
        finish(&self.metrics, std::mem::take(&mut self.rec));
    }
}

struct Tracked {
    inner: BoxStream<'static, Result<Bytes, ProviderError>>,
    guard: Option<FinishGuard>,
}

impl Stream for Tracked {
    type Item = Result<Bytes, ProviderError>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        match &poll {
            Poll::Ready(None) => {
                if let Some(g) = self.guard.as_mut() {
                    g.completed = true;
                }
            }
            Poll::Ready(Some(Err(e))) => {
                if let Some(g) = self.guard.as_mut() {
                    g.error = Some(e.to_string());
                }
            }
            _ => {}
        }
        poll
    }
}
