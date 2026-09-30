//! Deterministic policy: signals + session + catalog + health -> tier -> ordered candidates.
//! No I/O, no clock, no randomness.

use super::signals::{ContextSize, Mode, RoutingSignals, TaskType, Tier};
use crate::config::{Catalog, ModelSpec, RoutingConfig};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SessionView {
    pub tier: Tier,
    pub model: String,
    /// Tiers actually used by the last routed requests, oldest first.
    pub history: Vec<Tier>,
}

pub struct PolicyInput<'a> {
    pub mode: &'a Mode,
    /// None when the request is a tool-loop continuation (classification skipped).
    pub signals: Option<&'a RoutingSignals>,
    pub continuation: bool,
    pub session: Option<&'a SessionView>,
    pub est_tokens: u64,
    pub needs_tools: bool,
    /// Forced tier floor ("force an upgrade").
    pub min_tier: Option<Tier>,
    /// Models in cooldown / with recent errors: tried last.
    pub degraded: &'a HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Candidate {
    pub model: String,
    pub tier: Tier,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    pub tier: Tier,
    pub score: Option<f64>,
    /// Ordered: same tier first, then higher tiers, degraded models last.
    pub candidates: Vec<Candidate>,
    pub reason: String,
}

pub struct Policy {
    cfg: RoutingConfig,
    catalog: Catalog,
}

const OUTPUT_RESERVE_TOKENS: u64 = 1024;

/// Weighted complexity score in [0, 1].
pub fn score(s: &RoutingSignals) -> f64 {
    let mut v = 0.40 * s.reasoning + 0.30 * s.complexity + 0.15 * s.tool_intensity + 0.15 * s.ambiguity;
    v += match s.task_type {
        TaskType::Architecture => 0.10,
        TaskType::Debugging | TaskType::Refactor => 0.05,
        TaskType::Question | TaskType::RepoSearch | TaskType::SmallEdit | TaskType::Docs => -0.05,
        _ => 0.0,
    };
    v += match s.context_size {
        ContextSize::Small | ContextSize::Medium => 0.0,
        ContextSize::Large => 0.05,
        ContextSize::Huge => 0.15,
    };
    v -= 0.10 * s.latency_sensitivity;
    v.clamp(0.0, 1.0)
}

impl Policy {
    pub fn new(cfg: RoutingConfig, catalog: Catalog) -> Self {
        Self { cfg, catalog }
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    fn band(&self, t: Tier) -> (f64, f64) {
        let th = self.cfg.thresholds;
        match t {
            Tier::Fast => (f64::NEG_INFINITY, th[0]),
            Tier::Standard => (th[0], th[1]),
            Tier::Reasoning => (th[1], th[2]),
            Tier::Frontier => (th[2], f64::INFINITY),
        }
    }

    pub fn tier_of_score(&self, s: f64) -> Tier {
        Tier::ALL.into_iter().find(|t| s < self.band(*t).1).unwrap_or(Tier::Frontier)
    }

    fn distance_from_band(&self, s: f64, t: Tier) -> f64 {
        let (lo, hi) = self.band(t);
        if s < lo {
            lo - s
        } else if s >= hi {
            s - hi
        } else {
            0.0
        }
    }

    pub fn decide(&self, i: &PolicyInput) -> Decision {
        if let Mode::ForceModel(id) = i.mode {
            let tier = self.catalog.tier_of(id).unwrap_or(Tier::Standard);
            let candidates = if self.catalog.models.contains_key(id) {
                vec![Candidate { model: id.clone(), tier }]
            } else {
                vec![]
            };
            return Decision { tier, score: None, candidates, reason: format!("forced model {id}") };
        }

        let (mut tier, mut reason, sc) = match i.mode {
            Mode::Tier(t) => (*t, format!("forced tier {}", t.as_str()), None),
            _ => self.auto_tier(i),
        };
        if let Some(floor) = i.min_tier
            && tier < floor {
                tier = floor;
                reason = format!("{reason}; raised to floor {}", floor.as_str());
            }
        Decision { tier, score: sc, candidates: self.candidates(tier, i), reason }
    }

    fn auto_tier(&self, i: &PolicyInput) -> (Tier, String, Option<f64>) {
        if i.continuation
            && let Some(s) = i.session {
                return (s.tier, "tool-loop continuation: keep tier".into(), None);
            }
        let signals = match i.signals {
            Some(s) if s.confidence >= self.cfg.min_confidence => s,
            other => {
                let why = if other.is_some() { "low confidence" } else { "no signals" };
                return match i.session {
                    Some(s) => (s.tier, format!("{why}: hold {}", s.tier.as_str()), None),
                    None => (Tier::Standard, format!("{why}: default standard"), None),
                };
            }
        };
        let sc = score(signals);
        let target = self.tier_of_score(sc);
        let Some(sess) = i.session else {
            return (target, format!("new session, score {sc:.2}"), Some(sc));
        };
        if target == sess.tier {
            return (sess.tier, format!("score {sc:.2} within {}", sess.tier.as_str()), Some(sc));
        }
        let flapping = sess.history.contains(&target);
        let margin = if flapping { self.cfg.switch_threshold } else { self.cfg.stickiness };
        let dist = self.distance_from_band(sc, sess.tier);
        if dist >= margin {
            (target, format!("score {sc:.2} is {dist:.2} outside {} (margin {margin:.2}): switch", sess.tier.as_str()), Some(sc))
        } else {
            (
                sess.tier,
                format!("score {sc:.2} is {dist:.2} outside {} (margin {margin:.2}{}): stay", sess.tier.as_str(), if flapping { ", anti-flap" } else { "" }),
                Some(sc),
            )
        }
    }

    fn compatible(&self, m: &ModelSpec, i: &PolicyInput) -> bool {
        m.context_window as u64 >= i.est_tokens + OUTPUT_RESERVE_TOKENS && (!i.needs_tools || m.tools)
    }

    fn candidates(&self, start: Tier, i: &PolicyInput) -> Vec<Candidate> {
        let current_provider = i.session.map(|s| s.model.split('/').next().unwrap_or("").to_string());
        let mut seen = HashSet::new();
        let (mut healthy, mut degraded) = (vec![], vec![]);
        for tier in Tier::ALL.into_iter().filter(|t| *t >= start) {
            let mut models: Vec<&ModelSpec> = self
                .catalog
                .tiers
                .get(&tier)
                .into_iter()
                .flatten()
                .filter_map(|id| self.catalog.models.get(id))
                .filter(|m| self.compatible(m, i))
                .collect();
            if tier == start
                && let (Some(s), Some(p)) = (i.session, &current_provider) {
                    models.sort_by_key(|m| if m.id == s.model { 0 } else if &m.provider == p { 1 } else { 2 });
                }
            for m in models {
                if !seen.insert(m.id.clone()) {
                    continue;
                }
                let c = Candidate { model: m.id.clone(), tier };
                if i.degraded.contains(&m.id) { degraded.push(c) } else { healthy.push(c) }
            }
        }
        healthy.extend(degraded);
        healthy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn policy() -> Policy {
        let cfg: Config = toml::from_str(
            r#"
            [providers.openai]
            kind = "openai"
            [providers.mistral]
            kind = "openai"
            [providers.anthropic]
            kind = "anthropic"
            [models."openai/fast"]
            upstream = "f"
            context_window = 128000
            [models."mistral/cheap"]
            upstream = "c"
            context_window = 32000
            tools = false
            [models."anthropic/sonnet"]
            upstream = "s"
            context_window = 200000
            [models."openai/reasoning"]
            upstream = "r"
            context_window = 200000
            [models."anthropic/opus"]
            upstream = "o"
            context_window = 200000
            [tiers.fast]
            models = ["mistral/cheap", "openai/fast"]
            [tiers.standard]
            models = ["anthropic/sonnet"]
            [tiers.reasoning]
            models = ["openai/reasoning", "anthropic/sonnet"]
            [tiers.frontier]
            models = ["anthropic/opus"]
            "#,
        )
        .unwrap();
        Policy::new(cfg.routing.clone(), cfg.catalog().unwrap())
    }

    fn sig(task: TaskType, c: f64, r: f64, tool: f64, amb: f64, ctx: ContextSize) -> RoutingSignals {
        RoutingSignals {
            task_type: task,
            complexity: c,
            reasoning: r,
            tool_intensity: tool,
            latency_sensitivity: 0.3,
            ambiguity: amb,
            context_size: ctx,
            confidence: 0.9,
        }
    }
    fn trivial() -> RoutingSignals {
        sig(TaskType::SmallEdit, 0.05, 0.05, 0.1, 0.0, ContextSize::Small)
    }
    fn feature() -> RoutingSignals {
        sig(TaskType::Feature, 0.5, 0.45, 0.4, 0.1, ContextSize::Medium)
    }
    fn hard_bug() -> RoutingSignals {
        sig(TaskType::Debugging, 0.78, 0.84, 0.55, 0.3, ContextSize::Large)
    }
    fn big_arch() -> RoutingSignals {
        sig(TaskType::Architecture, 0.95, 0.95, 0.7, 0.6, ContextSize::Huge)
    }

    struct Env {
        degraded: HashSet<String>,
        mode: Mode,
        session: Option<SessionView>,
        est_tokens: u64,
        needs_tools: bool,
        min_tier: Option<Tier>,
    }
    impl Env {
        fn new() -> Self {
            Env { degraded: HashSet::new(), mode: Mode::Auto, session: None, est_tokens: 1000, needs_tools: true, min_tier: None }
        }
        fn run(&self, p: &Policy, s: Option<&RoutingSignals>) -> Decision {
            p.decide(&PolicyInput {
                mode: &self.mode,
                signals: s,
                continuation: false,
                session: self.session.as_ref(),
                est_tokens: self.est_tokens,
                needs_tools: self.needs_tools,
                min_tier: self.min_tier,
                degraded: &self.degraded,
            })
        }
        fn in_session(mut self, tier: Tier, model: &str, history: &[Tier]) -> Self {
            self.session = Some(SessionView { tier, model: model.into(), history: history.to_vec() });
            self
        }
    }

    fn first(d: &Decision) -> &str {
        &d.candidates[0].model
    }

    #[test]
    fn trivial_prompt_is_fast() {
        let d = Env::new().run(&policy(), Some(&trivial()));
        assert_eq!(d.tier, Tier::Fast);
        assert_eq!(first(&d), "openai/fast"); // mistral/cheap has no tools
    }

    #[test]
    fn standard_feature_is_standard() {
        let d = Env::new().run(&policy(), Some(&feature()));
        assert_eq!(d.tier, Tier::Standard);
        assert_eq!(first(&d), "anthropic/sonnet");
    }

    #[test]
    fn complex_bug_is_reasoning() {
        let d = Env::new().run(&policy(), Some(&hard_bug()));
        assert_eq!(d.tier, Tier::Reasoning);
        assert_eq!(first(&d), "openai/reasoning");
    }

    #[test]
    fn big_architecture_is_frontier() {
        let d = Env::new().run(&policy(), Some(&big_arch()));
        assert_eq!(d.tier, Tier::Frontier);
        assert_eq!(first(&d), "anthropic/opus");
    }

    #[test]
    fn classifier_unavailable_defaults_to_standard() {
        let unknown = RoutingSignals::unknown(ContextSize::Small);
        let d = Env::new().run(&policy(), Some(&unknown));
        assert_eq!(d.tier, Tier::Standard);
        let d = Env::new().run(&policy(), None);
        assert_eq!(d.tier, Tier::Standard);
    }

    #[test]
    fn classifier_unavailable_holds_current_tier() {
        let unknown = RoutingSignals::unknown(ContextSize::Small);
        let d = Env::new().in_session(Tier::Reasoning, "openai/reasoning", &[Tier::Reasoning]).run(&policy(), Some(&unknown));
        assert_eq!(d.tier, Tier::Reasoning);
    }

    #[test]
    fn degraded_provider_is_tried_last() {
        let mut env = Env::new();
        env.degraded.insert("openai/fast".into());
        env.needs_tools = false;
        let d = env.run(&policy(), Some(&trivial()));
        assert_eq!(first(&d), "mistral/cheap");
        assert_eq!(d.candidates.last().unwrap().model, "openai/fast");
    }

    #[test]
    fn fallback_chain_escalates_through_higher_tiers_without_duplicates() {
        let d = Env::new().run(&policy(), Some(&trivial()));
        let ids: Vec<_> = d.candidates.iter().map(|c| c.model.as_str()).collect();
        assert_eq!(ids, ["openai/fast", "anthropic/sonnet", "openai/reasoning", "anthropic/opus"]);
        assert_eq!(d.candidates[1].tier, Tier::Standard);
    }

    #[test]
    fn session_stickiness_keeps_tier_for_small_score_change() {
        let p = policy();
        // feature score ~0.39; in a FAST session the distance to the band is ~0.09 < 0.15
        let d = Env::new().in_session(Tier::Fast, "openai/fast", &[Tier::Fast]).run(&p, Some(&feature()));
        assert_eq!(d.tier, Tier::Fast, "{}", d.reason);
    }

    #[test]
    fn large_gap_switches_tier() {
        let d = Env::new().in_session(Tier::Fast, "openai/fast", &[Tier::Fast]).run(&policy(), Some(&hard_bug()));
        assert_eq!(d.tier, Tier::Reasoning);
    }

    #[test]
    fn switch_threshold_blocks_flapping() {
        let p = policy();
        // score just outside the STANDARD band (0.30): dist 0.17 -> passes stickiness (0.15)...
        let mut s = trivial();
        s.complexity = 0.0;
        s.reasoning = 0.0;
        s.ambiguity = 0.0;
        s.tool_intensity = 0.0;
        s.task_type = TaskType::Other;
        s.latency_sensitivity = 0.0;
        // score 0.0 -> dist from standard band = 0.30: switch when no flap history
        let d = Env::new().in_session(Tier::Standard, "anthropic/sonnet", &[Tier::Standard]).run(&p, Some(&s));
        assert_eq!(d.tier, Tier::Fast);
        // tune to dist 0.17: score 0.13
        s.reasoning = 0.325; // 0.4 * 0.325 = 0.13
        let d = Env::new().in_session(Tier::Standard, "anthropic/sonnet", &[Tier::Standard]).run(&p, Some(&s));
        assert_eq!(d.tier, Tier::Fast, "{}", d.reason);
        // same score, but FAST was used recently (FAST -> STANDARD -> ?): needs 0.20, only 0.17 -> stay
        let d = Env::new().in_session(Tier::Standard, "anthropic/sonnet", &[Tier::Fast, Tier::Standard]).run(&p, Some(&s));
        assert_eq!(d.tier, Tier::Standard, "{}", d.reason);
        assert!(d.reason.contains("anti-flap"));
    }

    #[test]
    fn continuation_keeps_tier() {
        let p = policy();
        let env = Env::new().in_session(Tier::Reasoning, "openai/reasoning", &[Tier::Reasoning]);
        let d = p.decide(&PolicyInput {
            mode: &env.mode,
            signals: None,
            continuation: true,
            session: env.session.as_ref(),
            est_tokens: 1000,
            needs_tools: true,
            min_tier: None,
            degraded: &env.degraded,
        });
        assert_eq!(d.tier, Tier::Reasoning);
        assert_eq!(first(&d), "openai/reasoning");
    }

    #[test]
    fn provider_stickiness_prefers_current_model_in_tier() {
        let env = Env::new().in_session(Tier::Reasoning, "anthropic/sonnet", &[Tier::Reasoning]);
        let d = env.run(&policy(), Some(&hard_bug()));
        assert_eq!(d.tier, Tier::Reasoning);
        assert_eq!(first(&d), "anthropic/sonnet");
    }

    #[test]
    fn force_model_bypasses_signals() {
        let mut env = Env::new();
        env.mode = Mode::ForceModel("anthropic/opus".into());
        let d = env.run(&policy(), Some(&trivial()));
        assert_eq!(d.candidates.len(), 1);
        assert_eq!(first(&d), "anthropic/opus");
        env.mode = Mode::ForceModel("nope/x".into());
        assert!(env.run(&policy(), None).candidates.is_empty());
    }

    #[test]
    fn forced_tier_and_min_tier() {
        let mut env = Env::new();
        env.mode = Mode::Tier(Tier::Reasoning);
        assert_eq!(env.run(&policy(), Some(&trivial())).tier, Tier::Reasoning);
        let mut env = Env::new();
        env.min_tier = Some(Tier::Reasoning);
        assert_eq!(env.run(&policy(), Some(&trivial())).tier, Tier::Reasoning);
    }

    #[test]
    fn context_too_large_for_small_models_skips_them() {
        let mut env = Env::new();
        env.needs_tools = false;
        env.est_tokens = 100_000; // mistral/cheap (32k) out, openai/fast (128k) ok
        let d = env.run(&policy(), Some(&trivial()));
        assert_eq!(first(&d), "openai/fast");
        env.est_tokens = 150_000; // both fast models out -> escalates
        let d = env.run(&policy(), Some(&trivial()));
        assert_eq!(first(&d), "anthropic/sonnet");
        env.est_tokens = 500_000;
        assert!(env.run(&policy(), Some(&trivial())).candidates.is_empty());
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(Mode::parse("auto"), Some(Mode::Auto));
        assert_eq!(Mode::parse("auto-frontier"), Some(Mode::Tier(Tier::Frontier)));
        assert_eq!(Mode::parse("anthropic/opus"), None);
    }
}
