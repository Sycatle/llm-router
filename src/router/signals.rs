use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Fast,
    Standard,
    Reasoning,
    Frontier,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Fast, Tier::Standard, Tier::Reasoning, Tier::Frontier];

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Fast => "fast",
            Tier::Standard => "standard",
            Tier::Reasoning => "reasoning",
            Tier::Frontier => "frontier",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Question,
    RepoSearch,
    SmallEdit,
    Feature,
    Debugging,
    Refactor,
    Architecture,
    Tests,
    Docs,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSize {
    Small,
    Medium,
    Large,
    Huge,
}

impl ContextSize {
    pub fn from_tokens(tokens: u64) -> Self {
        match tokens {
            0..=7_999 => ContextSize::Small,
            8_000..=39_999 => ContextSize::Medium,
            40_000..=119_999 => ContextSize::Large,
            _ => ContextSize::Huge,
        }
    }
}

/// Structured output of a classifier. Never contains a model name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingSignals {
    pub task_type: TaskType,
    pub complexity: f64,
    pub reasoning: f64,
    pub tool_intensity: f64,
    pub latency_sensitivity: f64,
    #[serde(default)]
    pub ambiguity: f64,
    pub context_size: ContextSize,
    pub confidence: f64,
}

impl RoutingSignals {
    /// Neutral signals used when the classifier is unavailable. Confidence 0
    /// makes the policy fall back to STANDARD (or hold the current tier).
    pub fn unknown(context_size: ContextSize) -> Self {
        Self {
            task_type: TaskType::Other,
            complexity: 0.5,
            reasoning: 0.5,
            tool_intensity: 0.5,
            latency_sensitivity: 0.5,
            ambiguity: 0.0,
            context_size,
            confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Auto,
    Tier(Tier),
    ForceModel(String),
}

impl Mode {
    /// "auto", "auto-fast", ... or "fast"; anything else is None.
    pub fn parse(s: &str) -> Option<Mode> {
        let s = s.trim().to_ascii_lowercase();
        let s = s.strip_prefix("auto-").unwrap_or(&s);
        match s {
            "auto" => Some(Mode::Auto),
            "fast" => Some(Mode::Tier(Tier::Fast)),
            "standard" => Some(Mode::Tier(Tier::Standard)),
            "reasoning" => Some(Mode::Tier(Tier::Reasoning)),
            "frontier" => Some(Mode::Tier(Tier::Frontier)),
            _ => None,
        }
    }
}
