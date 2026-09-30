use crate::router::signals::{Mode, Tier};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    pub server: ServerConfig,
    pub routing: RoutingConfig,
    pub jev: JevConfig,
    pub storage: StorageConfig,
    pub providers: HashMap<String, ProviderConfig>,
    pub models: HashMap<String, ModelConfig>,
    pub tiers: HashMap<String, TierConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub listen: String,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub default_mode: String,
    /// Distance (in score units) beyond the current tier band required to switch tier.
    pub stickiness: f64,
    /// Larger distance required when the switch would reverse a recent one (anti-flapping).
    pub switch_threshold: f64,
    /// Below this confidence the policy holds the current tier (or STANDARD).
    pub min_confidence: f64,
    /// Score boundaries: fast|standard, standard|reasoning, reasoning|frontier.
    pub thresholds: [f64; 3],
    pub session_ttl_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct JevConfig {
    pub api_base: String,
    pub api_key_env: String,
    pub model: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub db_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// "anthropic" or "openai" (any OpenAI-compatible endpoint).
    pub kind: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    /// anthropic only: "api_key" (default) or "oauth_opencode" (reads opencode's auth.json).
    pub auth: Option<String>,
    pub auth_file: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub upstream: String,
    pub context_window: u32,
    #[serde(default = "yes")]
    pub tools: bool,
    #[serde(default)]
    pub price_in: f64,
    #[serde(default)]
    pub price_out: f64,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TierConfig {
    pub models: Vec<String>,
}

fn yes() -> bool {
    true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { listen: "127.0.0.1:8787".into(), request_timeout_secs: 60 }
    }
}
impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_mode: "auto".into(),
            stickiness: 0.15,
            switch_threshold: 0.20,
            min_confidence: 0.40,
            thresholds: [0.30, 0.55, 0.80],
            session_ttl_secs: 6 * 3600,
        }
    }
}
impl Default for JevConfig {
    fn default() -> Self {
        Self {
            api_base: "https://api.typesafe.ai".into(),
            api_key_env: "TYPESAFE_API_KEY".into(),
            model: "jev-latest".into(),
            timeout_ms: 1500,
        }
    }
}
impl Default for StorageConfig {
    fn default() -> Self {
        Self { db_path: "router.db".into() }
    }
}

/// Model as seen by the routing logic: no provider-specific knowledge.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// "provider/name"
    pub id: String,
    pub provider: String,
    pub upstream: String,
    pub context_window: u32,
    pub tools: bool,
    pub price_in: f64,
    pub price_out: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub models: HashMap<String, ModelSpec>,
    pub tiers: BTreeMap<Tier, Vec<String>>,
}

impl Catalog {
    pub fn tier_of(&self, model_id: &str) -> Option<Tier> {
        self.tiers.iter().find(|(_, ms)| ms.iter().any(|m| m == model_id)).map(|(t, _)| *t)
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        cfg.catalog()?;
        Ok(cfg)
    }

    pub fn default_mode(&self) -> Mode {
        Mode::parse(&self.routing.default_mode).unwrap_or(Mode::Auto)
    }

    pub fn catalog(&self) -> Result<Catalog, String> {
        let mut cat = Catalog::default();
        for (id, m) in &self.models {
            let (provider, _) = id.split_once('/').ok_or(format!("model id '{id}' must be 'provider/name'"))?;
            if !self.providers.contains_key(provider) {
                return Err(format!("model '{id}': unknown provider '{provider}'"));
            }
            cat.models.insert(
                id.clone(),
                ModelSpec {
                    id: id.clone(),
                    provider: provider.to_string(),
                    upstream: m.upstream.clone(),
                    context_window: m.context_window,
                    tools: m.tools,
                    price_in: m.price_in,
                    price_out: m.price_out,
                },
            );
        }
        for (name, t) in &self.tiers {
            let tier = match Mode::parse(name) {
                Some(Mode::Tier(t)) => t,
                _ => return Err(format!("unknown tier '{name}'")),
            };
            for m in &t.models {
                if !cat.models.contains_key(m) {
                    return Err(format!("tier '{name}': model '{m}' is not defined in [models]"));
                }
            }
            cat.tiers.insert(tier, t.models.clone());
        }
        if cat.tiers.values().all(|v| v.is_empty()) {
            return Err("no model configured in any tier".into());
        }
        Ok(cat)
    }
}
