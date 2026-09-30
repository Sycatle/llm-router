use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LatencyStats {
    pub ema_ms: f64,
    pub samples: u64,
}

#[derive(Default)]
struct ModelStats {
    consecutive_errors: u32,
    cooldown_until: Option<Instant>,
    latency_ema_ms: Option<f64>,
    samples: u64,
}

/// In-memory per-model health: cooldown after failures and latency EMA.
#[derive(Default)]
pub struct HealthTracker {
    stats: Mutex<HashMap<String, ModelStats>>,
}

impl HealthTracker {
    pub fn record_success(&self, model: &str, latency: Duration) {
        let mut g = self.stats.lock().unwrap();
        let s = g.entry(model.to_string()).or_default();
        s.consecutive_errors = 0;
        s.cooldown_until = None;
        let ms = latency.as_secs_f64() * 1000.0;
        s.latency_ema_ms = Some(s.latency_ema_ms.map_or(ms, |e| 0.8 * e + 0.2 * ms));
        s.samples += 1;
    }

    /// `cooldown` is None for errors that do not indicate provider trouble (e.g. HTTP 400).
    pub fn record_failure(&self, model: &str, cooldown: Option<Duration>) {
        let Some(base) = cooldown else { return };
        let mut g = self.stats.lock().unwrap();
        let s = g.entry(model.to_string()).or_default();
        s.consecutive_errors += 1;
        let factor = 1u32 << (s.consecutive_errors - 1).min(4);
        s.cooldown_until = Some(Instant::now() + base * factor);
    }

    /// Skip `key` (a model id, or a provider id for account-wide limits) for `dur`.
    pub fn cooldown(&self, key: &str, dur: Duration) {
        let mut g = self.stats.lock().unwrap();
        g.entry(key.to_string()).or_default().cooldown_until = Some(Instant::now() + dur);
    }

    pub fn degraded(&self) -> HashSet<String> {
        let now = Instant::now();
        self.stats
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, s)| s.cooldown_until.is_some_and(|t| t > now))
            .map(|(k, _)| k.clone())
            .collect()
    }

    pub fn latency(&self, model: &str) -> Option<LatencyStats> {
        let g = self.stats.lock().unwrap();
        let s = g.get(model)?;
        Some(LatencyStats { ema_ms: s.latency_ema_ms?, samples: s.samples })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub request_id: String,
    pub session_id: String,
    pub ts_ms: u64,
    pub mode: String,
    pub task_type: Option<String>,
    pub jev_scores: Option<Value>,
    pub jev_error: Option<String>,
    pub previous_model: Option<String>,
    pub selected_tier: String,
    pub selected_model: Option<String>,
    pub reason: String,
    pub latency_ms: Option<u64>,
    pub tokens_input: Option<u64>,
    pub tokens_output: Option<u64>,
    pub estimated_cost: Option<f64>,
    pub success: bool,
    pub error: Option<String>,
    pub fallback_used: bool,
    pub attempts: Vec<String>,
}

pub struct Metrics {
    db: Mutex<Connection>,
}

impl Metrics {
    pub fn open(path: &str) -> rusqlite::Result<Self> {
        let conn = if path == ":memory:" { Connection::open_in_memory()? } else { Connection::open(path)? };
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                request_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                selected_tier TEXT NOT NULL,
                selected_model TEXT,
                success INTEGER NOT NULL,
                json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS decisions_session ON decisions(session_id);",
        )?;
        Ok(Self { db: Mutex::new(conn) })
    }

    pub fn insert(&self, r: &DecisionRecord) {
        let json = serde_json::to_string(r).unwrap_or_default();
        let res = self.db.lock().unwrap().execute(
            "INSERT INTO decisions (ts_ms, request_id, session_id, selected_tier, selected_model, success, json) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![r.ts_ms as i64, r.request_id, r.session_id, r.selected_tier, r.selected_model, r.success, json],
        );
        if let Err(e) = res {
            tracing::warn!(error = %e, "failed to persist decision");
        }
    }

    pub fn recent(&self, limit: usize) -> Vec<Value> {
        let db = self.db.lock().unwrap();
        let Ok(mut stmt) = db.prepare("SELECT json FROM decisions ORDER BY id DESC LIMIT ?1") else { return vec![] };
        stmt.query_map(params![limit as i64], |row| row.get::<_, String>(0))
            .map(|rows| rows.flatten().filter_map(|j| serde_json::from_str(&j).ok()).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooldown_marks_degraded_and_success_clears_it() {
        let h = HealthTracker::default();
        h.record_failure("a/b", Some(Duration::from_secs(30)));
        assert!(h.degraded().contains("a/b"));
        h.record_failure("x/y", None);
        assert!(!h.degraded().contains("x/y"));
        h.cooldown("prov", Duration::from_secs(60));
        assert!(h.degraded().contains("prov"));
        h.record_success("a/b", Duration::from_millis(100));
        assert_eq!(h.degraded().len(), 1, "only the provider-wide cooldown remains");
        assert_eq!(h.latency("a/b").unwrap().samples, 1);
    }

    #[test]
    fn decisions_roundtrip_newest_first() {
        let m = Metrics::open(":memory:").unwrap();
        for i in 0..3 {
            m.insert(&DecisionRecord { request_id: format!("r{i}"), selected_tier: "fast".into(), ..Default::default() });
        }
        let r = m.recent(2);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0]["request_id"], "r2");
    }
}
