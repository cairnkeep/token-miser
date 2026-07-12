use crate::config::Config;
use crate::router::Tier;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

/// One routing decision and its measured cost, emitted per request.
#[derive(Debug, Clone, Serialize)]
pub struct RoutingRecord {
    pub endpoint: &'static str,
    pub model: String,
    pub classified_tier: String,
    pub effective_tier: String,
    pub served_tier: String,
    pub escalations: u8,
    pub shadow: bool,
    pub stream: bool,
    pub input_tokens: u64,
    pub output_tokens: Option<u64>,
    pub estimated_cost_usd: Option<f64>,
    pub latency_ms: u64,
    /// Whether the served tier was the premium tier (Complex/Opus). The headline
    /// signal: the before/after Opus-escalation rate is the count of records with
    /// `premium_escalation == true` over total.
    pub premium_escalation: bool,
    /// Upstream FastContext explore-stage signals (all zero/false when the stage
    /// is disabled or skipped). `pre`/`post` token counts show how injecting clean
    /// context shifts the input-size distribution the router classifies.
    pub explore_ran: bool,
    pub pre_explore_input_tokens: u64,
    pub post_explore_input_tokens: u64,
    pub explore_turns: usize,
    pub explore_citations: usize,
    pub explore_expanded_tokens: usize,
}

/// Records routing decisions and estimated cost. Off unless `telemetry.enabled`.
#[derive(Clone)]
pub struct Telemetry {
    enabled: bool,
    shadow_mode: bool,
    shadow_tier: Option<Tier>,
    log_path: Option<String>,
}

impl Telemetry {
    pub fn from_config(config: &Config) -> Self {
        let cfg = &config.telemetry;
        let shadow_tier = cfg.shadow_tier.as_deref().and_then(Tier::from_name);
        if cfg.enabled && cfg.shadow_mode && shadow_tier.is_none() {
            warn!(
                "telemetry.shadow_mode is on but shadow_tier is unset/invalid; shadow routing disabled"
            );
        }
        Self {
            enabled: cfg.enabled,
            // Shadow routing only takes effect when telemetry is recording it,
            // so it can never silently override routing without a trace.
            shadow_mode: cfg.enabled && cfg.shadow_mode,
            shadow_tier,
            log_path: cfg.log_path.clone(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Whether the effective route is being overridden to the shadow baseline.
    pub fn is_shadow(&self) -> bool {
        self.shadow_mode && self.shadow_tier.is_some()
    }

    /// The tier to actually route to: the shadow baseline when shadow mode is
    /// active, otherwise the tier the router classified.
    pub fn effective_tier(&self, classified: &Tier) -> Tier {
        match (self.shadow_mode, &self.shadow_tier) {
            (true, Some(tier)) => tier.clone(),
            _ => classified.clone(),
        }
    }

    /// Estimates request cost in USD from the routed tier's configured pricing.
    pub fn cost_usd(
        &self,
        config: &Config,
        tier: &Tier,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Option<f64> {
        let provider = config.providers.get(tier.provider_key())?;
        let cost = (input_tokens as f64 / 1_000_000.0) * provider.input_cost_per_1m
            + (output_tokens as f64 / 1_000_000.0) * provider.output_cost_per_1m;
        Some(cost)
    }

    /// Emits the record to the `telemetry` tracing target and, if configured, a
    /// JSONL log file. Best-effort: file errors are logged, never propagated.
    pub async fn record(&self, record: &RoutingRecord) {
        if !self.enabled {
            return;
        }

        info!(
            target: "telemetry",
            endpoint = record.endpoint,
            model = %record.model,
            classified_tier = %record.classified_tier,
            effective_tier = %record.effective_tier,
            served_tier = %record.served_tier,
            escalations = record.escalations,
            shadow = record.shadow,
            stream = record.stream,
            input_tokens = record.input_tokens,
            output_tokens = ?record.output_tokens,
            estimated_cost_usd = ?record.estimated_cost_usd,
            latency_ms = record.latency_ms,
            premium_escalation = record.premium_escalation,
            explore_ran = record.explore_ran,
            pre_explore_input_tokens = record.pre_explore_input_tokens,
            post_explore_input_tokens = record.post_explore_input_tokens,
            explore_turns = record.explore_turns,
            explore_citations = record.explore_citations,
            explore_expanded_tokens = record.explore_expanded_tokens,
            "routing decision"
        );

        if let Some(path) = &self.log_path
            && let Err(e) = append_jsonl(path, record).await
        {
            warn!("failed to write telemetry log to {path}: {e}");
        }
    }
}

/// Wraps a streaming response body, passing bytes through unchanged while
/// scanning for the usage token count (`token_key`, e.g. `completion_tokens` for
/// OpenAI or `output_tokens` for Anthropic). When the stream ends it fills in
/// output tokens, cost, and total latency, then emits the telemetry record.
pub fn meter<S, E>(
    inner: S,
    telemetry: Telemetry,
    config: Arc<Config>,
    tier: Tier,
    record: RoutingRecord,
    token_key: &'static str,
    started: Instant,
) -> impl Stream<Item = Result<Bytes, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: Send + 'static,
{
    let state = MeterState {
        inner,
        tail: String::new(),
        tokens: None,
        telemetry,
        config,
        tier,
        record,
        token_key,
        started,
    };

    futures::stream::unfold(state, |mut st| async move {
        match st.inner.next().await {
            Some(item) => {
                if let Ok(bytes) = &item {
                    st.tail.push_str(&String::from_utf8_lossy(bytes));
                    if let Some(tokens) = extract_last_u64(&st.tail, st.token_key) {
                        st.tokens = Some(tokens);
                    }
                    // Bound memory while still spanning a usage frame that may be
                    // split across chunk boundaries.
                    if st.tail.len() > 4096 {
                        let keep_from = char_boundary(&st.tail, st.tail.len() - 2048);
                        st.tail.drain(..keep_from);
                    }
                }
                Some((item, st))
            }
            None => {
                st.record.latency_ms = st.started.elapsed().as_millis() as u64;
                if let Some(tokens) = st.tokens {
                    st.record.output_tokens = Some(tokens);
                    st.record.estimated_cost_usd =
                        st.telemetry
                            .cost_usd(&st.config, &st.tier, st.record.input_tokens, tokens);
                }
                st.telemetry.record(&st.record).await;
                None
            }
        }
    })
}

struct MeterState<S> {
    inner: S,
    tail: String,
    tokens: Option<u64>,
    telemetry: Telemetry,
    config: Arc<Config>,
    tier: Tier,
    record: RoutingRecord,
    token_key: &'static str,
    started: Instant,
}

/// Returns the integer following the last `"key":` occurrence in `haystack`.
fn extract_last_u64(haystack: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let mut result = None;
    let mut from = 0;
    while let Some(pos) = haystack[from..].find(&needle) {
        let value_start = from + pos + needle.len();
        let after = haystack[value_start..].trim_start_matches([':', ' ', '\t']);
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<u64>() {
            result = Some(n);
        }
        from = value_start;
    }
    result
}

fn char_boundary(s: &str, mut idx: usize) -> usize {
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

async fn append_jsonl(path: &str, record: &RoutingRecord) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let mut line = serde_json::to_string(record).map_err(std::io::Error::other)?;
    line.push('\n');

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(line.as_bytes()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_tier_respects_shadow() {
        let mut config = Config::default();
        config.telemetry.enabled = true;
        config.telemetry.shadow_mode = true;
        config.telemetry.shadow_tier = Some("complex".to_string());
        let telemetry = Telemetry::from_config(&config);

        assert!(telemetry.is_shadow());
        assert_eq!(telemetry.effective_tier(&Tier::Free), Tier::Complex);
    }

    #[test]
    fn test_effective_tier_passthrough_without_shadow() {
        let telemetry = Telemetry::from_config(&Config::default());

        assert!(!telemetry.is_shadow());
        assert_eq!(telemetry.effective_tier(&Tier::Free), Tier::Free);
    }

    #[test]
    fn test_extract_last_u64() {
        assert_eq!(
            extract_last_u64(
                r#"{"usage":{"completion_tokens": 42}}"#,
                "completion_tokens"
            ),
            Some(42)
        );
        // The last occurrence wins (message_start 0 -> message_delta 7).
        assert_eq!(
            extract_last_u64(
                r#""output_tokens":0 ... "output_tokens":7"#,
                "output_tokens"
            ),
            Some(7)
        );
        assert_eq!(
            extract_last_u64("no tokens here", "completion_tokens"),
            None
        );
    }

    #[test]
    fn test_cost_uses_tier_pricing() {
        let config = Config::default();
        let telemetry = Telemetry::from_config(&config);

        // tier2_standard (gpt-4o-mini): 0.15 in + 0.60 out per 1M tokens.
        let standard = telemetry
            .cost_usd(&config, &Tier::Standard, 1_000_000, 1_000_000)
            .unwrap();
        assert!((standard - 0.75).abs() < 1e-9);

        // tier1_free (local) costs nothing.
        let free = telemetry
            .cost_usd(&config, &Tier::Free, 1_000_000, 1_000_000)
            .unwrap();
        assert_eq!(free, 0.0);
    }
}
