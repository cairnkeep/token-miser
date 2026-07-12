use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub routing: RoutingConfig,
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<AuthConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private_cluster: Option<PrivateClusterConfig>,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub escalation: EscalationConfig,
    #[serde(default)]
    pub semantic_router: SemanticRouterConfig,
    #[serde(default)]
    pub fastcontext: FastContextConfig,
    #[serde(default)]
    pub explore: ExploreConfig,
}

/// Upstream repository-exploration stage. When enabled, before routing, a remote
/// FastContext model runs an agentic READ/GLOB/GREP loop (tools executed LOCALLY
/// against `explore.repo_root`) and the cleaned evidence is injected into the
/// request, shrinking the prompt the router then classifies. Backend-agnostic:
/// any OpenAI-compatible endpoint (cluster vLLM, local llama-server). FastContext
/// is never a routing target. Off by default.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct FastContextConfig {
    #[serde(default)]
    pub enabled: bool,
    /// OpenAI-compatible base (POSTs to `{endpoint_url}/chat/completions`).
    #[serde(default)]
    pub endpoint_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Bounds for the exploration loop and its evidence expansion. Defaults are kept
/// TIGHT on purpose: the stage returns little, targeted context — uncapped
/// expansion would reintroduce the prompt bloat the stage exists to remove.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExploreConfig {
    /// Root the local READ/GLOB/GREP tools are sandboxed to.
    #[serde(default = "default_repo_root")]
    pub repo_root: String,
    /// Hard cap on agentic turns. Some serving stacks serialize parallel tool
    /// calls, so leave headroom; on cap the loop returns best-effort evidence.
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    /// Hard cap on total expanded code lines across all citations.
    #[serde(default = "default_max_expanded_lines")]
    pub max_expanded_lines: usize,
    /// Hard cap on total expanded tokens (tiktoken cl100k) across all citations.
    #[serde(default = "default_max_expanded_tokens")]
    pub max_expanded_tokens: usize,
}

fn default_repo_root() -> String {
    ".".to_string()
}
fn default_max_turns() -> usize {
    16
}
fn default_max_expanded_lines() -> usize {
    200
}
fn default_max_expanded_tokens() -> usize {
    4000
}

impl Default for ExploreConfig {
    fn default() -> Self {
        Self {
            repo_root: default_repo_root(),
            max_turns: default_max_turns(),
            max_expanded_lines: default_max_expanded_lines(),
            max_expanded_tokens: default_max_expanded_tokens(),
        }
    }
}

/// Embedding-based routing. When enabled, the router embeds the request and
/// routes by cosine similarity to per-tier exemplar centroids instead of the
/// keyword/length heuristic. Falls back to the heuristic on any failure.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SemanticRouterConfig {
    #[serde(default)]
    pub enabled: bool,
    /// OpenAI-compatible embeddings endpoint base (POSTs to `{endpoint}/embeddings`).
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Quality-aware escalation. When the routed (cheaper) tier returns a transient
/// error or a degenerate response, retry on the next tier up, capped at Complex.
/// Off by default — when disabled, routing is single-shot as before.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EscalationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_escalations")]
    pub max_escalations: u8,
    #[serde(default = "default_true")]
    pub on_empty_response: bool,
    /// Escalate when the response was cut off (`finish_reason == "length"`).
    #[serde(default)]
    pub on_truncation: bool,
    /// Escalate when the response is a refusal / "I can't help" message.
    #[serde(default)]
    pub on_refusal: bool,
    /// Look-ahead heuristic buffer for STREAMED responses. A passthrough stream
    /// can't be escalated after the fact (its bytes are already on the client),
    /// so when enabled the proxy buffers the leading frames, runs the cheap
    /// degeneracy checks (empty/refusal/content_filter) on the prefix, and
    /// escalates BEFORE anything reaches the client if it's clearly failing.
    /// Adequate answers stream through after a short lead. Off by default → zero
    /// latency change. Does NOT enable the LLM judge (that needs the full text).
    #[serde(default)]
    pub stream_lookahead: bool,
    /// Chars of assistant text to buffer before the look-ahead verdict. Larger =
    /// stronger detection but more initial delay; ~240 (~60 tokens, <1s at 66 t/s).
    #[serde(default = "default_lookahead_chars")]
    pub lookahead_chars: usize,
    #[serde(default)]
    pub judge: JudgeConfig,
}

fn default_max_escalations() -> u8 {
    1
}
fn default_true() -> bool {
    true
}
fn default_lookahead_chars() -> usize {
    240
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_escalations: 1,
            on_empty_response: true,
            on_truncation: false,
            on_refusal: false,
            stream_lookahead: false,
            lookahead_chars: 240,
            judge: JudgeConfig::default(),
        }
    }
}

/// Optional LLM-judge layer: after the cheap tier returns a non-degenerate
/// response, a judge model scores how well it answers the request; a low score
/// triggers escalation. Catches plausible-but-wrong answers that heuristics miss.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JudgeConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Tier whose provider runs the judge (`free`/`standard`/`complex`).
    #[serde(default = "default_judge_tier")]
    pub tier: String,
    /// Escalate when the judge score (1-5) is below this threshold.
    #[serde(default = "default_min_score")]
    pub min_score: u8,
}

fn default_judge_tier() -> String {
    "standard".to_string()
}
fn default_min_score() -> u8 {
    3
}

impl Default for JudgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tier: "standard".to_string(),
            min_score: 3,
        }
    }
}

/// Routing/cost telemetry. Off by default; enable to log routing decisions and
/// estimated cost per request. `shadow_mode` routes every request to
/// `shadow_tier` (a safe baseline) while still recording the tier the router
/// *would* have chosen, so routing quality can be validated without risk.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub shadow_mode: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PrivateClusterConfig {
    pub endpoint: String,
    #[serde(default)]
    pub alternative_endpoints: Vec<String>,
    #[serde(default = "default_health_check")]
    pub health_check: String,
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_seconds: u64,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_seconds: u64,
    pub models: PrivateModels,
}

fn default_health_check() -> String {
    "/health".to_string()
}
fn default_connect_timeout() -> u64 {
    5
}
fn default_request_timeout() -> u64 {
    120
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PrivateModels {
    pub intent_classifier: String,
    pub standard: String,
    pub complex: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoutingConfig {
    pub tier1_threshold: usize,
    pub tier2_threshold: usize,
    pub complexity_keywords: Vec<String>,
    /// When true, only the explicit size/keyword rule may route to the Complex
    /// tier; semantic/intent picks of Complex are clamped to Standard, so the
    /// Complex tier is reached only by deliberate request. Default off.
    #[serde(default)]
    pub complex_requires_explicit: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    pub endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_type: Option<AuthType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_mapping: Option<HashMap<String, String>>,
    #[serde(default = "default_priority")]
    pub priority: u8,
    #[serde(default)]
    pub input_cost_per_1m: f64,
    #[serde(default)]
    pub output_cost_per_1m: f64,
    /// Extra JSON fields shallow-merged into the forwarded request body for this
    /// provider (provider values win over the client's). Use to force a local
    /// thinking-model into clean direct-answer mode, e.g.
    /// extra_body = { chat_template_kwargs = { enable_thinking = false } }.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,
}

fn default_priority() -> u8 {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AuthType {
    ApiKey,
    GitHubOAuth,
    None,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AuthConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub github_client_id: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            "tier1_free".to_string(),
            ProviderConfig {
                endpoint: "http://localhost:11434/v1".to_string(),
                api_key: None,
                auth_type: Some(AuthType::None),
                model_mapping: Some({
                    let mut map = HashMap::new();
                    map.insert("default".to_string(), "llama3.2".to_string());
                    map
                }),
                priority: 1,
                input_cost_per_1m: 0.0,
                output_cost_per_1m: 0.0,
                extra_body: None,
            },
        );

        providers.insert(
            "tier2_standard".to_string(),
            ProviderConfig {
                endpoint: "https://api.openai.com/v1".to_string(),
                api_key: std::env::var("OPENAI_API_KEY").ok(),
                auth_type: Some(AuthType::ApiKey),
                model_mapping: Some({
                    let mut map = HashMap::new();
                    map.insert("default".to_string(), "gpt-4o-mini".to_string());
                    map
                }),
                priority: 1,
                input_cost_per_1m: 0.15,
                output_cost_per_1m: 0.60,
                extra_body: None,
            },
        );

        providers.insert(
            "tier3_complex".to_string(),
            ProviderConfig {
                endpoint: "https://api.anthropic.com/v1".to_string(),
                api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
                auth_type: Some(AuthType::ApiKey),
                model_mapping: Some({
                    let mut map = HashMap::new();
                    map.insert(
                        "default".to_string(),
                        "claude-3-5-sonnet-20241022".to_string(),
                    );
                    map
                }),
                priority: 1,
                input_cost_per_1m: 3.0,
                output_cost_per_1m: 15.0,
                extra_body: None,
            },
        );

        Config {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
            },
            routing: RoutingConfig {
                tier1_threshold: 2000,
                tier2_threshold: 32000,
                // Kept tight: unambiguous complex-work phrases only. Single
                // ambiguous words (refactor, migrate) over-route trivial tasks to
                // Complex; semantic routing catches the genuine cases instead.
                complexity_keywords: vec![
                    "architect".to_string(),
                    "system design".to_string(),
                    "redesign".to_string(),
                ],
                complex_requires_explicit: false,
            },
            providers,
            auth: Some(AuthConfig {
                github_client_id: std::env::var("GITHUB_CLIENT_ID").ok(),
            }),
            private_cluster: None,
            telemetry: TelemetryConfig::default(),
            escalation: EscalationConfig::default(),
            semantic_router: SemanticRouterConfig::default(),
            fastcontext: FastContextConfig::default(),
            explore: ExploreConfig::default(),
        }
    }
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&content);
        let mut config: Config = toml::from_str(&expanded)?;
        // An unset ${VAR} expands to empty; treat an empty api_key as absent so
        // auth handling matches a key that was never configured.
        for provider in config.providers.values_mut() {
            if provider.api_key.as_deref() == Some("") {
                provider.api_key = None;
            }
        }
        Ok(config)
    }

    /// Applies per-launch environment overrides on top of the loaded config.
    /// Used by the per-project launcher (`cc-route-fc`) to bind one ephemeral
    /// token-miser instance to a single repo without writing a temp config:
    ///   TOKEN_MISER_PORT                 -> server.port
    ///   TOKEN_MISER_FASTCONTEXT_ENABLED  -> fastcontext.enabled (1/true/yes/on)
    ///   TOKEN_MISER_EXPLORE_REPO_ROOT    -> explore.repo_root
    pub fn apply_env_overrides(&mut self) {
        if let Ok(p) = std::env::var("TOKEN_MISER_PORT")
            && let Ok(port) = p.trim().parse::<u16>()
        {
            self.server.port = port;
        }
        if let Ok(v) = std::env::var("TOKEN_MISER_FASTCONTEXT_ENABLED") {
            self.fastcontext.enabled = matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
        if let Ok(r) = std::env::var("TOKEN_MISER_EXPLORE_REPO_ROOT")
            && !r.trim().is_empty()
        {
            self.explore.repo_root = r;
        }
    }
}

/// Expands `${VAR}` references from the environment. An unset variable expands
/// to an empty string; an unterminated `${` is left verbatim.
fn expand_env_vars(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find('}') {
            Some(end) => {
                out.push_str(&std::env::var(&after[..end]).unwrap_or_default());
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[start..]);
                return out;
            }
        }
    }

    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = Config::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_default_routing_config() {
        let config = Config::default();
        assert_eq!(config.routing.tier1_threshold, 2000);
        assert_eq!(config.routing.tier2_threshold, 32000);
        assert!(!config.routing.complexity_keywords.is_empty());
    }

    #[test]
    fn test_default_providers_exist() {
        let config = Config::default();
        assert!(config.providers.contains_key("tier1_free"));
        assert!(config.providers.contains_key("tier2_standard"));
        assert!(config.providers.contains_key("tier3_complex"));
    }

    #[test]
    fn test_committed_config_loads_and_matches_router_keys() {
        let config = Config::from_file("config.example.toml")
            .expect("config.example.toml should parse against the current structs");
        for key in ["tier1_free", "tier2_standard", "tier3_complex"] {
            assert!(config.providers.contains_key(key), "missing provider {key}");
        }
    }

    #[test]
    fn test_expand_env_vars() {
        assert_eq!(expand_env_vars("plain text"), "plain text");
        // Unset variables expand to empty.
        assert_eq!(
            expand_env_vars("key = \"${TOKEN_MISER_DEFINITELY_UNSET_XYZ}\""),
            "key = \"\""
        );
        // Unterminated placeholder is left verbatim.
        assert_eq!(expand_env_vars("a ${oops"), "a ${oops");
    }

    #[test]
    fn test_provider_endpoint_configuration() {
        let config = Config::default();
        let provider = config.providers.get("tier3_complex").unwrap();
        assert_eq!(provider.endpoint, "https://api.anthropic.com/v1");
    }

    #[test]
    fn test_model_mapping_default() {
        let config = Config::default();
        let provider = config.providers.get("tier2_standard").unwrap();
        let mapping = provider.model_mapping.as_ref().unwrap();
        assert!(mapping.contains_key("default"));
    }

    #[test]
    fn test_complexity_keywords() {
        let kws = Config::default().routing.complexity_keywords;
        for k in ["architect", "system design", "redesign"] {
            assert!(kws.contains(&k.to_string()), "missing keyword {k}");
        }
        // Ambiguous single words must NOT gate (semantic routing handles them).
        for k in ["refactor", "migrate"] {
            assert!(!kws.contains(&k.to_string()), "should not gate on {k}");
        }
    }

    #[test]
    fn test_explore_defaults_are_off_and_tight() {
        let config = Config::default();
        // The upstream explore stage must be off by default so existing behavior
        // is byte-identical until deliberately enabled.
        assert!(!config.fastcontext.enabled);
        // Expansion caps start tight (the whole point is little, targeted context).
        assert_eq!(config.explore.repo_root, ".");
        assert_eq!(config.explore.max_turns, 16);
        assert_eq!(config.explore.max_expanded_lines, 200);
        assert_eq!(config.explore.max_expanded_tokens, 4000);
    }

    #[test]
    fn test_config_without_explore_sections_parses() {
        // A config TOML with no [fastcontext]/[explore] tables must still load,
        // falling back to the off-by-default values.
        let toml = r#"
            [server]
            host = "127.0.0.1"
            port = 8080
            [routing]
            tier1_threshold = 2000
            tier2_threshold = 32000
            complexity_keywords = ["architect"]
            [providers.tier1_free]
            endpoint = "http://localhost:11434/v1"
        "#;
        let config: Config = toml::from_str(toml).expect("parses without explore tables");
        assert!(!config.fastcontext.enabled);
        assert_eq!(config.explore.max_turns, 16);
    }

    #[test]
    fn test_tier_thresholds() {
        let config = Config::default();
        assert!(config.routing.tier1_threshold < config.routing.tier2_threshold);
        assert_eq!(config.routing.tier1_threshold, 2000);
        assert_eq!(config.routing.tier2_threshold, 32000);
    }
}
