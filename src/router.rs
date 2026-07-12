use crate::cluster::IntentClassifierService;
use crate::config::RoutingConfig;
use crate::intent::{Intent, IntentClassifier};
use crate::models::ChatCompletionRequest;

#[derive(Debug, Clone, PartialEq)]
pub enum Tier {
    Free,
    Standard,
    Complex,
}

impl Tier {
    /// The provider config key this tier routes to.
    pub fn provider_key(&self) -> &'static str {
        match self {
            Tier::Free => "tier1_free",
            Tier::Standard => "tier2_standard",
            Tier::Complex => "tier3_complex",
        }
    }

    /// Parses a tier from a config string (`free`/`standard`/`complex`).
    pub fn from_name(name: &str) -> Option<Tier> {
        match name.to_lowercase().as_str() {
            "free" => Some(Tier::Free),
            "standard" => Some(Tier::Standard),
            "complex" => Some(Tier::Complex),
            _ => None,
        }
    }

    /// The next tier up for escalation, or `None` if already at the top.
    pub fn escalate(&self) -> Option<Tier> {
        match self {
            Tier::Free => Some(Tier::Standard),
            Tier::Standard => Some(Tier::Complex),
            Tier::Complex => None,
        }
    }
}

pub struct Router {
    config: RoutingConfig,
    tokenizer: tiktoken_rs::CoreBPE,
    intent_classifier: Option<IntentClassifierService>,
    fallback_classifier: IntentClassifier,
    semantic_router: Option<crate::semantic::SemanticRouter>,
}

impl Router {
    pub fn new(config: RoutingConfig) -> Self {
        let tokenizer = tiktoken_rs::cl100k_base().expect("Failed to load tokenizer");
        let fallback_classifier = IntentClassifier::new();
        Self {
            config,
            tokenizer,
            intent_classifier: None,
            fallback_classifier,
            semantic_router: None,
        }
    }

    pub fn with_cluster(mut self, cluster_config: crate::config::PrivateClusterConfig) -> Self {
        self.intent_classifier = Some(IntentClassifierService::new(cluster_config));
        self
    }

    pub fn with_semantic_router(mut self, config: crate::config::SemanticRouterConfig) -> Self {
        self.semantic_router = Some(crate::semantic::SemanticRouter::new(config));
        self
    }

    /// Reports private-cluster health, or `None` when no cluster is configured.
    pub async fn cluster_health(&self) -> Option<bool> {
        match &self.intent_classifier {
            Some(classifier) => Some(classifier.health_check().await),
            None => None,
        }
    }

    pub async fn classify(&self, request: &ChatCompletionRequest) -> Tier {
        let token_count = self.estimate_tokens(request);

        // Token/keyword rules force Complex outright, so skip intent
        // classification (a network round-trip when a cluster is configured).
        if token_count > self.config.tier2_threshold || self.check_keywords(request) {
            return Tier::Complex;
        }

        // Semantic routing replaces the keyword/length heuristic for the
        // ambiguous middle when enabled; it falls back on any embedding failure.
        let tier = if let Some(semantic) = &self.semantic_router
            && let Some(tier) = semantic.classify(request).await
        {
            tier
        } else {
            let intent = match &self.intent_classifier {
                Some(classifier) => classifier.classify(request).await,
                None => self.fallback_classifier.classify(request),
            };

            if token_count > self.config.tier1_threshold {
                match intent {
                    Intent::Agentic => Tier::Complex,
                    Intent::Standard | Intent::Trivial => Tier::Standard,
                }
            } else {
                match intent {
                    Intent::Agentic => Tier::Standard,
                    Intent::Standard | Intent::Trivial => Tier::Free,
                }
            }
        };

        // Governance gate: keep the non-deterministic classifiers (semantic /
        // intent) off the Complex tier. Only the explicit size/keyword rule
        // above may route there, so the Complex tier stays under the caller's
        // deliberate control. Default off; set routing.complex_requires_explicit.
        if self.config.complex_requires_explicit && tier == Tier::Complex {
            Tier::Standard
        } else {
            tier
        }
    }

    pub fn estimate_tokens(&self, request: &ChatCompletionRequest) -> usize {
        let mut total = 0;

        for message in &request.messages {
            let content = message.content.as_text();

            let tokens = self.tokenizer.encode_with_special_tokens(&content);
            total += tokens.len();

            if let Some(name) = &message.name {
                let name_tokens = self.tokenizer.encode_with_special_tokens(name);
                total += name_tokens.len();
            }

            if let Some(tool_calls) = &message.tool_calls {
                for tool_call in tool_calls {
                    let args_tokens = self
                        .tokenizer
                        .encode_with_special_tokens(&tool_call.function.arguments);
                    total += args_tokens.len();
                }
            }
        }

        if let Some(tools) = &request.tools {
            for tool in tools {
                let json = serde_json::to_string(&tool).unwrap_or_default();
                let tool_tokens = self.tokenizer.encode_with_special_tokens(&json);
                total += tool_tokens.len();
            }
        }

        total
    }

    fn check_keywords(&self, request: &ChatCompletionRequest) -> bool {
        for message in &request.messages {
            // Complexity keywords signal what the USER is asking for. Skip system
            // turns (the system prompt, and the FastContext explore stage's
            // injected code evidence) — otherwise a keyword that merely appears in
            // a cited code comment or system prompt would force the Complex tier.
            if message.role != "user" {
                continue;
            }
            let content = message.content.as_text().to_lowercase();

            for keyword in &self.config.complexity_keywords {
                if content.contains(&keyword.to_lowercase()) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Message, MessageContent};

    fn create_test_request(messages: Vec<Message>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            tools: None,
            tool_choice: None,
        }
    }

    fn text_message(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: MessageContent::Text(content.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    #[test]
    fn test_tier_escalation_path() {
        assert_eq!(Tier::Free.escalate(), Some(Tier::Standard));
        assert_eq!(Tier::Standard.escalate(), Some(Tier::Complex));
        assert_eq!(Tier::Complex.escalate(), None);
    }

    #[tokio::test]
    async fn test_short_request_routed_to_free() {
        let config = RoutingConfig {
            tier1_threshold: 2000,
            tier2_threshold: 32000,
            complexity_keywords: vec!["architect".to_string()],
            complex_requires_explicit: false,
        };
        let router = Router::new(config);

        let request = create_test_request(vec![text_message("Hello world")]);
        let tier = router.classify(&request).await;

        assert_eq!(tier, Tier::Free);
    }

    #[tokio::test]
    async fn test_keyword_triggers_complex() {
        let config = RoutingConfig {
            tier1_threshold: 2000,
            tier2_threshold: 32000,
            complexity_keywords: vec!["architect".to_string(), "refactor".to_string()],
            complex_requires_explicit: false,
        };
        let router = Router::new(config);

        let request = create_test_request(vec![text_message("Please architect this system")]);
        let tier = router.classify(&request).await;

        assert_eq!(tier, Tier::Complex);
    }

    #[tokio::test]
    async fn test_keyword_in_system_or_injected_evidence_does_not_trigger_complex() {
        // A complexity keyword appearing only in a system message (e.g. the
        // FastContext explore stage injecting code whose comments mention
        // "architect"/"redesign") must NOT force the Complex tier — only the
        // user's own ask should. The user turn here is a trivial lookup.
        let config = RoutingConfig {
            tier1_threshold: 2000,
            tier2_threshold: 32000,
            complexity_keywords: vec!["architect".to_string(), "redesign".to_string()],
            complex_requires_explicit: false,
        };
        let router = Router::new(config);

        let system = Message {
            role: "system".to_string(),
            content: MessageContent::Text(
                "## Repository context\n```\ncomplexity_keywords = [\"architect\", \"redesign\"]\n```"
                    .to_string(),
            ),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        };
        let request = create_test_request(vec![system, text_message("where is routing decided?")]);

        assert_ne!(router.classify(&request).await, Tier::Complex);
    }
}
