use crate::config::PrivateClusterConfig;
use crate::intent::Intent;
use crate::models::{ChatCompletionRequest, MessageContent};
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, error, info, warn};

pub struct IntentClassifierService {
    config: PrivateClusterConfig,
    client: Client,
    fallback_classifier: crate::intent::IntentClassifier,
}

impl IntentClassifierService {
    pub fn new(config: PrivateClusterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .connect_timeout(Duration::from_secs(config.connect_timeout_seconds))
            .build()
            .expect("Failed to create HTTP client for private cluster");

        Self {
            config,
            client,
            fallback_classifier: crate::intent::IntentClassifier::new(),
        }
    }

    pub async fn classify(&self, request: &ChatCompletionRequest) -> Intent {
        if let Ok(intent) = self.classify_via_cluster(request).await {
            info!("Classified intent via private cluster: {:?}", intent);
            return intent;
        }

        warn!("Private cluster unavailable, using fallback classifier");
        self.fallback_classifier.classify(request)
    }

    async fn classify_via_cluster(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Intent, Box<dyn std::error::Error>> {
        let prompt = self.build_classification_prompt(request);

        let classify_request = serde_json::json!({
            "model": self.config.models.intent_classifier,
            "messages": [
                {"role": "system", "content": "Classify the intent of this request into one of: agentic, standard, trivial. Reply with only the word."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        });

        let start = std::time::Instant::now();

        let response = self
            .client
            .post(format!("{}/chat/completions", self.config.endpoint))
            .json(&classify_request)
            .send()
            .await?;

        if !response.status().is_success() {
            error!("Private cluster returned status: {}", response.status());
            return Err("Cluster request failed".into());
        }

        let result: serde_json::Value = response.json().await?;
        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_lowercase();

        let elapsed = start.elapsed();
        debug!("Intent classification took {:?}", elapsed);

        if elapsed > Duration::from_millis(200) {
            warn!("Intent classification slow: {:?}", elapsed);
        }

        let intent = match content.trim() {
            "agentic" => Intent::Agentic,
            "trivial" => Intent::Trivial,
            "standard" => Intent::Standard,
            _ => self.fallback_classifier.classify(request),
        };

        Ok(intent)
    }

    fn build_classification_prompt(&self, request: &ChatCompletionRequest) -> String {
        let mut parts = Vec::new();

        parts.push(format!("Model requested: {}", request.model));

        let has_tools = request.tools.is_some();
        parts.push(format!("Has tool definitions: {}", has_tools));

        parts.push(format!("Number of messages: {}", request.messages.len()));

        let mut total_length = 0;
        let mut file_refs = 0;

        for msg in &request.messages {
            let content = msg.content.as_text();

            total_length += content.len();
            file_refs += content.matches("file://").count();
            file_refs += content.matches(".rs").count();
            file_refs += content.matches(".py").count();
        }

        parts.push(format!("Total content length: {} chars", total_length));
        parts.push(format!("File references: {}", file_refs));

        parts.push("Messages preview:".to_string());
        for (i, msg) in request.messages.iter().take(3).enumerate() {
            let content = match &msg.content {
                MessageContent::Text(text) => {
                    let mut preview: String = text.chars().take(200).collect();
                    if preview.len() < text.len() {
                        preview.push_str("...");
                    }
                    preview
                }
                MessageContent::Parts(_) => "[multimodal content]".to_string(),
            };
            parts.push(format!("  [{}] {}: {}", i, msg.role, content));
        }

        parts.join("\n")
    }

    pub async fn health_check(&self) -> bool {
        match self
            .client
            .get(format!(
                "{}{}",
                self.config.endpoint, self.config.health_check
            ))
            .send()
            .await
        {
            Ok(response) => {
                let healthy = response.status().is_success();
                if !healthy {
                    warn!("Private cluster health check failed: {}", response.status());
                }
                healthy
            }
            Err(e) => {
                error!("Private cluster health check error: {}", e);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PrivateModels;

    #[test]
    fn test_fallback_classifier() {
        let config = PrivateClusterConfig {
            endpoint: "http://invalid-endpoint".to_string(),
            alternative_endpoints: vec![],
            health_check: "/health".to_string(),
            connect_timeout_seconds: 1,
            request_timeout_seconds: 1,
            models: PrivateModels {
                intent_classifier: "test".to_string(),
                standard: "test".to_string(),
                complex: "test".to_string(),
            },
        };

        let _service = IntentClassifierService::new(config);
    }

    #[test]
    fn test_classification_prompt_handles_multibyte_preview() {
        let config = PrivateClusterConfig {
            endpoint: "http://invalid-endpoint".to_string(),
            alternative_endpoints: vec![],
            health_check: "/health".to_string(),
            connect_timeout_seconds: 1,
            request_timeout_seconds: 1,
            models: PrivateModels {
                intent_classifier: "test".to_string(),
                standard: "test".to_string(),
                complex: "test".to_string(),
            },
        };
        let service = IntentClassifierService::new(config);

        // 300 multi-byte chars: byte offset 200 falls mid-codepoint, which a
        // byte slice would panic on.
        let content = "あ".repeat(300);
        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![crate::models::Message {
                role: "user".to_string(),
                content: MessageContent::Text(content),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        let prompt = service.build_classification_prompt(&request);
        assert!(prompt.contains("..."));
    }
}
