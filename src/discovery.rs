use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotModel {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    #[serde(default)]
    pub billing: ModelBilling,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub supports_streaming: bool,
    #[serde(default)]
    pub supports_tools: bool,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelBilling {
    #[serde(default)]
    pub tier: String,
}

#[derive(Debug, Clone)]
pub struct ModelDiscovery {
    client: Client,
    cached_models: HashMap<String, Vec<CopilotModel>>,
}

impl ModelDiscovery {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            cached_models: HashMap::new(),
        }
    }

    pub async fn discover_copilot_models(
        &mut self,
    ) -> Result<Vec<CopilotModel>, Box<dyn std::error::Error>> {
        if let Some(models) = self.cached_models.get("github_copilot") {
            return Ok(models.clone());
        }

        info!("Discovering GitHub Copilot models...");

        // Use GitHub Copilot CLI to list models (requires authentication)
        let output = tokio::process::Command::new("gh")
            .arg("copilot")
            .arg("list-models")
            .arg("--json")
            .output()
            .await;

        let models = match output {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                match serde_json::from_str::<Vec<CopilotModel>>(&stdout) {
                    Ok(models) => {
                        info!("Discovered {} GitHub Copilot models", models.len());
                        models
                    }
                    Err(e) => {
                        warn!("Failed to parse Copilot models: {}", e);
                        Self::get_default_copilot_models()
                    }
                }
            }
            _ => {
                warn!("GitHub CLI not available or user not authenticated");
                Self::get_default_copilot_models()
            }
        };

        self.cached_models
            .insert("github_copilot".to_string(), models.clone());
        Ok(models)
    }

    pub async fn discover_claude_models(
        &mut self,
    ) -> Result<Vec<CopilotModel>, Box<dyn std::error::Error>> {
        if let Some(models) = self.cached_models.get("claude_max") {
            return Ok(models.clone());
        }

        info!("Discovering Claude models...");

        // Try multiple discovery methods
        let models = self
            .try_claude_discovery()
            .await
            .unwrap_or_else(Self::get_default_claude_models);

        info!("Discovered {} Claude models", models.len());
        self.cached_models
            .insert("claude_max".to_string(), models.clone());
        Ok(models)
    }

    async fn try_claude_discovery(&self) -> Option<Vec<CopilotModel>> {
        // Method 1: Check Claude settings.json
        if let Some(models) = self.discover_from_claude_settings().await {
            return Some(models);
        }

        // Method 2: Query Anthropic API (if API key available)
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY")
            && let Some(models) = self.discover_from_anthropic_api(&api_key).await
        {
            return Some(models);
        }

        None
    }

    async fn discover_from_claude_settings(&self) -> Option<Vec<CopilotModel>> {
        let settings_path = dirs::home_dir().map(|h| h.join(".claude/settings.json"))?;

        if !settings_path.exists() {
            return None;
        }

        let content = tokio::fs::read_to_string(&settings_path).await.ok()?;

        #[derive(Deserialize)]
        struct ClaudeSettings {
            model: Option<String>,
            env: Option<serde_json::Map<String, serde_json::Value>>,
        }

        let settings: ClaudeSettings = serde_json::from_str(&content).ok()?;

        let mut models = Vec::new();

        // Check default model
        if let Some(default_model) = settings.model {
            models.push(CopilotModel {
                id: Self::resolve_claude_model_alias(&default_model),
                name: default_model.clone(),
                capabilities: ModelCapabilities::default(),
                billing: ModelBilling {
                    tier: "subscription".to_string(),
                },
            });
        }

        // Check env variables for model configuration
        if let Some(env) = settings.env {
            for key in &[
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
            ] {
                if let Some(model) = env.get(*key).and_then(|v| v.as_str()) {
                    models.push(CopilotModel {
                        id: model.to_string(),
                        name: model.to_string(),
                        capabilities: ModelCapabilities::default(),
                        billing: ModelBilling::default(),
                    });
                }
            }
        }

        if models.is_empty() {
            None
        } else {
            Some(models)
        }
    }

    fn resolve_claude_model_alias(alias: &str) -> String {
        match alias.to_lowercase().as_str() {
            "sonnet" | "claude-sonnet" | "sonnet-4" | "claude-sonnet-4" => {
                "claude-sonnet-4-7".to_string()
            }
            "opus" | "claude-opus" | "opus-4" | "claude-opus-4" => "claude-opus-4-8".to_string(),
            "haiku" | "claude-haiku" | "haiku-4" | "claude-haiku-4" => {
                "claude-haiku-4-5".to_string()
            }
            alias => alias.to_string(),
        }
    }

    async fn discover_from_anthropic_api(&self, api_key: &str) -> Option<Vec<CopilotModel>> {
        let client = reqwest::Client::new();

        let response = client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            id: String,
            #[serde(default)]
            display_name: Option<String>,
        }

        let models_response: ModelsResponse = response.json().await.ok()?;

        let models: Vec<CopilotModel> = models_response
            .data
            .into_iter()
            .map(|m| CopilotModel {
                id: m.id.clone(),
                name: m.display_name.unwrap_or(m.id),
                capabilities: ModelCapabilities::default(),
                billing: ModelBilling::default(),
            })
            .collect();

        Some(models)
    }

    pub async fn check_cluster_models(
        &mut self,
        endpoint: &str,
    ) -> Result<Vec<CopilotModel>, Box<dyn std::error::Error>> {
        let cache_key = format!("cluster_{}", endpoint);

        if let Some(models) = self.cached_models.get(&cache_key) {
            return Ok(models.clone());
        }

        info!("Discovering private cluster models at {}", endpoint);

        // Query OpenAI-compatible /v1/models endpoint
        let response = self
            .client
            .get(format!("{}/models", endpoint))
            .send()
            .await?;

        if !response.status().is_success() {
            warn!("Failed to query cluster models: {}", response.status());
            return Ok(Self::get_default_cluster_models());
        }

        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            id: String,
        }

        let models_response: ModelsResponse = response.json().await?;

        let models: Vec<CopilotModel> = models_response
            .data
            .into_iter()
            .map(|m| CopilotModel {
                id: m.id.clone(),
                name: m.id,
                capabilities: ModelCapabilities::default(),
                billing: ModelBilling::default(),
            })
            .collect();

        info!("Discovered {} cluster models", models.len());
        self.cached_models.insert(cache_key, models.clone());
        Ok(models)
    }

    fn get_default_copilot_models() -> Vec<CopilotModel> {
        vec![
            CopilotModel {
                id: "gpt-4.1".to_string(),
                name: "GPT-4.1".to_string(),
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    max_tokens: Some(128000),
                },
                billing: ModelBilling {
                    tier: "business".to_string(),
                },
            },
            CopilotModel {
                id: "gpt-4.1-mini".to_string(),
                name: "GPT-4.1 Mini".to_string(),
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    max_tokens: Some(128000),
                },
                billing: ModelBilling {
                    tier: "individual".to_string(),
                },
            },
        ]
    }

    fn get_default_claude_models() -> Vec<CopilotModel> {
        vec![
            CopilotModel {
                id: "claude-sonnet-4-7".to_string(),
                name: "Claude Sonnet 4.7".to_string(),
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    max_tokens: Some(200000),
                },
                billing: ModelBilling {
                    tier: "subscription".to_string(),
                },
            },
            CopilotModel {
                id: "claude-opus-4-8".to_string(),
                name: "Claude Opus 4.8".to_string(),
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    max_tokens: Some(200000),
                },
                billing: ModelBilling {
                    tier: "subscription".to_string(),
                },
            },
            CopilotModel {
                id: "claude-haiku-4-5".to_string(),
                name: "Claude Haiku 4.5".to_string(),
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: true,
                    supports_vision: true,
                    max_tokens: Some(200000),
                },
                billing: ModelBilling {
                    tier: "subscription".to_string(),
                },
            },
        ]
    }

    fn get_default_cluster_models() -> Vec<CopilotModel> {
        vec![
            CopilotModel {
                id: "llama-3.1-70b".to_string(),
                name: "Llama 3.1 70B".to_string(),
                capabilities: ModelCapabilities::default(),
                billing: ModelBilling::default(),
            },
            CopilotModel {
                id: "deepseek-coder-33b".to_string(),
                name: "DeepSeek Coder 33B".to_string(),
                capabilities: ModelCapabilities::default(),
                billing: ModelBilling::default(),
            },
        ]
    }
}

impl Default for ModelDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_copilot_models() {
        let models = ModelDiscovery::get_default_copilot_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "gpt-4.1"));
    }

    #[test]
    fn test_default_claude_models() {
        let models = ModelDiscovery::get_default_claude_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id.contains("claude")));
        assert!(models.iter().any(|m| m.id.contains("sonnet")));
    }

    #[test]
    fn test_default_cluster_models() {
        let models = ModelDiscovery::get_default_cluster_models();
        assert!(!models.is_empty());
    }
}
