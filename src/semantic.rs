use crate::config::SemanticRouterConfig;
use crate::models::ChatCompletionRequest;
use crate::router::Tier;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::sync::OnceCell;
use tracing::{debug, warn};

// Representative prompts per tier. Their mean embedding forms each tier's
// centroid; a request routes to the nearest centroid by cosine similarity.
const FREE_EXEMPLARS: &[&str] = &[
    "fix this typo",
    "what's the syntax for a for loop",
    "format this json snippet",
    "autocomplete this line of code",
    "how do I print to stdout in python",
    "what does this operator do",
];
const STANDARD_EXEMPLARS: &[&str] = &[
    "implement a function to reverse a linked list",
    "debug why this function returns none",
    "write a unit test for this function",
    "explain what this block of code does",
    "optimize this sorting routine",
    "add input validation to this function",
];
const COMPLEX_EXEMPLARS: &[&str] = &[
    "architect a multi-service event-driven system",
    "migrate the codebase to a new framework",
    "redesign the authentication and authorization flow",
    "design a horizontally scalable rate limiter",
    "decouple persistence from domain logic across the module",
];

/// Per-tier centroid embeddings, or `None` if they could not be computed.
type Centroids = Option<Vec<(Tier, Vec<f32>)>>;

/// Routes requests by embedding similarity to per-tier exemplar centroids.
pub struct SemanticRouter {
    client: Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    centroids: OnceCell<Centroids>,
}

impl SemanticRouter {
    pub fn new(config: SemanticRouterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create embeddings HTTP client");
        Self {
            client,
            endpoint: config.endpoint,
            model: config.model,
            api_key: config.api_key,
            centroids: OnceCell::new(),
        }
    }

    /// Classifies a request to a tier, or `None` if embedding is unavailable
    /// (caller should fall back to the heuristic classifier).
    pub async fn classify(&self, request: &ChatCompletionRequest) -> Option<Tier> {
        let centroids = self.centroids().await.as_ref()?;
        let query = self.embed(&request_text(request)).await?;
        let tier = nearest_tier(&query, centroids);
        debug!(tier = ?tier, "semantic route");
        Some(tier)
    }

    /// Lazily computes and caches the per-tier centroids on first use.
    async fn centroids(&self) -> &Centroids {
        self.centroids
            .get_or_init(|| self.compute_centroids())
            .await
    }

    async fn compute_centroids(&self) -> Centroids {
        let groups = [
            (Tier::Free, FREE_EXEMPLARS),
            (Tier::Standard, STANDARD_EXEMPLARS),
            (Tier::Complex, COMPLEX_EXEMPLARS),
        ];
        let mut centroids = Vec::with_capacity(groups.len());
        for (tier, exemplars) in groups {
            let mut vectors = Vec::with_capacity(exemplars.len());
            for exemplar in exemplars {
                vectors.push(self.embed(exemplar).await?);
            }
            centroids.push((tier, centroid(&vectors)?));
        }
        debug!("semantic router centroids ready");
        Some(centroids)
    }

    async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut builder = self
            .client
            .post(format!("{}/embeddings", self.endpoint))
            .json(&json!({ "model": self.model, "input": text }));
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {key}"));
        }

        let response = builder.send().await.ok()?;
        if !response.status().is_success() {
            warn!(status = %response.status(), "embedding request failed");
            return None;
        }

        let body: serde_json::Value = response.json().await.ok()?;
        let array = body["data"][0]["embedding"].as_array()?;
        let vector: Vec<f32> = array
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        (!vector.is_empty()).then_some(vector)
    }
}

fn request_text(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_text())
        .unwrap_or_default()
}

fn centroid(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = vectors.first()?.len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (i, x) in v.iter().enumerate() {
            mean[i] += x;
        }
    }
    let n = vectors.len() as f32;
    for x in &mut mean {
        *x /= n;
    }
    Some(mean)
}

fn nearest_tier(query: &[f32], centroids: &[(Tier, Vec<f32>)]) -> Tier {
    centroids
        .iter()
        .map(|(tier, c)| (tier.clone(), cosine(query, c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(tier, _)| tier)
        .unwrap_or(Tier::Standard)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        -1.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[2.0, 2.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_is_mean() {
        let c = centroid(&[vec![0.0, 2.0], vec![2.0, 0.0]]).unwrap();
        assert_eq!(c, vec![1.0, 1.0]);
        assert!(centroid(&[vec![1.0], vec![1.0, 2.0]]).is_none());
    }

    #[test]
    fn test_nearest_tier_picks_highest_similarity() {
        let centroids = vec![
            (Tier::Free, vec![1.0, 0.0, 0.0]),
            (Tier::Standard, vec![0.0, 1.0, 0.0]),
            (Tier::Complex, vec![0.0, 0.0, 1.0]),
        ];
        assert_eq!(nearest_tier(&[0.1, 0.9, 0.2], &centroids), Tier::Standard);
        assert_eq!(nearest_tier(&[0.9, 0.1, 0.0], &centroids), Tier::Free);
        assert_eq!(nearest_tier(&[0.0, 0.2, 0.8], &centroids), Tier::Complex);
    }
}
