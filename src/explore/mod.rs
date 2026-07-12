//! Upstream repository-exploration stage.
//!
//! Before the router classifies a request, an OPTIONAL FastContext explorer runs
//! an agentic READ/GLOB/GREP loop against the repo on disk and returns a small,
//! capped bundle of evidence (file paths + line ranges expanded to real code).
//! That clean context is injected into the request so the existing classifier
//! sees a short, targeted prompt instead of a sprawling one.
//!
//! Invariants:
//! - FastContext is NEVER a routing destination; it runs strictly upstream.
//! - Inference is remote and backend-agnostic (any OpenAI-compatible endpoint).
//! - The three tools execute LOCALLY, sandboxed to `repo_root`.
//! - The value is returning LITTLE, targeted context — expansion is hard-capped.

mod client;
mod expand;
mod tools;

use crate::config::{ExploreConfig, FastContextConfig};
use client::{FastContextClient, run_loop};
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use tools::Sandbox;

/// A single (path, line-range) reference the explorer cited in its final answer.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Citation {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// A citation expanded into real, line-numbered source.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Snippet {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Line-numbered code text for the cited range, after the global caps.
    pub code: String,
}

/// Loop and output measurements, surfaced to telemetry.
#[derive(Debug, Clone, Serialize, Default, PartialEq, Eq)]
pub struct ExploreStats {
    pub turns: usize,
    pub tool_calls: usize,
    /// The loop stopped because it hit `max_turns` (evidence is best-effort).
    pub hit_turn_cap: bool,
    pub expanded_lines: usize,
    pub expanded_tokens: usize,
}

/// The cleaned, capped evidence returned upstream of the router.
#[derive(Debug, Clone, Serialize, Default)]
pub struct Evidence {
    pub citations: Vec<Citation>,
    pub expanded_snippets: Vec<Snippet>,
    pub stats: ExploreStats,
}

/// Typed failures from the exploration stage. The caller treats any error as
/// "no evidence" and proceeds with the original request unchanged.
#[derive(Debug, thiserror::Error)]
pub enum ExploreError {
    #[error("fastcontext not configured: {0}")]
    NotConfigured(String),
    #[error("sandbox violation: {0}")]
    Sandbox(String),
    #[error("explorer request failed: {0}")]
    Request(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl Evidence {
    /// Renders the evidence as a single clean context block to inject upstream of
    /// the router, or `None` when nothing was found (so the request is untouched).
    pub fn to_context_message(&self) -> Option<String> {
        if self.expanded_snippets.is_empty() {
            return None;
        }
        let mut out = String::from(
            "## Repository context (auto-gathered)\n\n\
             The following code was located as relevant to the task:\n",
        );
        for s in &self.expanded_snippets {
            out.push_str(&format!(
                "\n### {}:{}-{}\n```\n{}\n```\n",
                s.path, s.start_line, s.end_line, s.code
            ));
        }
        Some(out)
    }
}

/// Runs the FastContext explorer against `repo_root` for `query` and returns the
/// cleaned, capped evidence. `repo_root` is supplied by the caller (resolved from
/// `explore.repo_root` or the CLI); `explore` provides the loop/expansion caps and
/// `fastcontext` the remote endpoint. Any failure is returned as a typed error —
/// the caller treats that as "no evidence" and proceeds with the original request.
pub async fn explore_repo(
    query: &str,
    repo_root: &Path,
    explore: &ExploreConfig,
    fastcontext: &FastContextConfig,
) -> Result<Evidence, ExploreError> {
    if fastcontext.endpoint_url.trim().is_empty() {
        return Err(ExploreError::NotConfigured(
            "fastcontext.endpoint_url is empty".to_string(),
        ));
    }

    let sandbox = Arc::new(Sandbox::new(repo_root)?);
    let model = FastContextClient::new(
        fastcontext.endpoint_url.clone(),
        fastcontext.model.clone(),
        fastcontext.api_key.clone(),
    );

    let outcome = run_loop(&model, sandbox.clone(), query, explore.max_turns).await?;
    tracing::debug!(final_text = %outcome.final_text, "explorer final answer");

    let mut evidence = expand::build_evidence(
        &sandbox,
        &outcome.final_text,
        explore.max_expanded_lines,
        explore.max_expanded_tokens,
    );
    evidence.stats.turns = outcome.turns;
    evidence.stats.tool_calls = outcome.tool_calls;
    evidence.stats.hit_turn_cap = outcome.hit_turn_cap;
    Ok(evidence)
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use axum::{Json, Router, extract::State, routing::post};
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    /// Mock FastContext endpoint: first call asks to READ a file, second call
    /// returns a `<final_answer>` citing it. Exercises the real HTTP client,
    /// real local tool execution, citation parsing, and capped expansion.
    async fn mock_endpoint(
        State(count): State<Arc<AtomicUsize>>,
        Json(_body): Json<Value>,
    ) -> Json<Value> {
        let n = count.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{\"path\":\"src/main.rs\"}"}
                        }]
                    }
                }]
            }))
        } else {
            Json(json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<final_answer>\nsrc/main.rs:1-3\n</final_answer>"
                    }
                }]
            }))
        }
    }

    #[tokio::test]
    async fn explore_repo_drives_loop_against_mocked_endpoint() {
        // Fixture repo on disk.
        let root = std::env::temp_dir().join(format!("tm-int-{}", std::process::id()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();

        // Mocked model endpoint on an ephemeral loopback port.
        let count = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_endpoint))
            .with_state(count);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let fastcontext = FastContextConfig {
            enabled: true,
            endpoint_url: format!("http://{addr}/v1"),
            model: "mock".to_string(),
            api_key: None,
        };
        let explore = ExploreConfig::default();

        let evidence = explore_repo("where is main", &root, &explore, &fastcontext)
            .await
            .expect("exploration succeeds against mock");

        assert_eq!(
            evidence.citations,
            vec![Citation {
                path: "src/main.rs".to_string(),
                start_line: 1,
                end_line: 3,
            }]
        );
        assert_eq!(evidence.expanded_snippets.len(), 1);
        assert!(evidence.expanded_snippets[0].code.contains("fn main()"));
        assert_eq!(evidence.stats.tool_calls, 1);
        assert!(!evidence.stats.hit_turn_cap);
        assert!(evidence.to_context_message().is_some());

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn explore_repo_errors_when_endpoint_unset() {
        let root = std::env::temp_dir();
        let err = explore_repo(
            "q",
            &root,
            &ExploreConfig::default(),
            &FastContextConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ExploreError::NotConfigured(_)));
    }
}
