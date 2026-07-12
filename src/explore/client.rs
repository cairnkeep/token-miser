//! The FastContext explorer: an OpenAI-compatible chat client and the agentic
//! tool-calling loop that drives it.
//!
//! The model is asked to locate the code relevant to a query. Each turn it may
//! emit `tool_calls` (READ/GLOB/GREP); THIS process executes them locally against
//! the sandbox and feeds the observations back as `role:"tool"` messages. The
//! loop ends when the model emits a `<final_answer>` block (or stops calling
//! tools), and is hard-capped at `max_turns` — on the cap it returns the best
//! evidence gathered so far. Inference is backend-agnostic: any endpoint that
//! speaks `chat/completions` with `tools` works.

use super::ExploreError;
use super::tools::Sandbox;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// One tool call the model requested (OpenAI `message.tool_calls` entry).
#[derive(Clone, Debug)]
pub struct ToolCallReq {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments string, as the OpenAI schema delivers them.
    pub arguments: String,
}

/// A parsed assistant turn: free-text content and/or a batch of tool calls.
#[derive(Clone, Debug, Default)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallReq>,
}

/// The result of running the exploration loop.
#[derive(Clone, Debug)]
pub struct LoopOutcome {
    /// The model's last assistant text (contains the `<final_answer>` block on a
    /// clean finish; best-effort last text on a turn-cap finish).
    pub final_text: String,
    pub turns: usize,
    pub tool_calls: usize,
    pub hit_turn_cap: bool,
}

/// A remote chat model that returns one assistant turn for a message history.
/// Abstracted so the loop can be unit-tested with zero network.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// One turn with the exploration tools available.
    async fn chat(&self, messages: &[Value]) -> Result<AssistantTurn, ExploreError>;
    /// Final-answer flush: one turn with tools DISABLED, used when the loop hits
    /// the turn cap so the model is forced to emit its `<final_answer>` from what
    /// it has already gathered. Defaults to `chat` (sufficient for test mocks).
    async fn chat_no_tools(&self, messages: &[Value]) -> Result<AssistantTurn, ExploreError> {
        self.chat(messages).await
    }
}

/// System prompt: keep the explorer focused on returning a SMALL set of precise
/// citations. The `<final_answer>` citation format is the contract `expand.rs`
/// parses, so it is specified exactly here.
const SYSTEM_PROMPT: &str = "\
You are a repository exploration agent. Your job is to locate the few code \
locations most relevant to the user's task, using the READ, GLOB, and GREP \
tools. Do NOT attempt to solve the task or write code.\n\n\
Be decisive and efficient. Use broad GREP/GLOB queries to find candidates, batch \
parallel tool calls in a single turn, and READ only the specific spans you need \
to confirm relevance. Do not over-explore: a few well-chosen searches and reads \
are enough. As soon as you have located the relevant code, STOP calling tools and \
emit your final answer.\n\n\
Reply with a single <final_answer> block and nothing else. Inside it, list ONLY \
the citations that matter, one per line, each as `relative/path:START-END` where \
START-END are the 1-based line numbers you inspected, e.g.:\n\n\
<final_answer>\n\
src/router.rs:42-88\n\
src/config.rs:120-145\n\
</final_answer>\n\n\
Keep the list minimal — a handful of tight ranges, not whole files.";

/// The three tool schemas advertised to the model on every turn.
fn tool_schemas() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file's contents with line numbers. Use offset/limit to read a span.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the repo root."},
                        "offset": {"type": "integer", "description": "1-based first line to read."},
                        "limit": {"type": "integer", "description": "Number of lines to read."}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "List files matching a glob pattern (gitignore-aware). Pattern matches the path relative to the repo root, e.g. src/**/*.rs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "base": {"type": "string", "description": "Optional subdirectory to search under."}
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Regex search across the repo (gitignore-aware). Returns path:line:content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "regex": {"type": "string"},
                        "path": {"type": "string", "description": "Optional single file to search."},
                        "glob": {"type": "string", "description": "Optional glob to limit which files are searched."}
                    },
                    "required": ["regex"]
                }
            }
        }
    ])
}

/// OpenAI-compatible client to the FastContext endpoint. Mirrors the cluster
/// classifier's client conventions (shared reqwest client, bounded timeouts).
pub struct FastContextClient {
    client: Client,
    endpoint_url: String,
    model: String,
    api_key: Option<String>,
}

impl FastContextClient {
    pub fn new(endpoint_url: String, model: String, api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to create HTTP client for FastContext");
        Self {
            client,
            endpoint_url,
            model,
            api_key,
        }
    }

    /// One `chat/completions` round. With `with_tools`, the three exploration
    /// tools are offered (`tool_choice:auto`); without, tools are suppressed
    /// (`tool_choice:none`) to force a plain final answer.
    async fn complete(
        &self,
        messages: &[Value],
        with_tools: bool,
    ) -> Result<AssistantTurn, ExploreError> {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": false,
        });
        if with_tools {
            body["tools"] = tool_schemas();
            body["tool_choice"] = json!("auto");
        } else {
            body["tool_choice"] = json!("none");
        }

        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.endpoint_url))
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .map_err(|e| ExploreError::Request(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ExploreError::Request(format!(
                "fastcontext returned {status}: {text}"
            )));
        }

        let v: Value = response
            .json()
            .await
            .map_err(|e| ExploreError::Request(format!("parse fastcontext response: {e}")))?;
        Ok(parse_assistant_turn(&v))
    }
}

#[async_trait]
impl ChatModel for FastContextClient {
    async fn chat(&self, messages: &[Value]) -> Result<AssistantTurn, ExploreError> {
        self.complete(messages, true).await
    }

    async fn chat_no_tools(&self, messages: &[Value]) -> Result<AssistantTurn, ExploreError> {
        self.complete(messages, false).await
    }
}

/// Parses the assistant message out of an OpenAI `chat/completions` response.
fn parse_assistant_turn(v: &Value) -> AssistantTurn {
    let msg = &v["choices"][0]["message"];
    let content = msg["content"].as_str().map(|s| s.to_string());

    let mut tool_calls = Vec::new();
    if let Some(arr) = msg["tool_calls"].as_array() {
        for tc in arr {
            let func = &tc["function"];
            // `arguments` is a JSON string per spec, but some servers emit an
            // object — accept either by serializing the non-string form.
            let arguments = func["arguments"]
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| func["arguments"].to_string());
            tool_calls.push(ToolCallReq {
                id: tc["id"].as_str().unwrap_or_default().to_string(),
                name: func["name"].as_str().unwrap_or_default().to_string(),
                arguments,
            });
        }
    }

    AssistantTurn {
        content,
        tool_calls,
    }
}

/// Rebuilds the assistant message (with its tool calls) for the running history,
/// so the model sees its own prior calls on the next turn.
fn assistant_message(turn: &AssistantTurn) -> Value {
    let mut m = json!({"role": "assistant"});
    m["content"] = turn
        .content
        .clone()
        .map(Value::String)
        .unwrap_or(Value::Null);
    if !turn.tool_calls.is_empty() {
        m["tool_calls"] = Value::Array(
            turn.tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect(),
        );
    }
    m
}

fn parse_args(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| json!({}))
}

/// Drives the agentic exploration loop to a `<final_answer>` (or the turn cap).
///
/// Tool calls the model batches in one turn are executed concurrently on blocking
/// threads (the FS work is kept off the async reactor), then their observations
/// are appended in order as `role:"tool"` messages keyed by `tool_call_id`.
pub async fn run_loop(
    model: &dyn ChatModel,
    sandbox: Arc<Sandbox>,
    query: &str,
    max_turns: usize,
) -> Result<LoopOutcome, ExploreError> {
    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": SYSTEM_PROMPT}),
        json!({"role": "user", "content": query}),
    ];
    let mut total_tool_calls = 0usize;
    let mut last_text = String::new();
    let max = max_turns.max(1);

    for turn in 1..=max {
        // A mid-loop failure (transient error, or the wandering explorer
        // overflowing the context window) is non-fatal: stop and return the best
        // evidence gathered so far rather than failing the whole exploration.
        let assistant = match model.chat(&messages).await {
            Ok(a) => a,
            Err(e) => {
                warn!(turn, error = %e, "explorer turn failed; returning best-effort evidence");
                return Ok(LoopOutcome {
                    final_text: last_text,
                    turns: turn,
                    tool_calls: total_tool_calls,
                    hit_turn_cap: false,
                });
            }
        };
        if let Some(c) = &assistant.content {
            last_text = c.clone();
        }
        messages.push(assistant_message(&assistant));

        let has_final = assistant
            .content
            .as_deref()
            .is_some_and(|c| c.contains("<final_answer>"));
        if has_final {
            debug!(turn, total_tool_calls, "explorer emitted final answer");
            return Ok(LoopOutcome {
                final_text: last_text,
                turns: turn,
                tool_calls: total_tool_calls,
                hit_turn_cap: false,
            });
        }
        if assistant.tool_calls.is_empty() {
            // The model stopped calling tools but produced no <final_answer> —
            // typically prose, or (for a small model) a tool call mistakenly
            // emitted as plain text in `content`. Force a final-answer flush so we
            // still recover the citations it found instead of returning nothing.
            debug!(turn, "explorer stopped without final answer; forcing flush");
            last_text = flush_final(model, &mut messages, last_text).await;
            return Ok(LoopOutcome {
                final_text: last_text,
                turns: turn,
                tool_calls: total_tool_calls,
                hit_turn_cap: false,
            });
        }

        total_tool_calls += assistant.tool_calls.len();
        let futs = assistant.tool_calls.iter().map(|c| {
            let sb = sandbox.clone();
            let name = c.name.clone();
            let args = parse_args(&c.arguments);
            let id = c.id.clone();
            async move {
                let obs = tokio::task::spawn_blocking(move || sb.run_tool(&name, &args))
                    .await
                    .unwrap_or_else(|e| format!("ERROR: tool task failed: {e}"));
                (id, obs)
            }
        });
        let results = futures::future::join_all(futs).await;
        for (id, obs) in results {
            messages.push(json!({"role": "tool", "tool_call_id": id, "content": obs}));
        }
    }

    // Turn cap reached without convergence (a small explorer can wander). Force a
    // final-answer flush so we still extract the evidence it gathered.
    warn!(max, "explorer hit turn cap; forcing final-answer flush");
    last_text = flush_final(model, &mut messages, last_text).await;
    Ok(LoopOutcome {
        final_text: last_text,
        turns: max,
        tool_calls: total_tool_calls,
        hit_turn_cap: true,
    })
}

/// Forces a final answer with tools disabled and returns its text, falling back
/// to `last_text` if the flush yields nothing. Used both when the model stops
/// without a `<final_answer>` and when the turn cap is reached.
async fn flush_final(
    model: &dyn ChatModel,
    messages: &mut Vec<Value>,
    last_text: String,
) -> String {
    messages.push(json!({"role": "user", "content": FLUSH_PROMPT}));
    if let Ok(flush) = model.chat_no_tools(messages).await
        && let Some(c) = flush.content
        && !c.trim().is_empty()
    {
        return c;
    }
    last_text
}

/// Directive for the forced final-answer flush.
const FLUSH_PROMPT: &str = "\
Stop exploring and do not call any more tools. Based only on the files you have \
already examined, output your <final_answer> block now, listing the most relevant \
file paths and line ranges, one per line as `path:START-END`.";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scripted ChatModel: returns successive turns; when exhausted it either
    /// repeats the last turn (to force the cap) or yields an empty turn (ends).
    /// `flush` (if set) is what `chat_no_tools` returns, modeling the cap flush.
    struct MockModel {
        turns: Vec<AssistantTurn>,
        idx: AtomicUsize,
        repeat_last: bool,
        flush: Option<AssistantTurn>,
    }

    impl MockModel {
        fn new(turns: Vec<AssistantTurn>, repeat_last: bool) -> Self {
            Self {
                turns,
                idx: AtomicUsize::new(0),
                repeat_last,
                flush: None,
            }
        }
    }

    #[async_trait]
    impl ChatModel for MockModel {
        async fn chat(&self, _messages: &[Value]) -> Result<AssistantTurn, ExploreError> {
            let i = self.idx.fetch_add(1, Ordering::SeqCst);
            let turn = match self.turns.get(i) {
                Some(t) => t.clone(),
                None if self.repeat_last => self.turns.last().cloned().unwrap_or_default(),
                None => AssistantTurn::default(),
            };
            Ok(turn)
        }

        async fn chat_no_tools(&self, messages: &[Value]) -> Result<AssistantTurn, ExploreError> {
            match &self.flush {
                Some(t) => Ok(t.clone()),
                None => self.chat(messages).await,
            }
        }
    }

    fn tool_turn(name: &str, args: &str) -> AssistantTurn {
        AssistantTurn {
            content: None,
            tool_calls: vec![ToolCallReq {
                id: "call_1".to_string(),
                name: name.to_string(),
                arguments: args.to_string(),
            }],
        }
    }

    fn text_turn(text: &str) -> AssistantTurn {
        AssistantTurn {
            content: Some(text.to_string()),
            tool_calls: vec![],
        }
    }

    fn fixture_sandbox() -> (std::path::PathBuf, Arc<Sandbox>) {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("tm-loop-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
        let sb = Arc::new(Sandbox::new(&root).unwrap());
        (root, sb)
    }

    #[tokio::test]
    async fn loop_executes_tool_then_terminates_on_final_answer() {
        let (root, sb) = fixture_sandbox();
        let model = MockModel::new(
            vec![
                tool_turn("read", r#"{"path":"main.rs"}"#),
                text_turn("<final_answer>\nmain.rs:1-1\n</final_answer>"),
            ],
            false,
        );

        let outcome = run_loop(&model, sb, "where is main", 16).await.unwrap();

        assert_eq!(outcome.turns, 2);
        assert_eq!(outcome.tool_calls, 1);
        assert!(!outcome.hit_turn_cap);
        assert!(outcome.final_text.contains("<final_answer>"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn loop_terminates_when_no_tool_calls() {
        let (root, sb) = fixture_sandbox();
        let model = MockModel::new(vec![text_turn("just an answer, no tools")], false);

        let outcome = run_loop(&model, sb, "q", 16).await.unwrap();

        assert_eq!(outcome.turns, 1);
        assert_eq!(outcome.tool_calls, 0);
        assert!(!outcome.hit_turn_cap);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn loop_stops_at_turn_cap_best_effort() {
        let (root, sb) = fixture_sandbox();
        // Always asks for another tool call -> never emits <final_answer>.
        let model = MockModel::new(vec![tool_turn("read", r#"{"path":"main.rs"}"#)], true);

        let outcome = run_loop(&model, sb, "q", 3).await.unwrap();

        assert!(outcome.hit_turn_cap);
        assert_eq!(outcome.turns, 3);
        assert_eq!(outcome.tool_calls, 3);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn loop_flushes_final_answer_at_turn_cap() {
        let (root, sb) = fixture_sandbox();
        // Wanders (always tool calls), but the cap flush yields a final answer.
        let mut model = MockModel::new(vec![tool_turn("read", r#"{"path":"main.rs"}"#)], true);
        model.flush = Some(text_turn("<final_answer>\nmain.rs:1-1\n</final_answer>"));

        let outcome = run_loop(&model, sb, "q", 3).await.unwrap();

        assert!(outcome.hit_turn_cap);
        assert_eq!(outcome.turns, 3);
        // The forced flush recovered the evidence instead of returning nothing.
        assert!(outcome.final_text.contains("<final_answer>"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn loop_flushes_when_model_stops_without_final_answer() {
        let (root, sb) = fixture_sandbox();
        // Turn 1 reads a file; turn 2 emits a tool call as PLAIN TEXT in content
        // (no structured tool_calls, no <final_answer>) — the real 4B failure mode.
        // The flush must recover a final answer instead of returning nothing.
        let mut model = MockModel::new(
            vec![
                tool_turn("read", r#"{"path":"main.rs"}"#),
                text_turn(r#"{"name":"read","arguments":{"path":"nope.rs"}}"#),
            ],
            false,
        );
        model.flush = Some(text_turn("<final_answer>\nmain.rs:1-1\n</final_answer>"));

        let outcome = run_loop(&model, sb, "q", 16).await.unwrap();

        assert!(!outcome.hit_turn_cap);
        assert_eq!(outcome.turns, 2);
        assert!(outcome.final_text.contains("<final_answer>"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn parse_assistant_turn_reads_tool_calls() {
        let v = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "grep", "arguments": "{\"regex\":\"fn main\"}"}
                    }]
                }
            }]
        });
        let turn = parse_assistant_turn(&v);
        assert!(turn.content.is_none());
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "grep");
        assert_eq!(turn.tool_calls[0].arguments, "{\"regex\":\"fn main\"}");
    }

    #[test]
    fn parse_assistant_turn_accepts_object_arguments() {
        // Some servers emit `arguments` as an object instead of a JSON string.
        let v = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "read", "arguments": {"path": "main.rs"}}
                    }]
                }
            }]
        });
        let turn = parse_assistant_turn(&v);
        let args = parse_args(&turn.tool_calls[0].arguments);
        assert_eq!(args["path"], "main.rs");
    }
}
