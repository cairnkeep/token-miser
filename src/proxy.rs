use crate::config::{AuthType, Config, EscalationConfig, JudgeConfig, ProviderConfig};
use crate::models::{ChatCompletionRequest, ChatCompletionResponse, Message, MessageContent};
use crate::router::Tier;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// A boxed upstream byte stream (OpenAI SSE frames) as produced by reqwest.
type ByteStream = Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>;

pub enum ProxyResponse {
    Complete(ChatCompletionResponse),
    Stream(ByteStream),
}

/// The outcome of a (possibly escalated) forward: the tier that actually served
/// the response and how many times routing escalated to reach it.
pub struct Forwarded {
    pub served_tier: Tier,
    pub escalations: u8,
    pub response: ProxyResponse,
}

/// Whether an error from a tier warrants retrying on a higher tier. Transient or
/// overload conditions escalate; deterministic config/auth/client errors do not.
fn should_escalate_error(err: &ProxyError) -> bool {
    match err {
        ProxyError::RequestFailed(_) | ProxyError::ParseError(_) => true,
        ProxyError::ProviderError { status, .. } => *status >= 500 || *status == 429,
        ProxyError::ProviderNotFound(_) | ProxyError::AuthenticationFailed(_) => false,
    }
}

/// Whether a completed response looks like the cheaper tier failed to do the job,
/// per the configured heuristics. `content_filter` always counts.
fn response_is_degenerate(response: &ChatCompletionResponse, cfg: &EscalationConfig) -> bool {
    let Some(choice) = response.choices.first() else {
        return true;
    };
    if choice.finish_reason == "content_filter" {
        return true;
    }
    if cfg.on_truncation && choice.finish_reason == "length" {
        return true;
    }
    // A tool-call turn legitimately has empty text content — never "empty".
    let has_tool_call = choice
        .message
        .tool_calls
        .as_ref()
        .is_some_and(|c| !c.is_empty())
        || choice.finish_reason == "tool_calls";
    let text = choice.message.content.as_text();
    let trimmed = text.trim();
    if cfg.on_empty_response && trimmed.is_empty() && !has_tool_call {
        return true;
    }
    if cfg.on_refusal && looks_like_refusal(trimmed) {
        return true;
    }
    false
}

/// Signals parsed from a buffered stream prefix for the early degeneracy check.
struct StreamPrefix {
    /// Raw upstream frames, replayed verbatim downstream if the prefix is adequate.
    frames: Vec<Bytes>,
    /// Accumulated assistant `content` text seen so far.
    text: String,
    /// `finish_reason` if one arrived within the look-ahead window.
    finish: Option<String>,
    /// Any `tool_calls` delta seen — a legitimate empty-content (tool) turn.
    saw_tool_call: bool,
    /// Upstream hit `[DONE]`/EOF within the window.
    ended: bool,
}

/// Pulls upstream SSE frames until `max_chars` of assistant text accumulate, a
/// `finish_reason` arrives, or the stream ends — whichever comes first (also hard-
/// capped at a frame count so a pathological tiny-delta stream can't buffer
/// forever). Returns the buffered prefix (for verbatim replay) plus the parsed
/// signals, and hands back the un-consumed remainder of the stream.
async fn lookahead(mut stream: ByteStream, max_chars: usize) -> (StreamPrefix, ByteStream) {
    const FRAME_CAP: usize = 256;

    let mut frames: Vec<Bytes> = Vec::new();
    let mut parse_buf: Vec<u8> = Vec::new();
    let mut text = String::new();
    let mut finish: Option<String> = None;
    let mut saw_tool_call = false;
    let mut ended = false;

    while text.chars().count() < max_chars && finish.is_none() && frames.len() < FRAME_CAP {
        match stream.next().await {
            Some(Ok(bytes)) => {
                parse_buf.extend_from_slice(&bytes);
                frames.push(bytes);
                // Drain complete SSE lines; partial trailing lines stay buffered.
                while let Some(pos) = parse_buf.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = parse_buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line);
                    let line = line.trim_end_matches(['\r', '\n']);
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        ended = true;
                    } else if !payload.is_empty()
                        && let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload)
                    {
                        let delta = &chunk["choices"][0]["delta"];
                        if let Some(c) = delta["content"].as_str() {
                            text.push_str(c);
                        }
                        if delta["tool_calls"].is_array() {
                            saw_tool_call = true;
                        }
                        if let Some(r) = chunk["choices"][0]["finish_reason"].as_str() {
                            finish = Some(r.to_string());
                        }
                    }
                }
                if ended {
                    break;
                }
            }
            // Upstream error or EOF mid-window: treat as ended, replay what we have.
            Some(Err(_)) | None => {
                ended = true;
                break;
            }
        }
    }

    (
        StreamPrefix {
            frames,
            text,
            finish,
            saw_tool_call,
            ended,
        },
        stream,
    )
}

/// Early degeneracy check on a buffered stream prefix. Mirrors
/// `response_is_degenerate` but adapted to partial data: `content_filter` and
/// refusal fire on the prefix; empty/truncation only once the stream has ended
/// within the window. A turn carrying tool calls is never "empty".
fn prefix_is_degenerate(p: &StreamPrefix, cfg: &EscalationConfig) -> bool {
    if p.finish.as_deref() == Some("content_filter") {
        return true;
    }
    if cfg.on_truncation && p.ended && p.finish.as_deref() == Some("length") {
        return true;
    }
    let has_tool_call = p.saw_tool_call || p.finish.as_deref() == Some("tool_calls");
    let trimmed = p.text.trim();
    if cfg.on_empty_response && p.ended && trimmed.is_empty() && !has_tool_call {
        return true;
    }
    if cfg.on_refusal && looks_like_refusal(trimmed) {
        return true;
    }
    false
}

/// Conservative refusal detection: only flags short responses dominated by a
/// known refusal phrase, to avoid escalating long answers that merely mention one.
fn looks_like_refusal(text: &str) -> bool {
    if text.chars().count() > 400 {
        return false;
    }
    let lower = text.to_lowercase();
    const PATTERNS: &[&str] = &[
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i'm not able to",
        "i am not able to",
        "i'm unable to",
        "i am unable to",
        "i can't provide",
        "i cannot provide",
        "as an ai language model",
        "i don't have the ability",
        "i do not have the ability",
        "sorry, i can't",
        "sorry, i cannot",
    ];
    PATTERNS.iter().any(|p| lower.contains(p))
}

/// Truncates to at most `max` characters on a char boundary.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max).collect()
    }
}

pub struct Proxy {
    config: Arc<Config>,
    client: Client,
    token_cache: Arc<RwLock<HashMap<String, String>>>,
}

impl Proxy {
    pub fn new(config: Arc<Config>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            client,
            token_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Forwards a request to the routed tier, escalating to the next tier up on
    /// a transient error or degenerate response when escalation is enabled.
    pub async fn forward(
        &self,
        request: ChatCompletionRequest,
        tier: Tier,
    ) -> Result<Forwarded, ProxyError> {
        let esc = &self.config.escalation;
        let max = if esc.enabled { esc.max_escalations } else { 0 };
        let mut current = tier;
        let mut escalations = 0u8;

        loop {
            let result = self.forward_once(request.clone(), current.clone()).await;

            // The next tier up, only if escalation budget remains and one exists.
            let next = if escalations < max {
                current.escalate()
            } else {
                None
            };

            if let Some(next_tier) = next {
                match result {
                    Err(e) => {
                        if should_escalate_error(&e) {
                            warn!(from = ?current, to = ?next_tier, error = %e, "escalating after transient error");
                            current = next_tier;
                            escalations += 1;
                            continue;
                        }
                        return Err(e);
                    }
                    Ok(ProxyResponse::Complete(resp)) => {
                        if self.should_escalate_complete(&request, &resp, esc).await {
                            warn!(from = ?current, to = ?next_tier, "escalating after degenerate response");
                            current = next_tier;
                            escalations += 1;
                            continue;
                        }
                        return Ok(Forwarded {
                            served_tier: current,
                            escalations,
                            response: ProxyResponse::Complete(resp),
                        });
                    }
                    // Look-ahead heuristic buffer: a passthrough stream can't be
                    // escalated after the fact, so peek the leading frames, run the
                    // cheap degeneracy checks on the prefix, and escalate before any
                    // bytes reach the client if it's clearly failing. Otherwise
                    // replay the buffered frames and stream the rest live.
                    Ok(ProxyResponse::Stream(stream)) if esc.stream_lookahead => {
                        let (prefix, rest) = lookahead(stream, esc.lookahead_chars).await;
                        if prefix_is_degenerate(&prefix, esc) {
                            warn!(from = ?current, to = ?next_tier, "escalating after degenerate stream prefix");
                            current = next_tier;
                            escalations += 1;
                            continue;
                        }
                        let replay = futures::stream::iter(
                            prefix.frames.into_iter().map(Ok::<Bytes, reqwest::Error>),
                        )
                        .chain(rest);
                        return Ok(Forwarded {
                            served_tier: current,
                            escalations,
                            response: ProxyResponse::Stream(Box::new(replay)),
                        });
                    }
                    // Streaming with look-ahead off: forward verbatim.
                    Ok(response) => {
                        return Ok(Forwarded {
                            served_tier: current,
                            escalations,
                            response,
                        });
                    }
                }
            }

            return result.map(|response| Forwarded {
                served_tier: current.clone(),
                escalations,
                response,
            });
        }
    }

    async fn forward_once(
        &self,
        request: ChatCompletionRequest,
        tier: Tier,
    ) -> Result<ProxyResponse, ProxyError> {
        let provider_key = self.select_provider(&tier);
        let provider = self
            .config
            .providers
            .get(&provider_key)
            .ok_or_else(|| ProxyError::ProviderNotFound(provider_key.clone()))?;

        let (endpoint, headers) = self.prepare_request(provider).await?;
        let model = self.map_model(provider, &request.model);

        let mut request = request;
        request.model = model;

        let is_stream = request.stream.unwrap_or(false);

        info!(tier = ?tier, provider = %provider_key, endpoint = %endpoint, stream = is_stream, "Forwarding request");

        let mut body =
            serde_json::to_value(&request).map_err(|e| ProxyError::ParseError(e.to_string()))?;
        // Ask streaming providers to emit a final usage chunk so token counts
        // can be surfaced to the client.
        if is_stream && let Some(object) = body.as_object_mut() {
            object.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        }
        // Per-provider extra_body: shallow-merge provider-configured fields into the
        // request (provider wins). Lets a tier force e.g. chat_template_kwargs so a
        // local thinking-model answers directly instead of returning empty content.
        if let Some(extra) = &provider.extra_body
            && let (Some(object), Some(extra_obj)) = (body.as_object_mut(), extra.as_object())
        {
            for (k, v) in extra_obj {
                object.insert(k.clone(), v.clone());
            }
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", endpoint))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(ProxyError::RequestFailed)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Provider returned error");
            return Err(ProxyError::ProviderError {
                status: status.as_u16(),
                body,
            });
        }

        if is_stream {
            let stream = response.bytes_stream();
            Ok(ProxyResponse::Stream(Box::new(stream)))
        } else {
            let completion = response
                .json::<ChatCompletionResponse>()
                .await
                .map_err(|e| ProxyError::ParseError(e.to_string()))?;

            Ok(ProxyResponse::Complete(completion))
        }
    }

    /// Decides whether a completed cheap-tier response warrants escalation,
    /// combining heuristic degeneracy checks with an optional LLM judge.
    async fn should_escalate_complete(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
        cfg: &EscalationConfig,
    ) -> bool {
        if response_is_degenerate(response, cfg) {
            return true;
        }
        if cfg.judge.enabled {
            return !self.judge_adequate(request, response, &cfg.judge).await;
        }
        false
    }

    /// Asks a judge model to score how well `response` answers `request`. Returns
    /// true (adequate, don't escalate) when the score meets the threshold, and
    /// fails open (true) if the judge tier is misconfigured or unreachable.
    async fn judge_adequate(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
        cfg: &JudgeConfig,
    ) -> bool {
        let Some(tier) = Tier::from_name(&cfg.tier) else {
            warn!(tier = %cfg.tier, "invalid judge tier; skipping judge");
            return true;
        };

        let user_request = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| truncate(&m.content.as_text(), 1500))
            .unwrap_or_default();
        let answer = response
            .choices
            .first()
            .map(|c| truncate(&c.message.content.as_text(), 1500))
            .unwrap_or_default();

        let prompt = format!(
            "User request:\n{user_request}\n\nAssistant response:\n{answer}\n\n\
             On a scale of 1-5, how well does the response satisfy the request? \
             Reply with only the digit."
        );

        let judge_request = ChatCompletionRequest {
            model: request.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: MessageContent::Text(
                        "You are a strict QA reviewer. Reply with only a single digit 1-5."
                            .to_string(),
                    ),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                Message {
                    role: "user".to_string(),
                    content: MessageContent::Text(prompt),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
            ],
            max_tokens: Some(4),
            temperature: Some(0.0),
            top_p: None,
            stream: Some(false),
            tools: None,
            tool_choice: None,
        };

        match self.forward_once(judge_request, tier).await {
            Ok(ProxyResponse::Complete(judge)) => {
                let text = judge
                    .choices
                    .first()
                    .map(|c| c.message.content.as_text())
                    .unwrap_or_default();
                let score = text
                    .chars()
                    .find(|c| c.is_ascii_digit())
                    .and_then(|c| c.to_digit(10))
                    .unwrap_or(5) as u8;
                debug!(score, min = cfg.min_score, "judge verdict");
                score >= cfg.min_score
            }
            _ => true, // judge unavailable -> fail open, don't escalate
        }
    }

    fn select_provider(&self, tier: &Tier) -> String {
        tier.provider_key().to_string()
    }

    async fn prepare_request(
        &self,
        provider: &ProviderConfig,
    ) -> Result<(String, reqwest::header::HeaderMap), ProxyError> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());

        let auth_type = provider.auth_type.as_ref().unwrap_or(&AuthType::None);

        match auth_type {
            AuthType::ApiKey => {
                if let Some(key) = &provider.api_key {
                    headers.insert("Authorization", format!("Bearer {}", key).parse().unwrap());
                } else {
                    warn!("API key not configured for provider");
                }
            }
            AuthType::GitHubOAuth => {
                let token = self.get_github_token().await?;
                headers.insert(
                    "Authorization",
                    format!("Bearer {}", token).parse().unwrap(),
                );
                headers.insert("x-copilot-integration-id", "token-miser".parse().unwrap());
            }
            AuthType::None => {
                // No authentication required
            }
        }

        Ok((provider.endpoint.clone(), headers))
    }

    async fn get_github_token(&self) -> Result<String, ProxyError> {
        if let Some(token) = self.token_cache.read().await.get("github_oauth") {
            return Ok(token.clone());
        }

        // Copilot tokens are supplied via the environment (e.g. `gh auth token`);
        // cache the resolved value for subsequent requests.
        let token = std::env::var("GITHUB_COPILOT_TOKEN").map_err(|_| {
            ProxyError::AuthenticationFailed("GITHUB_COPILOT_TOKEN not set".to_string())
        })?;

        self.token_cache
            .write()
            .await
            .insert("github_oauth".to_string(), token.clone());

        Ok(token)
    }

    fn map_model(&self, provider: &ProviderConfig, requested_model: &str) -> String {
        if let Some(mapping) = &provider.model_mapping {
            mapping
                .get(requested_model)
                .or_else(|| mapping.get("default"))
                .cloned()
                .unwrap_or_else(|| requested_model.to_string())
        } else {
            requested_model.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Choice, Message, MessageContent, Usage};

    fn response(content: &str, finish_reason: &str) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "x".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(content.to_string()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                finish_reason: finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            system_fingerprint: None,
        }
    }

    #[test]
    fn test_response_is_degenerate() {
        let cfg = EscalationConfig::default(); // on_empty_response = true
        assert!(response_is_degenerate(&response("", "stop"), &cfg));
        assert!(response_is_degenerate(&response("   \n", "stop"), &cfg));
        assert!(response_is_degenerate(
            &response("blocked", "content_filter"),
            &cfg
        ));
        assert!(!response_is_degenerate(
            &response("here is your answer", "stop"),
            &cfg
        ));

        // Truncation only counts when enabled.
        assert!(!response_is_degenerate(
            &response("partial", "length"),
            &cfg
        ));
        let cfg_trunc = EscalationConfig {
            on_truncation: true,
            ..EscalationConfig::default()
        };
        assert!(response_is_degenerate(
            &response("partial", "length"),
            &cfg_trunc
        ));
    }

    #[test]
    fn test_tool_call_turn_is_not_empty() {
        // A tool-call response has empty text content but must NOT be treated as
        // a degenerate empty answer, even with on_empty_response on (the default).
        let cfg = EscalationConfig::default();
        let mut resp = response("", "tool_calls");
        resp.choices[0].message.tool_calls = Some(vec![crate::models::ToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            function: crate::models::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]);
        assert!(!response_is_degenerate(&resp, &cfg));
    }

    fn byte_stream(chunks: Vec<&'static [u8]>) -> ByteStream {
        let items: Vec<Result<Bytes, reqwest::Error>> = chunks
            .into_iter()
            .map(|c| Ok(Bytes::from_static(c)))
            .collect();
        Box::new(futures::stream::iter(items))
    }

    async fn drain(stream: ByteStream) -> String {
        let mut out = String::new();
        let mut s = Box::pin(stream);
        while let Some(c) = s.next().await {
            out.push_str(std::str::from_utf8(&c.unwrap()).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn test_lookahead_parses_and_replays_verbatim() {
        let frames = vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n" as &[u8],
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            b"data: [DONE]\n\n",
        ];
        let original: String = frames
            .iter()
            .map(|f| std::str::from_utf8(f).unwrap())
            .collect();

        let (prefix, rest) = lookahead(byte_stream(frames), 240).await;
        assert_eq!(prefix.text, "Hello world");
        assert_eq!(prefix.finish.as_deref(), Some("stop"));
        assert!(!prefix_is_degenerate(&prefix, &EscalationConfig::default()));

        // buffered prefix + un-consumed remainder must reproduce the input exactly.
        let replay =
            futures::stream::iter(prefix.frames.into_iter().map(Ok::<Bytes, reqwest::Error>))
                .chain(rest);
        assert_eq!(drain(Box::new(replay)).await, original);
    }

    #[tokio::test]
    async fn test_lookahead_flags_refusal_prefix() {
        let frames = vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"I can't help with that.\"}}]}\n\n"
                as &[u8],
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            b"data: [DONE]\n\n",
        ];
        let cfg = EscalationConfig {
            on_refusal: true,
            ..EscalationConfig::default()
        };
        let (prefix, _rest) = lookahead(byte_stream(frames), 240).await;
        assert!(prefix_is_degenerate(&prefix, &cfg));
    }

    #[tokio::test]
    async fn test_lookahead_tool_call_stream_not_empty() {
        // Empty text content but a tool call + finish "tool_calls" -> not degenerate.
        let frames = vec![
            b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"ls\",\"arguments\":\"\"}}]}}]}\n\n" as &[u8],
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            b"data: [DONE]\n\n",
        ];
        let (prefix, _rest) = lookahead(byte_stream(frames), 240).await;
        assert!(prefix.saw_tool_call);
        assert!(prefix.text.trim().is_empty());
        assert!(!prefix_is_degenerate(&prefix, &EscalationConfig::default()));
    }

    #[tokio::test]
    async fn test_lookahead_empty_stream_is_degenerate() {
        // Stream ends with no content and no tool call -> empty -> degenerate.
        let frames = vec![b"data: [DONE]\n\n" as &[u8]];
        let (prefix, _rest) = lookahead(byte_stream(frames), 240).await;
        assert!(prefix.ended);
        assert!(prefix_is_degenerate(&prefix, &EscalationConfig::default()));
    }

    #[test]
    fn test_looks_like_refusal() {
        assert!(looks_like_refusal("I'm sorry, I can't help with that."));
        assert!(looks_like_refusal(
            "As an AI language model, I cannot provide that."
        ));
        assert!(!looks_like_refusal(
            "Here is a function that reverses a list."
        ));
        // A long, substantive answer that merely mentions a phrase is not a refusal.
        let long =
            "Here is the implementation. ".repeat(40) + "Note: i can't help if input is null.";
        assert!(!looks_like_refusal(&long));
    }

    #[test]
    fn test_should_escalate_error() {
        assert!(should_escalate_error(&ProxyError::ProviderError {
            status: 503,
            body: String::new()
        }));
        assert!(should_escalate_error(&ProxyError::ProviderError {
            status: 429,
            body: String::new()
        }));
        assert!(should_escalate_error(&ProxyError::ParseError("x".into())));
        // Deterministic client/config errors should not escalate.
        assert!(!should_escalate_error(&ProxyError::ProviderError {
            status: 400,
            body: String::new()
        }));
        assert!(!should_escalate_error(&ProxyError::AuthenticationFailed(
            "x".into()
        )));
        assert!(!should_escalate_error(&ProxyError::ProviderNotFound(
            "x".into()
        )));
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("Provider returned status {status}")]
    ProviderError { status: u16, body: String },
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
}
