mod anthropic_stream;
mod cluster;
mod config;
mod discovery;
mod explore;
mod intent;
mod models;
mod proxy;
mod router;
mod semantic;
mod telemetry;

use axum::{
    Json, Router as AxumRouter,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use futures::Stream;
use serde_json::json;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
struct AppState {
    config: Arc<config::Config>,
    router: Arc<router::Router>,
    proxy: Arc<proxy::Proxy>,
    discovery: Arc<tokio::sync::Mutex<discovery::ModelDiscovery>>,
    telemetry: telemetry::Telemetry,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        // Logs go to stderr so stdout stays clean — the `explore` subcommand
        // prints machine-readable JSON there for external callers to pipe.
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    let _ = dotenvy::dotenv();

    let config = Arc::new(load_config());

    // Standalone `explore` subcommand: run the FastContext explorer once and print
    // the Evidence as JSON, so an external caller (a coding agent, an implementer
    // step) can gather repo context without going through the proxy.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("explore") {
        return run_explore_cli(&args[2..], &config).await;
    }

    let mut router = router::Router::new(config.routing.clone());
    if let Some(cluster) = &config.private_cluster {
        info!("Private cluster configured; enabling cluster intent classifier");
        router = router.with_cluster(cluster.clone());
    }
    if config.semantic_router.enabled {
        info!("Semantic router enabled; routing by embedding similarity");
        router = router.with_semantic_router(config.semantic_router.clone());
    }
    let router = Arc::new(router);

    let proxy = Arc::new(proxy::Proxy::new(config.clone()));
    let discovery = Arc::new(tokio::sync::Mutex::new(discovery::ModelDiscovery::new()));
    let telemetry = telemetry::Telemetry::from_config(&config);

    let state = AppState {
        config: config.clone(),
        router,
        proxy,
        discovery,
        telemetry,
    };

    let app = AxumRouter::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/messages", post(handle_messages))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Loads configuration from `TOKEN_MISER_CONFIG`, then `./config.toml`, falling
/// back to built-in defaults when neither is present or a file fails to load.
fn load_config() -> config::Config {
    let mut config = if let Ok(path) = std::env::var("TOKEN_MISER_CONFIG") {
        load_config_file(&path)
    } else if std::path::Path::new("config.toml").exists() {
        load_config_file("config.toml")
    } else {
        info!("No config file found; using default configuration");
        config::Config::default()
    };
    // Per-launch env overrides (port / fastcontext / repo_root) win over the file.
    config.apply_env_overrides();
    config
}

fn load_config_file(path: &str) -> config::Config {
    match config::Config::from_file(path) {
        Ok(config) => {
            info!("Loaded configuration from {path}");
            config
        }
        Err(e) => {
            warn!("Failed to load config from {path}: {e}; using defaults");
            config::Config::default()
        }
    }
}

/// Runs the `explore` subcommand: `token_miser explore --query <text>
/// [--repo-root <path>]`. Resolves `repo_root` (defaulting to `explore.repo_root`
/// from config), runs the explorer, and prints the `Evidence` as pretty JSON.
async fn run_explore_cli(
    args: &[String],
    config: &config::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut query: Option<String> = None;
    let mut repo_root = config.explore.repo_root.clone();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--query" | "-q" => {
                i += 1;
                query = args.get(i).cloned();
            }
            "--repo-root" | "-r" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    repo_root = v.clone();
                }
            }
            other => return Err(format!("unknown explore argument: {other}").into()),
        }
        i += 1;
    }

    let query = query.ok_or("explore requires --query <text>")?;
    let evidence = explore::explore_repo(
        &query,
        std::path::Path::new(&repo_root),
        &config.explore,
        &config.fastcontext,
    )
    .await?;
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}

/// Resolves the effective route (applying shadow mode) and seeds a telemetry
/// record with the pre-flight fields.
fn build_route(
    state: &AppState,
    request: &models::ChatCompletionRequest,
    endpoint: &'static str,
    classified: router::Tier,
) -> (router::Tier, telemetry::RoutingRecord) {
    let effective = state.telemetry.effective_tier(&classified);
    // Always estimated: used for the Anthropic message_start usage and telemetry.
    let input_tokens = state.router.estimate_tokens(request) as u64;
    let record = telemetry::RoutingRecord {
        endpoint,
        model: request.model.clone(),
        classified_tier: format!("{classified:?}"),
        effective_tier: format!("{effective:?}"),
        served_tier: format!("{effective:?}"),
        escalations: 0,
        shadow: state.telemetry.is_shadow(),
        stream: request.stream.unwrap_or(false),
        input_tokens,
        output_tokens: None,
        estimated_cost_usd: None,
        latency_ms: 0,
        // Set after forwarding (served tier) and after the explore stage runs.
        premium_escalation: false,
        explore_ran: false,
        pre_explore_input_tokens: input_tokens,
        post_explore_input_tokens: input_tokens,
        explore_turns: 0,
        explore_citations: 0,
        explore_expanded_tokens: 0,
    };
    (effective, record)
}

/// Per-request signals from the upstream explore stage, folded into telemetry.
#[derive(Default)]
struct ExploreMetrics {
    ran: bool,
    pre_tokens: u64,
    post_tokens: u64,
    turns: usize,
    citations: usize,
    expanded_tokens: usize,
}

/// A fresh task has no prior assistant/tool turns — exploring mid agentic-loop
/// would re-gather context on every round-trip, so the stage runs only here.
fn is_fresh_task(request: &models::ChatCompletionRequest) -> bool {
    !request
        .messages
        .iter()
        .any(|m| m.role == "assistant" || m.role == "tool")
}

/// The query the explorer searches for: the latest user turn's text.
fn explore_query(request: &models::ChatCompletionRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_text())
        .filter(|q| !q.trim().is_empty())
}

/// Injects the auto-gathered evidence as a system message, placed after any
/// leading system messages and before the first user/assistant turn.
fn inject_evidence(request: &mut models::ChatCompletionRequest, context: String) {
    let msg = models::Message {
        role: "system".to_string(),
        content: models::MessageContent::Text(context),
        name: None,
        tool_call_id: None,
        tool_calls: None,
    };
    let pos = request
        .messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(request.messages.len());
    request.messages.insert(pos, msg);
}

/// OPTIONAL upstream stage: when `fastcontext.enabled` and this is a fresh task,
/// run the FastContext explorer and inject its cleaned evidence into `request`,
/// so the router classifies a targeted prompt. Any failure is logged and
/// swallowed — the request proceeds unchanged. Returns metrics for telemetry.
async fn run_explore_stage(
    state: &AppState,
    request: &mut models::ChatCompletionRequest,
) -> ExploreMetrics {
    let pre = state.router.estimate_tokens(request) as u64;
    let mut metrics = ExploreMetrics {
        pre_tokens: pre,
        post_tokens: pre,
        ..Default::default()
    };

    if !state.config.fastcontext.enabled || !is_fresh_task(request) {
        return metrics;
    }
    let Some(query) = explore_query(request) else {
        return metrics;
    };

    let repo_root = std::path::PathBuf::from(&state.config.explore.repo_root);
    match explore::explore_repo(
        &query,
        &repo_root,
        &state.config.explore,
        &state.config.fastcontext,
    )
    .await
    {
        Ok(evidence) => {
            if let Some(context) = evidence.to_context_message() {
                inject_evidence(request, context);
                metrics.ran = true;
                metrics.turns = evidence.stats.turns;
                metrics.citations = evidence.citations.len();
                metrics.expanded_tokens = evidence.stats.expanded_tokens;
                metrics.post_tokens = state.router.estimate_tokens(request) as u64;
                info!(
                    pre = metrics.pre_tokens,
                    post = metrics.post_tokens,
                    citations = metrics.citations,
                    "explore stage injected repository context"
                );
            }
        }
        Err(e) => warn!(error = %e, "explore stage failed; proceeding without evidence"),
    }
    metrics
}

/// Copies the explore metrics into the telemetry record.
fn apply_explore_metrics(record: &mut telemetry::RoutingRecord, metrics: &ExploreMetrics) {
    record.explore_ran = metrics.ran;
    record.pre_explore_input_tokens = metrics.pre_tokens;
    record.post_explore_input_tokens = metrics.post_tokens;
    record.explore_turns = metrics.turns;
    record.explore_citations = metrics.citations;
    record.explore_expanded_tokens = metrics.expanded_tokens;
}

/// Fills in actual token usage and estimated cost from a non-streaming response.
fn finalize_record(
    state: &AppState,
    record: &mut telemetry::RoutingRecord,
    tier: &router::Tier,
    usage: &models::Usage,
) {
    record.input_tokens = usage.prompt_tokens as u64;
    record.output_tokens = Some(usage.completion_tokens as u64);
    record.estimated_cost_usd = state.telemetry.cost_usd(
        &state.config,
        tier,
        usage.prompt_tokens as u64,
        usage.completion_tokens as u64,
    );
}

async fn handle_chat_completions(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(request): Json<models::ChatCompletionRequest>,
) -> Result<Response, AppError> {
    info!(model = %request.model, messages = request.messages.len(), "Received chat completion request");

    let mut request = request;
    let explore_metrics = run_explore_stage(&state, &mut request).await;

    let classified = state.router.classify(&request).await;
    info!(tier = ?classified, "Classified request");
    let (tier, mut record) = build_route(&state, &request, "/v1/chat/completions", classified);
    apply_explore_metrics(&mut record, &explore_metrics);

    let started = std::time::Instant::now();
    let fwd = state.proxy.forward(request, tier).await?;
    record.served_tier = format!("{:?}", fwd.served_tier);
    record.escalations = fwd.escalations;
    record.premium_escalation = fwd.served_tier == router::Tier::Complex;
    let served = fwd.served_tier;

    match fwd.response {
        proxy::ProxyResponse::Complete(completion) => {
            record.latency_ms = started.elapsed().as_millis() as u64;
            finalize_record(&state, &mut record, &served, &completion.usage);
            state.telemetry.record(&record).await;
            Ok(Json(completion).into_response())
        }
        proxy::ProxyResponse::Stream(stream) => {
            if state.telemetry.enabled() {
                let metered = telemetry::meter(
                    stream,
                    state.telemetry.clone(),
                    state.config.clone(),
                    served,
                    record,
                    "completion_tokens",
                    started,
                );
                Ok(sse_response(Body::from_stream(metered)))
            } else {
                Ok(sse_passthrough(stream))
            }
        }
    }
}

async fn handle_messages(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(request): Json<models::AnthropicRequest>,
) -> Result<Response, AppError> {
    info!(model = %request.model, messages = request.messages.len(), "Received Anthropic messages request");

    let mut openai_request = convert_anthropic_to_openai(request);
    let explore_metrics = run_explore_stage(&state, &mut openai_request).await;

    let classified = state.router.classify(&openai_request).await;
    info!(tier = ?classified, "Classified request");
    let (tier, mut record) = build_route(&state, &openai_request, "/v1/messages", classified);
    apply_explore_metrics(&mut record, &explore_metrics);

    let started = std::time::Instant::now();
    let fwd = state.proxy.forward(openai_request, tier).await?;
    record.served_tier = format!("{:?}", fwd.served_tier);
    record.escalations = fwd.escalations;
    record.premium_escalation = fwd.served_tier == router::Tier::Complex;
    let served = fwd.served_tier;

    match fwd.response {
        proxy::ProxyResponse::Complete(completion) => {
            record.latency_ms = started.elapsed().as_millis() as u64;
            finalize_record(&state, &mut record, &served, &completion.usage);
            state.telemetry.record(&record).await;
            let anthropic_response = convert_openai_to_anthropic(completion);
            Ok(Json(anthropic_response).into_response())
        }
        // Anthropic clients expect the Messages event format, so translate the
        // upstream OpenAI chunk stream rather than forwarding it verbatim.
        proxy::ProxyResponse::Stream(stream) => {
            let translated = anthropic_stream::translate(stream, record.input_tokens);
            if state.telemetry.enabled() {
                let metered = telemetry::meter(
                    Box::pin(translated),
                    state.telemetry.clone(),
                    state.config.clone(),
                    served,
                    record,
                    "output_tokens",
                    started,
                );
                Ok(sse_response(Body::from_stream(metered)))
            } else {
                Ok(sse_response(Body::from_stream(translated)))
            }
        }
    }
}

/// Lists available models discovered across the configured providers, in the
/// OpenAI `/v1/models` response shape.
async fn handle_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut discovery = state.discovery.lock().await;
    let mut models: Vec<serde_json::Value> = Vec::new();

    if let Ok(copilot) = discovery.discover_copilot_models().await {
        models.extend(copilot.into_iter().map(model_entry));
    }
    if let Ok(claude) = discovery.discover_claude_models().await {
        models.extend(claude.into_iter().map(model_entry));
    }
    if let Some(cluster) = &state.config.private_cluster
        && let Ok(cluster_models) = discovery.check_cluster_models(&cluster.endpoint).await
    {
        models.extend(cluster_models.into_iter().map(model_entry));
    }

    Json(json!({ "object": "list", "data": models }))
}

fn model_entry(model: discovery::CopilotModel) -> serde_json::Value {
    json!({ "id": model.id, "object": "model", "owned_by": model.billing.tier })
}

/// Liveness probe; reports private-cluster health when one is configured.
async fn handle_health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "cluster_healthy": state.router.cluster_health().await,
    }))
}

/// Builds an SSE streaming response with the standard event-stream headers.
fn sse_response(body: Body) -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("static SSE headers are always valid")
}

/// Relays an upstream SSE byte stream to the client verbatim.
///
/// The upstream response is already SSE-framed (`data: ...\n\n`), so forwarding
/// the raw bytes preserves event boundaries and the terminal `[DONE]` sentinel.
/// Re-wrapping each chunk in an `Event` would double the `data:` prefix and split
/// frames on arbitrary TCP chunk boundaries.
fn sse_passthrough(
    stream: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>,
) -> Response {
    sse_response(Body::from_stream(stream))
}

fn convert_anthropic_to_openai(
    anthropic: models::AnthropicRequest,
) -> models::ChatCompletionRequest {
    let mut messages: Vec<models::Message> = Vec::with_capacity(anthropic.messages.len() + 1);

    if let Some(system) = anthropic.system {
        // `system` may be a string or an array of (text) content blocks; flatten
        // the block form by joining the text segments so the downstream OpenAI
        // system message is always a plain string.
        let system_text = match system {
            models::AnthropicContent::Text(text) => text,
            models::AnthropicContent::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(models::Message {
            role: "system".to_string(),
            content: models::MessageContent::Text(system_text),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        });
    }

    for msg in anthropic.messages {
        match msg.content {
            models::AnthropicContent::Text(text) => messages.push(models::Message {
                role: msg.role,
                content: models::MessageContent::Text(text),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            }),
            models::AnthropicContent::Blocks(blocks) => {
                let mut parts: Vec<models::ContentPart> = Vec::new();
                let mut tool_calls: Vec<models::ToolCall> = Vec::new();
                // tool_result blocks become standalone OpenAI `tool` messages.
                let mut tool_results: Vec<models::Message> = Vec::new();

                for block in blocks {
                    match block.block_type.as_str() {
                        "text" => parts.push(models::ContentPart {
                            part_type: "text".to_string(),
                            text: block.text,
                            image_url: None,
                        }),
                        "image" => {
                            if let Some(source) = block.source {
                                parts.push(models::ContentPart {
                                    part_type: "image_url".to_string(),
                                    text: None,
                                    image_url: Some(models::ImageUrl {
                                        url: format!(
                                            "data:{};base64,{}",
                                            source.media_type, source.data
                                        ),
                                    }),
                                });
                            }
                        }
                        // The assistant's prior tool call -> OpenAI `tool_calls`.
                        "tool_use" => tool_calls.push(models::ToolCall {
                            id: block.id.unwrap_or_default(),
                            r#type: "function".to_string(),
                            function: models::FunctionCall {
                                name: block.name.unwrap_or_default(),
                                arguments: block
                                    .input
                                    .map(|v| v.to_string())
                                    .unwrap_or_else(|| "{}".to_string()),
                            },
                        }),
                        // The user feeding a result back -> OpenAI `tool` message.
                        "tool_result" => tool_results.push(models::Message {
                            role: "tool".to_string(),
                            content: models::MessageContent::Text(
                                block.content.map(|c| c.as_text()).unwrap_or_default(),
                            ),
                            name: None,
                            tool_call_id: block.tool_use_id,
                            tool_calls: None,
                        }),
                        _ => {}
                    }
                }

                // OpenAI ordering: a `tool` message must directly follow the
                // assistant message whose call it answers. An Anthropic user turn
                // carrying tool_result blocks IS that follow-up, so emit those
                // first, then any remaining text/image from the same turn.
                messages.append(&mut tool_results);

                let has_content = parts.iter().any(|p| {
                    p.image_url.is_some() || p.text.as_deref().is_some_and(|t| !t.is_empty())
                });

                if !tool_calls.is_empty() {
                    messages.push(models::Message {
                        role: msg.role,
                        content: parts_to_content(parts),
                        name: None,
                        tool_call_id: None,
                        tool_calls: Some(tool_calls),
                    });
                } else if has_content {
                    messages.push(models::Message {
                        role: msg.role,
                        content: parts_to_content(parts),
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
            }
        }
    }

    let tools = anthropic.tools.map(|tools| {
        tools
            .into_iter()
            .map(|tool| models::Tool {
                r#type: "function".to_string(),
                function: models::FunctionDef {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                },
            })
            .collect()
    });

    models::ChatCompletionRequest {
        model: anthropic.model,
        messages,
        max_tokens: anthropic.max_tokens,
        temperature: anthropic.temperature,
        top_p: None,
        stream: anthropic.stream,
        tools,
        tool_choice: None,
    }
}

/// Collapses converted content parts into the simplest OpenAI message content:
/// empty text when there's nothing, plain text when there are no images, and a
/// multimodal `Parts` array only when an image is present.
fn parts_to_content(parts: Vec<models::ContentPart>) -> models::MessageContent {
    if parts.is_empty() {
        return models::MessageContent::Text(String::new());
    }
    if parts.iter().any(|p| p.image_url.is_some()) {
        models::MessageContent::Parts(parts)
    } else {
        let text = parts
            .iter()
            .filter_map(|p| p.text.clone())
            .collect::<Vec<_>>()
            .join("");
        models::MessageContent::Text(text)
    }
}

fn convert_openai_to_anthropic(
    openai: models::ChatCompletionResponse,
) -> models::AnthropicResponse {
    // Derive the Anthropic stop_reason from the first choice before consuming it;
    // a tool call must surface as `tool_use` so the client resumes the loop.
    let stop_reason = openai
        .choices
        .first()
        .map(|c| match c.finish_reason.as_str() {
            "tool_calls" | "function_call" => "tool_use",
            "length" => "max_tokens",
            _ => "end_turn",
        })
        .unwrap_or("end_turn")
        .to_string();

    let content: Vec<models::ContentBlock> = openai
        .choices
        .into_iter()
        .flat_map(|choice| {
            let mut blocks: Vec<models::ContentBlock> = Vec::new();
            match choice.message.content {
                models::MessageContent::Text(text) => {
                    if !text.is_empty() {
                        blocks.push(models::ContentBlock {
                            block_type: "text".to_string(),
                            text: Some(text),
                            ..Default::default()
                        });
                    }
                }
                models::MessageContent::Parts(parts) => {
                    blocks.extend(parts.into_iter().map(|part| models::ContentBlock {
                        block_type: part.part_type,
                        text: part.text,
                        ..Default::default()
                    }));
                }
            }
            // tool_calls -> Anthropic tool_use blocks (round-trips the request-side
            // translation, so a non-streamed tool turn isn't lost).
            if let Some(tool_calls) = choice.message.tool_calls {
                for tc in tool_calls {
                    blocks.push(models::ContentBlock {
                        block_type: "tool_use".to_string(),
                        id: Some(tc.id),
                        name: Some(tc.function.name),
                        input: Some(
                            serde_json::from_str(&tc.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({})),
                        ),
                        ..Default::default()
                    });
                }
            }
            blocks
        })
        .collect();

    models::AnthropicResponse {
        id: openai.id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: openai.model,
        stop_reason: Some(stop_reason),
        usage: models::AnthropicUsage {
            input_tokens: openai.usage.prompt_tokens,
            output_tokens: openai.usage.completion_tokens,
        },
    }
}

struct AppError(proxy::ProxyError);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!(error = ?self.0, "Request failed");

        match self.0 {
            // Forward the provider's real status and error body so clients can
            // distinguish 429/401/400 and react (back off, fix auth) correctly.
            proxy::ProxyError::ProviderError { status, body } => {
                let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
                (code, [(header::CONTENT_TYPE, "application/json")], body).into_response()
            }
            proxy::ProxyError::ProviderNotFound(_) => {
                (StatusCode::NOT_FOUND, "Provider not found").into_response()
            }
            proxy::ProxyError::RequestFailed(_) => {
                (StatusCode::BAD_GATEWAY, "Request to provider failed").into_response()
            }
            proxy::ProxyError::ParseError(_) => {
                (StatusCode::BAD_GATEWAY, "Failed to parse response").into_response()
            }
            proxy::ProxyError::AuthenticationFailed(_) => {
                (StatusCode::UNAUTHORIZED, "Authentication failed").into_response()
            }
        }
    }
}

impl From<proxy::ProxyError> for AppError {
    fn from(err: proxy::ProxyError) -> Self {
        AppError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_req(messages: Vec<(&str, &str)>) -> models::ChatCompletionRequest {
        models::ChatCompletionRequest {
            model: "m".to_string(),
            messages: messages
                .into_iter()
                .map(|(role, text)| models::Message {
                    role: role.to_string(),
                    content: models::MessageContent::Text(text.to_string()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                })
                .collect(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            tools: None,
            tool_choice: None,
        }
    }

    #[test]
    fn test_fresh_task_gate() {
        // System + a single user turn is a fresh task.
        assert!(is_fresh_task(&user_req(vec![
            ("system", "be terse"),
            ("user", "explain the router"),
        ])));
        // Any assistant or tool turn means we're mid agentic-loop -> not fresh.
        assert!(!is_fresh_task(&user_req(vec![
            ("user", "go"),
            ("assistant", "working on it"),
        ])));
        assert!(!is_fresh_task(&user_req(vec![
            ("user", "go"),
            ("tool", "result"),
        ])));
    }

    #[test]
    fn test_explore_query_picks_latest_user_turn() {
        let req = user_req(vec![
            ("system", "sys"),
            ("user", "first"),
            ("user", "second"),
        ]);
        assert_eq!(explore_query(&req).as_deref(), Some("second"));
        // No user turn -> no query.
        assert_eq!(explore_query(&user_req(vec![("system", "sys")])), None);
    }

    #[test]
    fn test_inject_evidence_after_leading_system_messages() {
        let mut req = user_req(vec![("system", "sys"), ("user", "task")]);
        inject_evidence(&mut req, "CONTEXT".to_string());
        assert_eq!(req.messages.len(), 3);
        // Evidence sits after the system prompt, before the user turn.
        assert_eq!(req.messages[0].role, "system");
        assert_eq!(req.messages[0].content.as_text(), "sys");
        assert_eq!(req.messages[1].role, "system");
        assert_eq!(req.messages[1].content.as_text(), "CONTEXT");
        assert_eq!(req.messages[2].role, "user");
    }

    #[test]
    fn test_provider_error_preserves_upstream_status() {
        let err = AppError(proxy::ProxyError::ProviderError {
            status: 429,
            body: "{\"error\":\"rate limited\"}".to_string(),
        });
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_anthropic_system_prompt_becomes_system_message() {
        let request = models::AnthropicRequest {
            model: "claude".to_string(),
            messages: vec![models::AnthropicMessage {
                role: "user".to_string(),
                content: models::AnthropicContent::Text("go".to_string()),
            }],
            max_tokens: None,
            system: Some(models::AnthropicContent::Text(
                "You are an architect".to_string(),
            )),
            temperature: None,
            stream: None,
            tools: None,
        };

        let converted = convert_anthropic_to_openai(request);

        assert_eq!(converted.messages.len(), 2);
        assert_eq!(converted.messages[0].role, "system");
        match &converted.messages[0].content {
            models::MessageContent::Text(t) => assert_eq!(t, "You are an architect"),
            _ => panic!("expected text system content"),
        }
        assert_eq!(converted.messages[1].role, "user");
    }

    #[test]
    fn test_anthropic_system_blocks_deserialize_and_flatten() {
        // Claude Code sends `system` as an array of text content blocks (with
        // cache_control). Ensure it deserializes and flattens to a single string.
        let body = r#"{
            "model": "claude",
            "messages": [{"role": "user", "content": "go"}],
            "system": [
                {"type": "text", "text": "line one", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "line two"}
            ]
        }"#;
        let request: models::AnthropicRequest = serde_json::from_str(body).unwrap();
        let converted = convert_anthropic_to_openai(request);

        assert_eq!(converted.messages[0].role, "system");
        match &converted.messages[0].content {
            models::MessageContent::Text(t) => assert_eq!(t, "line one\nline two"),
            _ => panic!("expected text system content"),
        }
    }

    #[test]
    fn test_anthropic_tools_are_preserved() {
        let request = models::AnthropicRequest {
            model: "claude".to_string(),
            messages: vec![models::AnthropicMessage {
                role: "user".to_string(),
                content: models::AnthropicContent::Text("use the tool".to_string()),
            }],
            max_tokens: None,
            system: None,
            temperature: None,
            stream: None,
            tools: Some(vec![models::AnthropicTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                input_schema: json!({"type": "object"}),
            }]),
        };

        let converted = convert_anthropic_to_openai(request);

        let tools = converted.tools.expect("tools should be preserved");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].function.parameters, json!({"type": "object"}));
    }

    // An assistant turn carrying a tool_use block must become an OpenAI assistant
    // message with `tool_calls` (id/name preserved, input JSON-encoded), not an
    // empty text message — otherwise the model loses its own prior call.
    #[test]
    fn test_tool_use_block_becomes_tool_calls() {
        let request = models::AnthropicRequest {
            model: "claude".to_string(),
            messages: vec![models::AnthropicMessage {
                role: "assistant".to_string(),
                content: models::AnthropicContent::Blocks(vec![
                    models::ContentBlock {
                        block_type: "text".to_string(),
                        text: Some("Let me check.".to_string()),
                        ..Default::default()
                    },
                    models::ContentBlock {
                        block_type: "tool_use".to_string(),
                        id: Some("toolu_1".to_string()),
                        name: Some("get_weather".to_string()),
                        input: Some(json!({"city": "Berlin"})),
                        ..Default::default()
                    },
                ]),
            }],
            max_tokens: None,
            system: None,
            temperature: None,
            stream: None,
            tools: None,
        };

        let converted = convert_anthropic_to_openai(request);

        assert_eq!(converted.messages.len(), 1);
        let msg = &converted.messages[0];
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.as_text(), "Let me check.");
        let calls = msg.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
            json!({"city": "Berlin"})
        );
    }

    // A user turn carrying a tool_result block must become an OpenAI `tool`
    // message with the matching tool_call_id, not a dropped/empty user message.
    #[test]
    fn test_tool_result_block_becomes_tool_message() {
        let request = models::AnthropicRequest {
            model: "claude".to_string(),
            messages: vec![models::AnthropicMessage {
                role: "user".to_string(),
                content: models::AnthropicContent::Blocks(vec![models::ContentBlock {
                    block_type: "tool_result".to_string(),
                    tool_use_id: Some("toolu_1".to_string()),
                    content: Some(models::ToolResultContent::Text("18°C, clear".to_string())),
                    ..Default::default()
                }]),
            }],
            max_tokens: None,
            system: None,
            temperature: None,
            stream: None,
            tools: None,
        };

        let converted = convert_anthropic_to_openai(request);

        assert_eq!(converted.messages.len(), 1);
        let msg = &converted.messages[0];
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(msg.content.as_text(), "18°C, clear");
    }

    // The full agentic loop: user asks -> assistant calls a tool -> user returns
    // the result. All three turns must survive, in OpenAI order, with ids intact.
    #[test]
    fn test_full_tool_loop_round_trips() {
        let request = models::AnthropicRequest {
            model: "claude".to_string(),
            messages: vec![
                models::AnthropicMessage {
                    role: "user".to_string(),
                    content: models::AnthropicContent::Text("weather in Berlin?".to_string()),
                },
                models::AnthropicMessage {
                    role: "assistant".to_string(),
                    content: models::AnthropicContent::Blocks(vec![models::ContentBlock {
                        block_type: "tool_use".to_string(),
                        id: Some("toolu_1".to_string()),
                        name: Some("get_weather".to_string()),
                        input: Some(json!({"city": "Berlin"})),
                        ..Default::default()
                    }]),
                },
                models::AnthropicMessage {
                    role: "user".to_string(),
                    content: models::AnthropicContent::Blocks(vec![models::ContentBlock {
                        block_type: "tool_result".to_string(),
                        tool_use_id: Some("toolu_1".to_string()),
                        content: Some(models::ToolResultContent::Text("18°C".to_string())),
                        ..Default::default()
                    }]),
                },
            ],
            max_tokens: None,
            system: None,
            temperature: None,
            stream: None,
            tools: None,
        };

        let converted = convert_anthropic_to_openai(request);

        assert_eq!(converted.messages.len(), 3);
        assert_eq!(converted.messages[0].role, "user");
        assert_eq!(converted.messages[1].role, "assistant");
        assert!(converted.messages[1].tool_calls.is_some());
        assert_eq!(converted.messages[2].role, "tool");
        assert_eq!(
            converted.messages[2].tool_call_id.as_deref(),
            Some("toolu_1")
        );
    }

    // Response direction: an OpenAI tool call must surface as an Anthropic
    // tool_use block with stop_reason "tool_use" so the client resumes the loop.
    #[test]
    fn test_openai_tool_calls_become_tool_use() {
        let response = models::ChatCompletionResponse {
            id: "cmpl-1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "gpt".to_string(),
            choices: vec![models::Choice {
                index: 0,
                message: models::Message {
                    role: "assistant".to_string(),
                    content: models::MessageContent::Text(String::new()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: Some(vec![models::ToolCall {
                        id: "call_1".to_string(),
                        r#type: "function".to_string(),
                        function: models::FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"city":"Berlin"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: "tool_calls".to_string(),
            }],
            usage: models::Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            system_fingerprint: None,
        };

        let anthropic = convert_openai_to_anthropic(response);

        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(anthropic.content.len(), 1);
        let block = &anthropic.content[0];
        assert_eq!(block.block_type, "tool_use");
        assert_eq!(block.id.as_deref(), Some("call_1"));
        assert_eq!(block.name.as_deref(), Some("get_weather"));
        assert_eq!(block.input, Some(json!({"city": "Berlin"})));
    }

    #[tokio::test]
    async fn test_sse_passthrough_forwards_bytes_verbatim() {
        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from_static(b"data: {\"choices\":[]}\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let stream = futures::stream::iter(chunks);
        let boxed: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin> =
            Box::new(stream);

        let response = sse_passthrough(boxed);

        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n");
    }
}
