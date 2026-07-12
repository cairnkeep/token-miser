use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    // Reasoning models (e.g. glm-5, qwen) return `content: null` when the answer
    // lives only in `reasoning_content` — notably on truncation. Tolerate a
    // null/absent content by mapping it to empty text so the proxy can still
    // parse the response and apply its escalation heuristics.
    #[serde(
        default = "empty_message_content",
        deserialize_with = "deserialize_message_content"
    )]
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Flattens content to plain text, joining the text of multimodal parts.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// The fallback used when a message carries no usable content.
fn empty_message_content() -> MessageContent {
    MessageContent::Text(String::new())
}

/// Deserializes `Message.content`, tolerating an explicit `null` (which reasoning
/// models emit when the reply lives only in `reasoning_content`) by mapping it to
/// empty text. Without this, `null` fails to match either untagged variant and
/// the whole response fails to parse — breaking escalation and the client.
fn deserialize_message_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<MessageContent>::deserialize(deserializer)?.unwrap_or_else(empty_message_content))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    None(String),
    Auto(String),
    Required(String),
    Function {
        r#type: String,
        function: FunctionChoice,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    // Anthropic allows `system` as either a plain string or an array of content
    // blocks (Claude Code sends the block form, with `cache_control`). Reuse the
    // same untagged enum as message content so both shapes deserialize.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ImageSource>,
    // `tool_use` blocks (assistant turns): the model's call. `input` is the raw
    // JSON arguments object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    // `tool_result` blocks (user turns): the result fed back. `tool_use_id` ties
    // it to the matching `tool_use`; `content` is a string or nested blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolResultContent>,
}

/// The `content` of an Anthropic `tool_result` block — either a plain string or
/// an array of nested content blocks (text/image). Flattened to text for the
/// OpenAI `tool` message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl ToolResultContent {
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContent::Text(text) => text.clone(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_content_parses() {
        let msg: Message = serde_json::from_str(r#"{"role":"assistant","content":"hi"}"#).unwrap();
        assert_eq!(msg.content.as_text(), "hi");
    }

    #[test]
    fn null_content_parses_as_empty() {
        let msg: Message = serde_json::from_str(r#"{"role":"assistant","content":null}"#).unwrap();
        assert_eq!(msg.content.as_text(), "");
    }

    #[test]
    fn missing_content_parses_as_empty() {
        let msg: Message = serde_json::from_str(r#"{"role":"assistant"}"#).unwrap();
        assert_eq!(msg.content.as_text(), "");
    }

    #[test]
    fn reasoning_model_truncated_response_parses() {
        // glm-5 / qwen truncated mid-CoT: content is null, finish_reason "length",
        // and an unknown `reasoning_content` field is present. Must still parse.
        let body = r#"{
            "id":"x","object":"chat.completion","created":0,"model":"glm-5",
            "choices":[{"index":0,
              "message":{"role":"assistant","content":null,
                         "reasoning_content":"1. analyze the request..."},
              "finish_reason":"length"}],
            "usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}
        }"#;
        let resp: ChatCompletionResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.choices[0].message.content.as_text(), "");
        assert_eq!(resp.choices[0].finish_reason, "length");
        assert_eq!(resp.usage.completion_tokens, 50);
    }
}
