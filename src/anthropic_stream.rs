use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use std::collections::VecDeque;

/// Translates an upstream OpenAI `chat.completion.chunk` SSE stream into the
/// Anthropic Messages streaming event format.
///
/// OpenAI emits `data: {chunk}` frames terminated by `data: [DONE]`. Anthropic
/// clients expect a typed event sequence: `message_start`, `content_block_start`,
/// repeated `content_block_delta`, `content_block_stop`, `message_delta`,
/// `message_stop`. This adapter buffers across TCP chunk boundaries, parses each
/// OpenAI delta, and re-emits the equivalent Anthropic events. Text content
/// becomes a `text` block; each streamed tool call becomes a `tool_use` block
/// with `input_json_delta` fragments.
pub fn translate(
    upstream: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>,
    input_tokens: u64,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    let state = State {
        upstream,
        buf: Vec::new(),
        out: VecDeque::new(),
        started: false,
        stopped: false,
        finished: false,
        id: String::new(),
        model: String::new(),
        stop_reason: "end_turn".to_string(),
        next_index: 0,
        open: None,
        tool_blocks: Vec::new(),
        input_tokens,
        output_tokens: 0,
    };

    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(frame) = state.out.pop_front() {
                return Some((Ok(frame), state));
            }
            if state.finished {
                return None;
            }

            match state.upstream.next().await {
                Some(Ok(bytes)) => {
                    if state.ingest(&bytes) {
                        state.finalize();
                        state.finished = true;
                    }
                }
                Some(Err(_)) | None => {
                    state.finalize();
                    state.finished = true;
                }
            }
        }
    })
}

struct State {
    upstream: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>,
    buf: Vec<u8>,
    out: VecDeque<Bytes>,
    started: bool,
    stopped: bool,
    finished: bool,
    id: String,
    model: String,
    stop_reason: String,
    next_index: usize,
    open: Option<OpenBlock>,
    tool_blocks: Vec<(u64, usize)>,
    input_tokens: u64,
    output_tokens: u64,
}

/// The single content block currently open in the Anthropic stream (only one
/// block may be open at a time), tagged by kind so it can be closed correctly.
#[derive(Clone, Copy)]
enum OpenBlock {
    Text(usize),
    Tool(usize),
}

impl OpenBlock {
    fn index(self) -> usize {
        match self {
            OpenBlock::Text(i) | OpenBlock::Tool(i) => i,
        }
    }
}

impl State {
    /// Appends bytes, processes every complete line, and returns `true` once the
    /// upstream `[DONE]` sentinel has been seen.
    fn ingest(&mut self, bytes: &[u8]) -> bool {
        self.buf.extend_from_slice(bytes);
        let mut done = false;

        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line);
            let line = text.trim_end_matches(['\r', '\n']);

            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = payload.trim();

            if payload == "[DONE]" {
                done = true;
            } else if !payload.is_empty()
                && let Ok(chunk) = serde_json::from_str::<Value>(payload)
            {
                self.handle_chunk(&chunk);
            }
        }

        done
    }

    fn handle_chunk(&mut self, chunk: &Value) {
        if !self.started {
            self.id = chunk["id"].as_str().unwrap_or("msg_stream").to_string();
            self.model = chunk["model"].as_str().unwrap_or("").to_string();
            self.emit_start();
        }

        let delta = &chunk["choices"][0]["delta"];

        if let Some(text) = delta["content"].as_str()
            && !text.is_empty()
        {
            let index = self.ensure_text_block();
            self.out.push_back(frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                }),
            ));
        }

        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            for call in tool_calls {
                self.handle_tool_call(call);
            }
        }

        if let Some(reason) = chunk["choices"][0]["finish_reason"].as_str() {
            self.stop_reason = map_stop_reason(reason);
            self.close_open_block();
        }

        // The usage chunk (include_usage) arrives after finish_reason, so the
        // terminal message_delta is deferred to finalize() to pick it up.
        if let Some(tokens) = chunk["usage"]["completion_tokens"].as_u64() {
            self.output_tokens = tokens;
        }
    }

    fn handle_tool_call(&mut self, call: &Value) {
        let oai_index = call["index"].as_u64().unwrap_or(0);

        let block_index = match self.tool_blocks.iter().find(|(o, _)| *o == oai_index) {
            Some((_, a)) => *a,
            None => {
                let id = call["id"].as_str().unwrap_or("").to_string();
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let index = self.open_tool_block(&id, &name);
                self.tool_blocks.push((oai_index, index));
                index
            }
        };

        if let Some(args) = call["function"]["arguments"].as_str()
            && !args.is_empty()
            && matches!(self.open, Some(OpenBlock::Tool(i)) if i == block_index)
        {
            self.out.push_back(frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                }),
            ));
        }
    }

    fn emit_start(&mut self) {
        self.out.push_back(frame(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            }),
        ));
        self.started = true;
    }

    /// Returns the index of the open text block, opening one (and closing any
    /// other open block) if needed.
    fn ensure_text_block(&mut self) -> usize {
        if let Some(OpenBlock::Text(i)) = self.open {
            return i;
        }
        self.close_open_block();
        let index = self.next_index;
        self.next_index += 1;
        self.out.push_back(frame(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }),
        ));
        self.open = Some(OpenBlock::Text(index));
        index
    }

    fn open_tool_block(&mut self, id: &str, name: &str) -> usize {
        self.close_open_block();
        let index = self.next_index;
        self.next_index += 1;
        self.out.push_back(frame(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}},
            }),
        ));
        self.open = Some(OpenBlock::Tool(index));
        index
    }

    fn close_open_block(&mut self) {
        if let Some(block) = self.open.take() {
            self.out.push_back(frame(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": block.index()}),
            ));
        }
    }

    /// Emits the terminal events once the upstream ends ([DONE] or EOF). Also
    /// covers the empty-completion case (no content chunks) and a missing
    /// terminal `finish_reason`.
    fn finalize(&mut self) {
        if self.stopped {
            return;
        }
        if !self.started {
            self.emit_start();
        }
        self.close_open_block();
        self.out.push_back(frame(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": Value::Null},
                "usage": {"output_tokens": self.output_tokens},
            }),
        ));
        self.out
            .push_back(frame("message_stop", json!({"type": "message_stop"})));
        self.stopped = true;
    }
}

fn frame(event: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn map_stop_reason(openai: &str) -> String {
    match openai {
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        _ => "end_turn",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(
        chunks: Vec<&'static [u8]>,
    ) -> Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin> {
        let items: Vec<Result<Bytes, reqwest::Error>> = chunks
            .into_iter()
            .map(|c| Ok(Bytes::from_static(c)))
            .collect();
        Box::new(futures::stream::iter(items))
    }

    async fn collect(
        stream: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin>,
    ) -> String {
        let mut out = String::new();
        let mut s = Box::pin(translate(stream, 0));
        while let Some(chunk) = s.next().await {
            out.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn test_translates_text_stream_in_order() {
        let stream = boxed(vec![
            b"data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            b"data: [DONE]\n\n",
        ]);

        let out = collect(stream).await;

        let start = out.find("event: message_start").expect("message_start");
        let block_start = out
            .find("event: content_block_start")
            .expect("content_block_start");
        let delta = out.find("\"text\":\"Hello\"").expect("first delta");
        let stop = out.find("event: message_stop").expect("message_stop");

        assert!(start < block_start && block_start < delta && delta < stop);
        assert!(out.contains("\"text\":\" world\""));
        assert!(out.contains("\"stop_reason\":\"end_turn\""));
        assert!(out.contains("\"model\":\"gpt-4o-mini\""));
        // Anthropic streams must not carry the OpenAI sentinel.
        assert!(!out.contains("[DONE]"));
        // No double-prefixing of the upstream frames.
        assert!(!out.contains("data: data:"));
    }

    #[tokio::test]
    async fn test_reassembles_frame_split_across_chunks() {
        // A single OpenAI frame arrives split mid-JSON across two TCP chunks.
        let stream = boxed(vec![
            b"data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"delta\":{\"cont",
            b"ent\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        ]);

        let out = collect(stream).await;

        assert!(out.contains("\"text\":\"Hi\""));
        assert!(out.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn test_translates_tool_call_stream() {
        let stream = boxed(vec![
            b"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"NYC\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            b"data: [DONE]\n\n",
        ]);

        let out = collect(stream).await;

        let block_start = out.find("\"type\":\"tool_use\"").expect("tool_use block");
        let name = out.find("\"name\":\"get_weather\"").expect("tool name");
        let first_arg = out.find("input_json_delta").expect("input_json_delta");
        let stop = out.find("event: message_stop").expect("message_stop");

        assert!(block_start < first_arg && first_arg < stop);
        assert!(out.contains("\"id\":\"call_1\""));
        assert!(name < stop);
        assert!(out.contains("\"stop_reason\":\"tool_use\""));
        assert!(out.contains("event: content_block_stop"));
        // No spurious text block for a tool-only response.
        assert!(!out.contains("text_delta"));
    }

    #[tokio::test]
    async fn test_reports_usage_output_tokens() {
        // The usage chunk arrives after the finish_reason chunk.
        let stream = boxed(vec![
            b"data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
            b"data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            b"data: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
            b"data: [DONE]\n\n",
        ]);

        let out = collect(stream).await;

        assert!(out.contains("\"output_tokens\":7"));
        // The terminal events are emitted exactly once.
        assert_eq!(out.matches("event: message_stop").count(), 1);
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
    }
}
