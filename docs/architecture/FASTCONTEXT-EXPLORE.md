# FastContext repository exploration (upstream stage)

The router guesses task difficulty from the prompt it receives. When a client
sends a short, vague task ("fix the routing bug") the router has little to go on;
when it sends a giant pasted-in context dump, the prompt is bloated and noisy.
Both push requests toward the premium tier (Opus) more often than necessary.

The FastContext stage runs **before** the router. Given the task, a small remote
model drives an agentic READ/GLOB/GREP loop against the repository on disk and
returns a **small, capped bundle of evidence** — a handful of file ranges
expanded to real code. That clean context is injected into the request, so the
existing classifier sees a short, targeted prompt. The economic goal is to
**lower the Opus-escalation rate**.

```
task ──▶ explore_repo(query) ──▶ [FastContext explorer, REMOTE inference]
              │  model emits READ/GLOB/GREP tool calls;
              │  THIS process executes them LOCALLY against repo_root,
              │  feeds observations back, until <final_answer>
              ▼
        clean, capped evidence (file paths + line ranges → expanded code)
              ▼
        existing classifier → cluster pool | escalate to Opus
```

## Invariants

- **FastContext is never a routing target.** It runs strictly upstream and
  produces the prompt the router classifies; it is not a tier or a provider.
- **Inference is remote and backend-agnostic.** Any OpenAI-compatible
  `chat/completions` endpoint with `tools` works — cluster vLLM, a local
  `llama-server`, anything. The code does not assume a specific backend.
- **Tool execution is local and sandboxed.** READ/GLOB/GREP run in-process
  (pure Rust — `ignore`/`globset` for globbing, the ripgrep `grep` crates for
  search), confined to `repo_root`. Absolute paths and `..` escapes are rejected;
  every read is size- and line-bounded.
- **No index.** Exploration is agentic and on-demand. There is no embeddings
  store, no persistent index, and no re-indexing on commit — that is the point
  versus a RAG layer.

## Enabling

```toml
[fastcontext]
enabled = false                              # master switch
endpoint_url = "http://localhost:8081/v1"    # any OpenAI-compatible endpoint
model = "fastcontext-4b"
# api_key = "${FASTCONTEXT_API_KEY}"          # only if the endpoint needs one

[explore]
repo_root = "."                              # sandbox root for the local tools
max_turns = 16                               # agentic-loop cap; best-effort on cap
max_expanded_lines = 200                     # keep tight — little, targeted context
max_expanded_tokens = 4000
```

When `enabled = false` (the default), the stage is skipped and behavior is
byte-identical to before.

## Running the model

The reference model is Microsoft's **FastContext-1.0-4B** (a Qwen3-4B explorer,
SFT→RL). The **RL** checkpoint is the deployment target; the community GGUF
`mitkox/FastContext-1.0-4B-RL-Q8_0-GGUF` (4.3 GB) runs well under llama.cpp.
Serve it with any OpenAI-compatible runtime — the proxy is backend-agnostic.
Validated recipe (llama.cpp server, NVIDIA GPU via a container):

```bash
# Download the GGUF (single ~4.3 GB file)
curl -L -o fastcontext-1.0-4b-rl-q8_0.gguf \
  https://huggingface.co/mitkox/FastContext-1.0-4B-RL-Q8_0-GGUF/resolve/main/fastcontext-1.0-4b-rl-q8_0.gguf

# llama-server with the OpenAI-compatible API + tool-calling (--jinja is required)
llama-server -m fastcontext-1.0-4b-rl-q8_0.gguf --alias fastcontext-4b-rl \
  -ngl 99 -c 24576 --jinja --host 0.0.0.0 --port 8081 \
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
```

Then point the proxy at it: `fastcontext.endpoint_url = "http://<host>:8081/v1"`,
`fastcontext.model = "fastcontext-4b-rl"`.

Notes from bring-up:
- **`--jinja` is mandatory** — without the model's chat template, `tool_calls`
  are not parsed and the loop never runs.
- **Context size matters.** The explorer accumulates tool observations; a 4B can
  over-explore on broad queries and overflow a small window. Give it as much
  context as VRAM allows (`-c 24576` worked alongside a 16 GB-GPU co-tenant via
  q4_0 KV cache + flash attention). On overflow the loop degrades gracefully
  (best-effort, no hard failure) — but yield improves markedly with headroom.
- **Tool names** are advertised lowercase (`read`/`glob`/`grep`); the model
  adapts to the provided schema. The trained citation format is
  `path:START-END (optional note)`, which the parser handles (including a leading
  `/`).
- The `explore` CLI prints **only JSON on stdout** (logs go to stderr), so it
  pipes cleanly: `token_miser explore --query … | jq .citations`.

## When the stage runs

Only on a **fresh task** — a request with no prior `assistant` or `tool` turns.
Exploring mid agentic-loop would re-gather context on every tool round-trip, so
the stage stays out of the way once a conversation is underway. The query is the
latest user turn's text. The evidence is injected as a `system` message placed
after any leading system prompt and before the first user turn.

## The loop

Each turn sends the running message history (with the three tool schemas). If the
model returns `tool_calls`, this process executes them locally — concurrently
when the model batches several in one turn — and appends each result as a
`role:"tool"` message. The loop ends when the model emits a `<final_answer>`
block (or stops calling tools), and is hard-capped at `max_turns`; on the cap it
returns the best evidence gathered so far. (Some serving stacks serialize
parallel tool calls, so the default cap leaves headroom.)

The `<final_answer>` block lists citations, one per line, as
`relative/path:START-END (optional note)`. Each is expanded into line-numbered
code under two global caps — `max_expanded_lines` and `max_expanded_tokens` (tiktoken cl100k).
The snippet that crosses the token cap is truncated rather than dropped. Tight
defaults are deliberate: uncapped expansion would reintroduce the bloat the stage
exists to remove.

## Standalone CLI

The explorer is also exposed as a subcommand, so a coding agent or an implementer
step can gather context without going through the proxy:

```bash
token_miser explore --query "where is request routing decided?" --repo-root .
# prints Evidence { citations, expanded_snippets, stats } as JSON
```

`--repo-root` defaults to `explore.repo_root` from config. The CLI needs
`fastcontext.endpoint_url` set, but does not require `fastcontext.enabled`.

## Observability

Injecting clean context upstream shifts the input-size distribution, so the
existing complexity thresholds will mis-fire until they are **re-tuned by hand**
against this data. The stage does **not** auto-tune anything — it only makes the
before/after measurable. Each telemetry record carries:

- `premium_escalation` — whether the served tier was the premium tier (Complex /
  Opus). The **Opus-escalation rate** is the share of records with this `true`.
- `pre_explore_input_tokens` / `post_explore_input_tokens` — the per-prompt token
  volume before and after evidence injection.
- `explore_ran`, `explore_turns`, `explore_citations`, `explore_expanded_tokens`
  — stage activity.

```bash
# Opus-escalation rate, split by whether the explore stage ran.
jq -rs '
  group_by(.explore_ran)[] |
  { explore_ran: .[0].explore_ran,
    n: length,
    opus_rate: ((map(select(.premium_escalation)) | length) / length) }' \
  telemetry.jsonl

# Per-prompt token shift from injection.
jq -r 'select(.explore_ran) |
  "\(.pre_explore_input_tokens) -> \(.post_explore_input_tokens)"' telemetry.jsonl
```

Compare the `opus_rate` for `explore_ran=false` (baseline) against `true`
(with evidence), then re-tune `[routing]` thresholds against the observed
`post_explore_input_tokens` distribution.
