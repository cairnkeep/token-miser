# Semantic routing

Keyword/length routing can't judge difficulty: a short, substantive request like
*"implement binary search"* or *"design a scalable rate limiter"* has no
complexity keyword and few tokens, so the heuristic dumps it on the Free tier.
Semantic routing fixes this by routing on **meaning** — it embeds the request and
sends it to the tier whose exemplars it most resembles.

This is opt-in and **falls back to the heuristic** on any embedding failure, so
enabling it can only help.

## How it works

1. Each tier (Free / Standard / Complex) has a small set of built-in exemplar
   prompts. On first use the router embeds them and stores each tier's
   **centroid** (mean embedding).
2. For each request it embeds the latest user message and routes to the tier
   whose centroid has the highest **cosine similarity**.
3. The hard rules still run first: a request over `tier2_threshold` tokens or
   matching a complexity keyword goes straight to Complex without embedding.

## Enabling

```toml
[semantic_router]
enabled = true
endpoint = "https://api.openai.com/v1"   # OpenAI-compatible /embeddings base
model = "text-embedding-3-small"
api_key = "${OPENAI_API_KEY}"            # optional; ${VAR} is expanded
```

Any OpenAI-compatible embeddings endpoint works — a hosted API, or a local /
private-cluster embedding model.

## Cost and latency

Routing now makes an embedding call per request (embeddings are cheap and fast,
but non-zero). The exemplar centroids are embedded **once** on the first request
and cached — so the first request after startup pays for ~17 exemplar embeddings.
Warm the router with a throwaway request after boot if first-request latency
matters.

Pairs naturally with [escalation](ESCALATION.md): semantic routing reduces *how
often* you misroute; escalation makes the remaining misroutes recover safely.

## Limitations

- **Quality is only as good as the embedding model.** A weak embedder routes
  poorly. Use a real embedding model in production.
- **Centroids are cached for the process lifetime.** If the embeddings endpoint
  is down at the first request, the router caches the failure and stays on the
  heuristic until restarted.
- **Exemplars are built-in**, tuned for coding-assistant traffic. They are not
  yet configurable; adjust them in `src/semantic.rs` for a different domain.
- Routing replaces the heuristic for the middle band; it does not currently
  respect the `tier1_threshold` floor, so a large-context request could still be
  routed to Free if its content reads as trivial. Tune exemplars accordingly.
