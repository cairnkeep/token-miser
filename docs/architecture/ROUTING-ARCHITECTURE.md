# Routing Heuristics Architecture

## Decision Flow Diagram

```mermaid
graph TD
    A[Incoming Request] --> B{Token Count Check}
    B -->|< 2K tokens| C[Tier 1: Free/Local]
    B -->|2K-32K tokens| D[Complexity Analysis]
    B -->|> 32K tokens| E[Tier 3: Premium]
    
    D --> F{Keyword Detection}
    F -->|architect, refactor, system design| E
    F -->|debug, optimize| G{Tier 2: Standard}
    F -->|simple, format| C
    
    D --> H{Intent Classification}
    H -->|Multi-file, agentic| E
    H -->|Single-file, standard| G
    H -->|Boilerplate, autocomplete| C
    
    E --> I[Claude Opus 4.6<br/>MAI-Thinking-1]
    G --> J[Claude Sonnet 4.6<br/>GPT-4o-mini]
    C --> K[Local: Llama 3<br/>MAI-Code-1 via Ollama]
    
    style E fill:#ff6b6b
    style G fill:#feca57
    style C fill:#48dbfb
```

## Parallel Evaluation Strategy

```mermaid
graph LR
    A[Request Payload] --> B[Token Estimator]
    A --> C[Keyword Scanner]
    A --> D[Intent Classifier]
    
    B --> E[Decision Arbiter]
    C --> E
    D --> E
    
    E --> F[Final Route Selection]
    
    style B fill:#a8e6cf
    style C fill:#a8e6cf
    style D fill:#dfe6e9
    style E fill:#ffeaa7
```

## Payload Inspection Methods

### 1. Token Counting (Fast: ~5-10ms)
- **Method**: Use `tiktoken-rs` cl100k_base tokenizer
- **Threshold**: 32K tokens triggers Tier 3
- **Cost**: Near-zero latency overhead

### 2. Keyword Matching (Fast: <2ms)
- **Tier 3 Keywords**: `architect`, `system design`, `multi-file`, `migrate`, `redesign`
- **Tier 2 Keywords**: `refactor`, `implement`, `debug`, `optimize`, `explain`
- **Tier 1 Keywords**: `format`, `simple`, `quick`, `autocomplete`

### 3. Fast LLM Classifier (Fast: ~50-200ms)
- **Model**: Local Llama 3.2 1B or quantized 0.5B model
- **Purpose**: Classify intent before routing main request
- **Categories**: `agentic`, `standard`, `trivial`

### 4. Context Window Analysis
- **Method**: Count file references, code blocks, context injection size
- **Threshold**: >10 files or >20K injected context → Tier 3

## Intent Classification Categories

```mermaid
graph TD
    A[Intent Categories] --> B[Agentic/Multi-file]
    A --> C[Standard/Single-file]
    A --> D[Trivial/Autocomplete]
    
    B --> B1[Architecture Design]
    B --> B2[System Refactoring]
    B --> B3[Multi-module Migration]
    
    C --> C1[Function Generation]
    C --> C2[Test Writing]
    C --> C3[Bug Fixing]
    C --> C4[Code Explanation]
    
    D --> D1[Syntax Correction]
    D --> D2[Formatting]
    D --> D3[Simple Completions]
    D --> D4[Boilerplate Generation]
    
    style B fill:#ff6b6b
    style C fill:#feca57
    style D fill:#48dbfb
```

## RAG Integration Context

```mermaid
graph TB
    A[User Request] --> B[RAG Context Injection]
    B --> C{Context Size Check}
    C -->|Small context| D[Original Request]
    C -->|Large context| E[Augmented Request]
    
    D --> F[Standard Routing]
    E --> G{Context-Aware Routing}
    
    G --> H[Tier 2 Boost]
    G --> I[Tier 3 for Enterprise Knowledge]
    
    F --> J[Execute Request]
    H --> J
    I --> J
    
    style B fill:#dfe6e9
    style E fill:#ffeaa7
    style G fill:#a29bfe
```

## Performance Budgets

| Inspection Method | Latency Budget | CPU Impact | Accuracy |
|-------------------|----------------|------------|----------|
| Token Counting | 5-10ms | Low | High |
| Keyword Matching | 1-2ms | Very Low | Medium |
| Intent Classifier | 50-200ms | Medium | Very High |
| Context Analysis | 2-5ms | Low | High |

**Target Total**: <50ms overhead for routing decision (without intent classifier)
**With Intent Classifier**: <250ms (only invoked on ambiguous cases)

## Implementation Priority

1. **Phase 1** (Current): Token counting + keyword matching
2. **Phase 2**: Add intent classifier for ambiguous cases
3. **Phase 3**: Integrate RAG context size analysis
4. **Phase 4**: Add adaptive learning based on routing outcomes
