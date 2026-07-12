//! Turns the explorer's `<final_answer>` into capped, line-numbered evidence.
//!
//! The whole value of the stage is returning LITTLE, targeted context, so
//! expansion is bounded twice: a global line cap and a global token cap (tiktoken
//! cl100k, the same family the router counts with). Citations are filled in order
//! until either cap is reached; the snippet that crosses the token cap is
//! truncated rather than dropped whole.

use super::tools::Sandbox;
use super::{Citation, Evidence, ExploreStats, Snippet};

/// Parses citations out of a model final answer. Reads only the `<final_answer>`
/// block when present (else the whole text, best-effort for turn-cap finishes).
/// Matches the FastContext citation format `path:START-END (optional note)`:
/// splits on the FIRST colon (notes may contain colons), parses only the leading
/// `START[-END]` (ignoring any trailing note), tolerates `path:LINE`, leading
/// `/` or `./`, `- `/`*` bullets, and backticks.
pub fn parse_citations(text: &str) -> Vec<Citation> {
    let section = extract_section(text);
    let mut out: Vec<Citation> = Vec::new();

    for raw in section.lines() {
        let line = raw
            .trim()
            .trim_start_matches(['-', '*', ' '])
            .trim()
            .trim_matches('`')
            .trim();
        if line.is_empty() || line.starts_with('<') {
            continue;
        }
        // First colon separates the path from the range; the model emits notes
        // after the range that may themselves contain colons.
        let Some((path, range)) = line.split_once(':') else {
            continue;
        };
        let Some((start, end)) = parse_range(range) else {
            continue;
        };
        // The explorer may cite repo-relative paths with a leading `/` or `./`.
        let path = path.trim().trim_start_matches("./").trim_start_matches('/');
        if path.is_empty() {
            continue;
        }
        let (start, end) = (start.max(1), end.max(1));
        let (start, end) = if end < start {
            (start, start)
        } else {
            (start, end)
        };
        let citation = Citation {
            path: path.to_string(),
            start_line: start,
            end_line: end,
        };
        if !out.contains(&citation) {
            out.push(citation);
        }
    }
    out
}

/// Returns the text inside the first `<final_answer>...</final_answer>` block, or
/// the whole input when no block is present.
fn extract_section(text: &str) -> &str {
    const OPEN: &str = "<final_answer>";
    const CLOSE: &str = "</final_answer>";
    if let Some(start) = text.find(OPEN) {
        let after = &text[start + OPEN.len()..];
        return match after.find(CLOSE) {
            Some(end) => &after[..end],
            None => after,
        };
    }
    text
}

/// Parses a leading `START-END` (or single `LINE`) into an inclusive range,
/// ignoring any trailing text such as a ` (note)`. So `249-258 (RoutingConfig)`
/// yields `(249, 258)` and `120 (Default)` yields `(120, 120)`.
fn parse_range(s: &str) -> Option<(usize, usize)> {
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    let start: usize = s[..i].parse().ok()?;
    // Optional `-END`; a dash not followed by digits (or anything else) ends it.
    if i < bytes.len() && bytes[i] == b'-' {
        let j0 = i + 1;
        let mut j = j0;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > j0 {
            let end: usize = s[j0..j].parse().ok()?;
            return Some((start, end));
        }
    }
    Some((start, start))
}

/// Expands citations into real, line-numbered snippets under the global caps and
/// assembles the `Evidence`. A citation whose file can't be read is skipped.
pub fn build_evidence(
    sandbox: &Sandbox,
    final_text: &str,
    max_lines: usize,
    max_tokens: usize,
) -> Evidence {
    let citations = parse_citations(final_text);

    // Fall back to a rough char/4 estimate if the tokenizer fails to load, so
    // expansion is always bounded even without tiktoken.
    let bpe = tiktoken_rs::cl100k_base().ok();
    let count = |s: &str| match &bpe {
        Some(b) => b.encode_with_special_tokens(s).len(),
        None => s.len() / 4 + 1,
    };

    let mut snippets: Vec<Snippet> = Vec::new();
    let mut used_lines = 0usize;
    let mut used_tokens = 0usize;

    for c in &citations {
        if used_lines >= max_lines || used_tokens >= max_tokens {
            break;
        }
        let want = c.end_line - c.start_line + 1;
        let line_budget = (max_lines - used_lines).min(want);
        if line_budget == 0 {
            break;
        }

        let code = match sandbox.read(&c.path, Some(c.start_line), Some(line_budget)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if code.trim_start().starts_with("(no lines") {
            continue;
        }

        let (code, toks, kept) = fit_to_tokens(&code, max_tokens - used_tokens, &count);
        if kept == 0 {
            // Couldn't fit even one line within the remaining token budget.
            break;
        }
        used_lines += kept;
        used_tokens += toks;
        snippets.push(Snippet {
            path: c.path.clone(),
            start_line: c.start_line,
            end_line: c.start_line + kept - 1,
            code,
        });
    }

    Evidence {
        citations,
        expanded_snippets: snippets,
        stats: ExploreStats {
            expanded_lines: used_lines,
            expanded_tokens: used_tokens,
            ..Default::default()
        },
    }
}

/// Fits a line-numbered block within `budget` tokens by dropping trailing lines.
/// Returns (text, tokens, source_lines_kept). The truncation marker is not
/// counted as a source line.
fn fit_to_tokens(
    code: &str,
    budget: usize,
    count: &impl Fn(&str) -> usize,
) -> (String, usize, usize) {
    let total = count(code);
    if total <= budget {
        return (code.to_string(), total, code.lines().count());
    }
    let lines: Vec<&str> = code.lines().collect();
    let mut kept = lines.len();
    while kept > 0 {
        kept -= 1;
        let candidate = format!("{}\n... [truncated: token cap]", lines[..kept].join("\n"));
        let toks = count(&candidate);
        if toks <= budget {
            return (candidate, toks, kept);
        }
    }
    (String::new(), 0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parses_ranges_singles_bullets_and_backticks() {
        let text = "preamble\n<final_answer>\n\
                    src/router.rs:42-88\n\
                    - src/config.rs:120\n\
                    `src/proxy.rs:10-12`\n\
                    not a citation line\n\
                    </final_answer>\ntrailing";
        let cites = parse_citations(text);
        assert_eq!(
            cites,
            vec![
                Citation {
                    path: "src/router.rs".into(),
                    start_line: 42,
                    end_line: 88
                },
                Citation {
                    path: "src/config.rs".into(),
                    start_line: 120,
                    end_line: 120
                },
                Citation {
                    path: "src/proxy.rs".into(),
                    start_line: 10,
                    end_line: 12
                },
            ]
        );
    }

    #[test]
    fn parses_fastcontext_format_with_notes_and_leading_slash() {
        // The real FastContext-1.0 output: path:START-END (note), and the paper's
        // leading-slash variant. Notes (which may contain commas/colons) and the
        // slash must not break parsing.
        let text = "Here is a summary.\n<final_answer>\n\
                    src/config.rs:249-258 (RoutingConfig struct defining thresholds)\n\
                    /src/router.rs:82-124 (classify(): escalates to Tier::Complex)\n\
                    ./src/proxy.rs:7 (Proxy)\n\
                    </final_answer>";
        let cites = parse_citations(text);
        assert_eq!(
            cites,
            vec![
                Citation {
                    path: "src/config.rs".into(),
                    start_line: 249,
                    end_line: 258
                },
                Citation {
                    path: "src/router.rs".into(),
                    start_line: 82,
                    end_line: 124
                },
                Citation {
                    path: "src/proxy.rs".into(),
                    start_line: 7,
                    end_line: 7
                },
            ]
        );
    }

    #[test]
    fn falls_back_to_whole_text_without_tags() {
        let cites = parse_citations("src/main.rs:1-5");
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].path, "src/main.rs");
    }

    #[test]
    fn dedupes_identical_citations() {
        let text = "<final_answer>\nsrc/a.rs:1-2\nsrc/a.rs:1-2\n</final_answer>";
        assert_eq!(parse_citations(text).len(), 1);
    }

    fn fixture(lines: usize) -> (PathBuf, Sandbox) {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("tm-expand-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let body: String = (1..=lines).map(|n| format!("line {n}\n")).collect();
        std::fs::write(root.join("big.rs"), body).unwrap();
        (root.clone(), Sandbox::new(&root).unwrap())
    }

    #[test]
    fn expansion_respects_line_cap() {
        let (root, sb) = fixture(100);
        let text = "<final_answer>\nbig.rs:1-100\n</final_answer>";
        let ev = build_evidence(&sb, text, 10, 100_000);
        assert_eq!(ev.expanded_snippets.len(), 1);
        assert_eq!(ev.stats.expanded_lines, 10);
        // end_line reflects the clamped span, not the requested 100.
        assert_eq!(ev.expanded_snippets[0].end_line, 10);
        assert!(ev.expanded_snippets[0].code.contains("     1\tline 1"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn line_cap_spans_multiple_citations() {
        let (root, sb) = fixture(100);
        let text = "<final_answer>\nbig.rs:1-6\nbig.rs:50-60\n</final_answer>";
        // 6 lines from the first citation, then only 4 left for the second.
        let ev = build_evidence(&sb, text, 10, 100_000);
        assert_eq!(ev.stats.expanded_lines, 10);
        assert_eq!(ev.expanded_snippets.len(), 2);
        assert_eq!(ev.expanded_snippets[1].end_line, 53);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn expansion_respects_token_cap_with_truncation() {
        let (root, sb) = fixture(100);
        let text = "<final_answer>\nbig.rs:1-100\n</final_answer>";
        let ev = build_evidence(&sb, text, 100_000, 20);
        assert!(
            ev.stats.expanded_tokens <= 20,
            "tokens: {}",
            ev.stats.expanded_tokens
        );
        assert!(
            ev.expanded_snippets[0]
                .code
                .contains("truncated: token cap")
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
