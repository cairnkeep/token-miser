//! Pure-Rust, read-only repository tools executed LOCALLY for the explorer loop.
//!
//! Three tools — READ / GLOB / GREP — all sandboxed to a single `repo_root`. No
//! shelling out: GLOB walks via the `ignore` crate (gitignore-aware) and matches
//! with `globset`; GREP uses the ripgrep `grep` crates. Every read is size- and
//! line-bounded so a hostile or confused model can't pull the whole tree into the
//! prompt. Path access is hard-sandboxed: absolute paths and `..` escapes outside
//! the canonicalized root are rejected.

use super::ExploreError;
use globset::Glob;
use grep::regex::RegexMatcher;
use grep::searcher::Searcher;
use grep::searcher::sinks::UTF8;
use ignore::WalkBuilder;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Refuse to read files larger than this (avoids loading a giant blob to slice).
const HARD_FILE_CEILING: u64 = 4 * 1024 * 1024;
/// Output byte cap for a single READ result. Kept small: the explorer feeds a
/// modest-context model, so a single observation must not flood the window.
const MAX_READ_BYTES: usize = 64 * 1024;
/// Default (and maximum) number of lines a single READ returns when the caller
/// omits `limit`. Small on purpose — the model pages with `offset` when it needs
/// more, instead of pulling a whole file into the exploration context.
const MAX_READ_LINES: usize = 400;
/// Cap on paths returned by one GLOB call.
const MAX_GLOB_RESULTS: usize = 150;
/// Cap on matches returned by one GREP call.
const MAX_GREP_MATCHES: usize = 100;
/// Per-line character cap for GREP output (keeps a single long line from bloating).
const MAX_GREP_LINE_LEN: usize = 300;

/// A repo root the tools are confined to. Constructed once per exploration.
pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    /// Canonicalizes `root` (resolving symlinks) so the containment check is exact.
    pub fn new(root: &Path) -> Result<Self, ExploreError> {
        let root = root
            .canonicalize()
            .map_err(|e| ExploreError::Sandbox(format!("repo_root {root:?}: {e}")))?;
        if !root.is_dir() {
            return Err(ExploreError::Sandbox(format!(
                "repo_root is not a directory: {root:?}"
            )));
        }
        Ok(Self { root })
    }

    /// Resolves a caller-supplied relative path to a real file path inside the
    /// root, rejecting absolute paths and any `..` traversal that escapes it.
    fn resolve(&self, rel: &str) -> Result<PathBuf, ExploreError> {
        self.guard_relative(rel)?;
        let candidate = self.root.join(rel);
        let canonical = candidate
            .canonicalize()
            .map_err(|e| ExploreError::Sandbox(format!("{rel}: {e}")))?;
        if !canonical.starts_with(&self.root) {
            return Err(ExploreError::Sandbox(format!(
                "path escapes repo_root: {rel}"
            )));
        }
        Ok(canonical)
    }

    /// Resolves a caller-supplied relative directory (defaulting to the root).
    fn resolve_dir(&self, rel: Option<&str>) -> Result<PathBuf, ExploreError> {
        match rel {
            None | Some("") | Some(".") => Ok(self.root.clone()),
            Some(rel) => {
                let dir = self.resolve(rel)?;
                if !dir.is_dir() {
                    return Err(ExploreError::Sandbox(format!("not a directory: {rel}")));
                }
                Ok(dir)
            }
        }
    }

    /// Rejects absolute paths and lexical parent-dir escapes before any FS touch,
    /// so a path that would leave the root never reaches `canonicalize`.
    fn guard_relative(&self, rel: &str) -> Result<(), ExploreError> {
        let p = Path::new(rel);
        if p.is_absolute() {
            return Err(ExploreError::Sandbox(format!(
                "absolute path rejected: {rel}"
            )));
        }
        let mut depth: i32 = 0;
        for comp in p.components() {
            use std::path::Component::*;
            match comp {
                ParentDir => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(ExploreError::Sandbox(format!(
                            "path escapes repo_root: {rel}"
                        )));
                    }
                }
                Prefix(_) | RootDir => {
                    return Err(ExploreError::Sandbox(format!(
                        "absolute path rejected: {rel}"
                    )));
                }
                CurDir => {}
                Normal(_) => depth += 1,
            }
        }
        Ok(())
    }

    /// READ: returns line-numbered file contents. `offset` is the 1-based first
    /// line, `limit` the number of lines (both bounded). The model relies on the
    /// line numbers to cite ranges, so they are always emitted.
    pub fn read(
        &self,
        rel: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<String, ExploreError> {
        let path = self.resolve(rel)?;
        let meta = std::fs::metadata(&path)?;
        if !meta.is_file() {
            return Err(ExploreError::Sandbox(format!("not a file: {rel}")));
        }
        if meta.len() > HARD_FILE_CEILING {
            return Err(ExploreError::Sandbox(format!(
                "file too large to read ({} bytes > {} cap): {rel}",
                meta.len(),
                HARD_FILE_CEILING
            )));
        }

        let bytes = std::fs::read(&path)?;
        let text = String::from_utf8_lossy(&bytes);

        let start = offset.unwrap_or(1).max(1);
        let count = limit.unwrap_or(MAX_READ_LINES).min(MAX_READ_LINES);

        let mut out = String::new();
        let mut emitted = 0usize;
        for (idx, line) in text.lines().enumerate() {
            let lineno = idx + 1;
            if lineno < start {
                continue;
            }
            if emitted >= count {
                break;
            }
            let entry = format!("{lineno:>6}\t{line}\n");
            if out.len() + entry.len() > MAX_READ_BYTES {
                out.push_str("... [truncated: output byte cap reached]\n");
                break;
            }
            out.push_str(&entry);
            emitted += 1;
        }
        if out.is_empty() {
            out.push_str("(no lines in requested range)\n");
        }
        Ok(out)
    }

    /// GLOB: gitignore-aware path discovery under `base` (default root). Patterns
    /// match against the path RELATIVE to root (e.g. `src/**/*.rs`).
    pub fn glob(&self, pattern: &str, base: Option<&str>) -> Result<Vec<String>, ExploreError> {
        let base_dir = self.resolve_dir(base)?;
        let matcher = Glob::new(pattern)
            .map_err(|e| ExploreError::Sandbox(format!("bad glob {pattern:?}: {e}")))?
            .compile_matcher();

        let mut results = Vec::new();
        for dent in WalkBuilder::new(&base_dir).build() {
            let Ok(dent) = dent else { continue };
            if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let rel = dent.path().strip_prefix(&self.root).unwrap_or(dent.path());
            if matcher.is_match(rel) {
                results.push(rel.to_string_lossy().to_string());
                if results.len() >= MAX_GLOB_RESULTS {
                    break;
                }
            }
        }
        results.sort();
        Ok(results)
    }

    /// GREP: regex search via the ripgrep crates. Searches a single `path` when
    /// given, otherwise walks the root (gitignore-aware) optionally filtered by a
    /// `glob`. Returns `path:line:content`, line-truncated and globally capped.
    pub fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob: Option<&str>,
    ) -> Result<Vec<String>, ExploreError> {
        let matcher = RegexMatcher::new(pattern)
            .map_err(|e| ExploreError::Sandbox(format!("bad regex {pattern:?}: {e}")))?;

        let glob_matcher = match glob {
            Some(g) => Some(
                Glob::new(g)
                    .map_err(|e| ExploreError::Sandbox(format!("bad glob {g:?}: {e}")))?
                    .compile_matcher(),
            ),
            None => None,
        };

        let files: Vec<PathBuf> = if let Some(p) = path {
            vec![self.resolve(p)?]
        } else {
            WalkBuilder::new(&self.root)
                .build()
                .filter_map(|d| d.ok())
                .filter(|d| d.file_type().map(|t| t.is_file()).unwrap_or(false))
                .map(|d| d.path().to_path_buf())
                .filter(|p| match &glob_matcher {
                    Some(m) => {
                        let rel = p.strip_prefix(&self.root).unwrap_or(p);
                        m.is_match(rel)
                    }
                    None => true,
                })
                .collect()
        };

        let mut results: Vec<String> = Vec::new();
        let mut capped = false;
        for file in files {
            let rel = file
                .strip_prefix(&self.root)
                .unwrap_or(&file)
                .to_string_lossy()
                .to_string();
            let mut searcher = Searcher::new();
            // search_path errors on binary/non-UTF8 files; skip them silently.
            let _ = searcher.search_path(
                &matcher,
                &file,
                UTF8(|lnum, line| {
                    let trimmed = line.trim_end_matches(['\r', '\n']);
                    let truncated: String = trimmed.chars().take(MAX_GREP_LINE_LEN).collect();
                    results.push(format!("{rel}:{lnum}:{truncated}"));
                    Ok(results.len() < MAX_GREP_MATCHES)
                }),
            );
            if results.len() >= MAX_GREP_MATCHES {
                capped = true;
                break;
            }
        }
        if capped {
            results.push("... [truncated: match cap reached]".to_string());
        }
        Ok(results)
    }

    /// Dispatches one model tool call (by name + JSON args) to the matching tool
    /// and renders a model-readable observation. Never panics: a sandbox/argument
    /// error becomes an `ERROR: ...` string so the model can recover, rather than
    /// aborting the loop.
    pub fn run_tool(&self, name: &str, args: &Value) -> String {
        let result = match name.to_ascii_lowercase().as_str() {
            "read" => {
                let Some(path) = args.get("path").and_then(|v| v.as_str()) else {
                    return "ERROR: READ requires a `path` argument".to_string();
                };
                let offset = args
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let limit = args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                self.read(path, offset, limit)
            }
            "glob" => {
                let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) else {
                    return "ERROR: GLOB requires a `pattern` argument".to_string();
                };
                let base = args.get("base").and_then(|v| v.as_str());
                self.glob(pattern, base)
                    .map(|v| join_or_empty(v, "(no matching paths)"))
            }
            "grep" => {
                let Some(regex) = args.get("regex").and_then(|v| v.as_str()) else {
                    return "ERROR: GREP requires a `regex` argument".to_string();
                };
                let path = args.get("path").and_then(|v| v.as_str());
                let glob = args.get("glob").and_then(|v| v.as_str());
                self.grep(regex, path, glob)
                    .map(|v| join_or_empty(v, "(no matches)"))
            }
            other => return format!("ERROR: unknown tool {other:?}"),
        };
        match result {
            Ok(s) => s,
            Err(e) => format!("ERROR: {e}"),
        }
    }
}

fn join_or_empty(items: Vec<String>, empty: &str) -> String {
    if items.is_empty() {
        empty.to_string()
    } else {
        items.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Builds a unique temp fixture repo and returns (root, Sandbox).
    fn fixture() -> (PathBuf, Sandbox) {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("tm-explore-{}-{seq}", std::process::id()));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        )
        .unwrap();
        std::fs::write(root.join("README.md"), "# Fixture\nhello world\n").unwrap();
        let sandbox = Sandbox::new(&root).unwrap();
        (root, sandbox)
    }

    #[test]
    fn read_emits_line_numbers() {
        let (root, sb) = fixture();
        let out = sb.read("src/main.rs", None, None).unwrap();
        assert!(out.contains("     1\tfn main() {"), "got: {out}");
        assert!(out.contains("     2\t    println!(\"hello\");"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn read_respects_offset_and_limit() {
        let (root, sb) = fixture();
        let out = sb.read("src/main.rs", Some(2), Some(1)).unwrap();
        assert!(out.contains("     2\t    println!(\"hello\");"));
        assert!(!out.contains("     1\t"));
        assert!(!out.contains("     3\t"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sandbox_rejects_parent_escape() {
        let (root, sb) = fixture();
        let err = sb.read("../../etc/passwd", None, None).unwrap_err();
        assert!(matches!(err, ExploreError::Sandbox(_)), "got: {err:?}");
        // And via the dispatcher it becomes a recoverable ERROR string.
        let obs = sb.run_tool("read", &serde_json::json!({"path": "../secret"}));
        assert!(obs.starts_with("ERROR:"), "got: {obs}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sandbox_rejects_absolute_path() {
        let (root, sb) = fixture();
        let err = sb.read("/etc/passwd", None, None).unwrap_err();
        assert!(matches!(err, ExploreError::Sandbox(_)), "got: {err:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn glob_finds_rust_files() {
        let (root, sb) = fixture();
        let hits = sb.glob("src/**/*.rs", None).unwrap();
        assert!(hits.iter().any(|p| p.ends_with("main.rs")));
        assert!(hits.iter().any(|p| p.ends_with("lib.rs")));
        assert!(!hits.iter().any(|p| p.ends_with("README.md")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn grep_finds_matches_with_line_numbers() {
        let (root, sb) = fixture();
        let hits = sb.grep("println", None, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].contains("src/main.rs:2:"), "got: {:?}", hits);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn grep_scoped_to_single_path() {
        let (root, sb) = fixture();
        // `hello` appears in main.rs and README.md; scope to main.rs only.
        let hits = sb.grep("hello", Some("src/main.rs"), None).unwrap();
        assert!(hits.iter().all(|h| h.starts_with("src/main.rs:")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn run_tool_unknown_name_is_error_string() {
        let (root, sb) = fixture();
        let obs = sb.run_tool("delete", &serde_json::json!({}));
        assert!(obs.starts_with("ERROR: unknown tool"), "got: {obs}");
        let _ = std::fs::remove_dir_all(root);
    }
}
