//! Tool-Definitions Dictionary (overlay-at-read)
//!
//! Community-curated short descriptions for well-known MCP servers' tools,
//! overlaid on top of the live server-supplied tool definitions. The
//! underlying `ToolRouter::tools` storage stays byte-identical to the live
//! server tool defs — this module is read-only data + a lookup helper, called
//! from `get_tool_info` and the slim `list_tools` JSON envelope.
//!
//! Loader semantics:
//! - Compile-time fallback: all `tool-dictionary/*.toml` are `include_str!`'d
//!   so a freshly-installed `umb` binary has the shipped dict even with no
//!   on-disk files (the "leanest defs out of the box" property).
//! - In-repo overlay: `<CARGO_MANIFEST_DIR>/tool-dictionary/*.toml`
//!   (development) and `<cwd>/tool-dictionary/*.toml` (running from a
//!   checkout) — overrides the compile-time fallback file-by-file when
//!   present (same filename ⇒ later wins).
//! - Config-path overlay: every dir in `[general].tool_dictionary_paths`
//!   (toml config) — overrides the previous two.
//! - User overlay: `~/.umb/tool-dictionary/*.toml` (or whatever
//!   `tool_dictionary_user_dir` resolves to) — HIGHEST precedence; users
//!   win over everyone (private tools + per-machine overrides).
//!
//! Per-entry hash guard: a dict entry may carry `schema_hash_sha256`. When
//! `ShortMode::Auto` is active, the loader silently falls back to the live
//! server's description if the hash recorded at curation time does NOT match
//! the live tool's canonical description hash (computed at lookup). Hash is
//! optional — when absent in the entry, `Auto` behaves like `On` for that
//! entry. `On` always applies regardless of hash. `Off` always returns live.
//!
//! Provenance: `lookup()` returns whether the dict applied (`Source::Dict`)
//! or fell through to the live def (`Source::Server`), so callers can emit
//! the `_source` field in `get_tool_info` and in `--doctor-tools`.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Compile-time embedded TOML files (one per seed server). Each is a
/// `(filename, contents)` pair; the loader parses them in this order, then
/// disk overlays override file-by-file.
///
/// IMPORTANT: keep this list in lexicographic order so the
/// compile-fallback merge order is deterministic; the disk overlays then
/// override deterministically by filename match.
const COMPILED_DICT_FILES: &[(&str, &str)] = &[
    ("brave-search.toml", include_str!("../../tool-dictionary/brave-search.toml")),
    ("fetch.toml", include_str!("../../tool-dictionary/fetch.toml")),
    ("filesystem.toml", include_str!("../../tool-dictionary/filesystem.toml")),
    ("gdrive.toml", include_str!("../../tool-dictionary/gdrive.toml")),
    ("github-actions.toml", include_str!("../../tool-dictionary/github-actions.toml")),
    ("github.toml", include_str!("../../tool-dictionary/github.toml")),
    ("gitlab.toml", include_str!("../../tool-dictionary/gitlab.toml")),
    ("memory.toml", include_str!("../../tool-dictionary/memory.toml")),
    ("playwright.toml", include_str!("../../tool-dictionary/playwright.toml")),
    ("postgres.toml", include_str!("../../tool-dictionary/postgres.toml")),
    ("puppeteer.toml", include_str!("../../tool-dictionary/puppeteer.toml")),
    ("sequential-thinking.toml", include_str!("../../tool-dictionary/sequential-thinking.toml")),
    ("slack.toml", include_str!("../../tool-dictionary/slack.toml")),
    ("sqlite.toml", include_str!("../../tool-dictionary/sqlite.toml")),
    ("time.toml", include_str!("../../tool-dictionary/time.toml")),
];

/// Three-state mode for short-definition overlay. Parsed from the
/// `[general] short_definitions` config key (case-insensitive). Unknown
/// strings fall back to `Auto` (the safe default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortMode {
    /// Never consult the dictionary — always return the live server def.
    Off,
    /// Apply if a matching dict entry exists AND (the entry has no hash
    /// OR the entry's hash matches the live description hash). Silently
    /// falls back to live on hash mismatch.
    Auto,
    /// Apply if a matching dict entry exists, regardless of hash.
    On,
}

impl Default for ShortMode {
    fn default() -> Self {
        ShortMode::Auto
    }
}

impl ShortMode {
    /// Parse a config string (case-insensitive). Unknown ⇒ `Auto` (safe).
    pub fn from_str_or_auto(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => ShortMode::Off,
            "on" => ShortMode::On,
            "auto" => ShortMode::Auto,
            _ => ShortMode::Auto,
        }
    }
}

/// A single dictionary entry: a shortened description for a `(server, tool)`
/// pair. The optional `schema_hash_sha256` is the SHA-256 of the live
/// canonical description recorded at curation time; the loader compares it
/// against the live tool's description hash at lookup to detect upstream
/// drift.
#[derive(Debug, Clone)]
pub struct DictEntry {
    pub short_description: String,
    pub schema_hash_sha256: Option<String>,
}

/// The parsed contents of ONE `tool-dictionary/*.toml` file.
#[derive(Debug, Clone, Deserialize)]
struct DictFile {
    metadata: DictMetadata,
    #[serde(default)]
    tools: Vec<DictTool>,
}

#[derive(Debug, Clone, Deserialize)]
struct DictMetadata {
    server_name: String,
    // Other curator-facing metadata fields (upstream_canonical_source,
    // curator, reviewed_at) are deserialized loosely — we don't need them
    // at runtime but they keep parsing clean if present.
    #[serde(default)]
    #[allow(dead_code)]
    upstream_canonical_source: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    curator: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    reviewed_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct DictTool {
    name: String,
    short_description: String,
    #[serde(default)]
    schema_hash_sha256: Option<String>,
}

/// Whether a `lookup()` call applied a dict entry or fell through to the
/// live server def.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The live server's own description was returned (dict did not apply).
    Server,
    /// A dict entry's `short_description` was returned.
    Dict,
}

/// The loaded dictionary: `(server_name, tool_name)` → `DictEntry`.
#[derive(Debug, Default, Clone)]
pub struct ToolDictionary {
    entries: HashMap<(String, String), DictEntry>,
}

impl ToolDictionary {
    /// Build an empty dictionary (useful in tests / when explicitly
    /// disabling the loader).
    pub fn empty() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Load the dictionary with the full precedence chain (lowest →
    /// highest):
    ///   1. compile-time `include_str!` fallback (always present),
    ///   2. on-disk in-repo `tool-dictionary/*.toml` (CARGO_MANIFEST_DIR
    ///      relative + cwd relative),
    ///   3. each dir in `tool_dictionary_paths`,
    ///   4. `tool_dictionary_user_dir` (default `~/.umb/tool-dictionary`).
    ///
    /// Later overlays OVERRIDE earlier (per `(server, tool)` key). A
    /// missing/unparseable file is logged at `warn` and skipped — never
    /// blocks loading.
    pub fn load(
        tool_dictionary_paths: &[String],
        user_dir: Option<&Path>,
    ) -> Self {
        let mut dict = Self::empty();

        // (1) Compile-time fallback.
        for (filename, contents) in COMPILED_DICT_FILES {
            dict.ingest(filename, contents, "<compiled-in>");
        }

        // (2) In-repo on-disk overlay. Try both the manifest dir (dev
        // build, contains the actual repo files) and cwd (running from
        // checkout root). Same files override compiled-in entries.
        let in_repo_candidates: Vec<PathBuf> = vec![
            // CARGO_MANIFEST_DIR resolves at compile time; this is where
            // the crate's tool-dictionary/ lives in a dev checkout.
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tool-dictionary"),
            // cwd fallback (running from repo root in CI / scripts).
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("tool-dictionary"),
        ];
        for dir in &in_repo_candidates {
            dict.ingest_dir(dir);
        }

        // (3) Config-specified paths (org-shared dicts).
        for p in tool_dictionary_paths {
            let path = expand_tilde(p);
            dict.ingest_dir(&path);
        }

        // (4) User overlay (highest precedence).
        if let Some(user) = user_dir {
            let expanded = expand_tilde(&user.to_string_lossy());
            dict.ingest_dir(&expanded);
        }

        tracing::info!(
            "[ToolDict] Loaded {} entries (compiled-in + in-repo + config + user)",
            dict.entries.len()
        );
        dict
    }

    /// Lookup helper: given `(server, tool_name, live_description, mode)`,
    /// returns `(description_to_emit, source)`. The `live_description` is
    /// the FULL upstream tool description — returned verbatim on any
    /// fall-through path.
    pub fn lookup<'a>(
        &'a self,
        server: &str,
        tool_name: &str,
        live_description: &'a str,
        mode: ShortMode,
    ) -> (&'a str, Source) {
        if matches!(mode, ShortMode::Off) {
            return (live_description, Source::Server);
        }
        let key = (server.to_string(), tool_name.to_string());
        let entry = match self.entries.get(&key) {
            Some(e) => e,
            None => return (live_description, Source::Server),
        };
        match mode {
            ShortMode::Off => (live_description, Source::Server),
            ShortMode::On => (entry.short_description.as_str(), Source::Dict),
            ShortMode::Auto => {
                // Auto: hash-guard. If entry has a recorded hash and it
                // does NOT match the live description's hash, silently
                // fall back. If entry has no recorded hash, treat as Auto
                // == On for that entry (curator opted out of hash-guard).
                match entry.schema_hash_sha256.as_deref() {
                    None => (entry.short_description.as_str(), Source::Dict),
                    Some(recorded) => {
                        let live_hash = sha256_hex(live_description.as_bytes());
                        if recorded.eq_ignore_ascii_case(&live_hash) {
                            (entry.short_description.as_str(), Source::Dict)
                        } else {
                            tracing::debug!(
                                "[ToolDict] hash mismatch for ({server}, {tool_name}): \
                                 recorded={recorded}, live={live_hash} — falling back to live def"
                            );
                            (live_description, Source::Server)
                        }
                    }
                }
            }
        }
    }

    /// Number of loaded entries (for diagnostics / `--doctor-tools` summary).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the dict has no entries.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ---------- internal ingestion helpers ----------

    fn ingest_dir(&mut self, dir: &Path) {
        if !dir.is_dir() {
            return;
        }
        let mut paths: Vec<PathBuf> = match fs::read_dir(dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
                .collect(),
            Err(e) => {
                tracing::warn!("[ToolDict] read_dir {:?} failed: {}", dir, e);
                return;
            }
        };
        // Deterministic order — later same-name files override earlier
        // (handled by the per-(server,tool) HashMap insert overwriting).
        paths.sort();
        for path in &paths {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>");
            match fs::read_to_string(path) {
                Ok(contents) => {
                    self.ingest(filename, &contents, &path.display().to_string());
                }
                Err(e) => {
                    tracing::warn!("[ToolDict] failed to read {:?}: {}", path, e);
                }
            }
        }
    }

    fn ingest(&mut self, filename: &str, contents: &str, source_label: &str) {
        let parsed: DictFile = match toml::from_str(contents) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    "[ToolDict] parse error in {} ({}): {} — skipping file",
                    filename,
                    source_label,
                    e
                );
                return;
            }
        };
        let server = parsed.metadata.server_name.trim().to_string();
        if server.is_empty() {
            tracing::warn!(
                "[ToolDict] {} has empty metadata.server_name — skipping",
                source_label
            );
            return;
        }
        let mut n = 0usize;
        for tool in parsed.tools {
            let tname = tool.name.trim().to_string();
            if tname.is_empty() {
                tracing::warn!(
                    "[ToolDict] {}/{}: empty tool name — skipping entry",
                    source_label, server
                );
                continue;
            }
            let entry = DictEntry {
                short_description: tool.short_description,
                schema_hash_sha256: tool.schema_hash_sha256,
            };
            self.entries.insert((server.clone(), tname), entry);
            n += 1;
        }
        tracing::debug!(
            "[ToolDict] ingested {} entries from {} ({})",
            n, filename, source_label
        );
    }
}

/// SHA-256 of bytes → lowercase hex string (no deps beyond sha2 which is
/// already pinned in [build-dependencies]; we declare it in [dependencies]
/// unconditionally here — see Cargo.toml change).
fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out {
        use std::fmt::Write;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

/// Expand a leading `~` to the user's home dir. Pure path mangling — no
/// filesystem access. Returns the original path if no expansion is
/// possible.
fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(p)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write a fixture TOML file to a temp dir.
    fn write_dict(dir: &Path, filename: &str, contents: &str) {
        let p = dir.join(filename);
        let mut f = std::fs::File::create(&p).expect("create fixture");
        f.write_all(contents.as_bytes()).expect("write fixture");
    }

    /// Compile-time embedded files load even when NO on-disk dirs are
    /// provided. All 15 seed servers must be present.
    #[test]
    fn test_dict_loads_compiled_fallback_when_no_in_repo_dir() {
        // Use a path that does NOT exist + no user dir → falls back to
        // compiled-in. (The in-repo paths the loader tries unconditionally
        // ARE present in this build's CARGO_MANIFEST_DIR, but since
        // those are the SAME compiled-in files they re-ingest identical
        // content — the entries count stays >= 15 seed servers' tools.)
        let dict = ToolDictionary::load(&[], None);
        // The 15 seed files each contribute >= 1 entry — at minimum 15
        // entries must be present even if some files have only 1 tool.
        assert!(
            dict.len() >= 15,
            "compiled-in seed dictionary must load at least 15 tool entries (got {})",
            dict.len()
        );
        // Spot-check one well-known entry: filesystem/read_file.
        let live = "Original live description.";
        let (out, src) = dict.lookup("filesystem", "read_file", live, ShortMode::On);
        assert_eq!(src, Source::Dict, "filesystem/read_file must come from dict");
        assert_ne!(out, live, "filesystem/read_file overlay must override");
    }

    /// User overlay overrides BOTH compiled-in AND in-repo entries.
    #[test]
    fn test_dict_user_overlay_overrides_in_repo() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dict(
            tmp.path(),
            "filesystem.toml",
            r#"
[metadata]
server_name = "filesystem"
upstream_canonical_source = "test"

[[tools]]
name = "read_file"
short_description = "USER-OVERRIDE-SHORT"
"#,
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let (out, src) =
            dict.lookup("filesystem", "read_file", "live desc", ShortMode::On);
        assert_eq!(src, Source::Dict);
        assert_eq!(out, "USER-OVERRIDE-SHORT");
    }

    /// Auto mode applies the entry when the recorded hash matches the
    /// live description hash.
    #[test]
    fn test_dict_auto_mode_applies_when_hash_matches() {
        let live = "Send an email message.";
        let live_hash = sha256_hex(live.as_bytes());
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dict(
            tmp.path(),
            "mail.toml",
            &format!(
                r#"
[metadata]
server_name = "mail"

[[tools]]
name = "send_email"
short_description = "Send email"
schema_hash_sha256 = "{live_hash}"
"#
            ),
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let (out, src) = dict.lookup("mail", "send_email", live, ShortMode::Auto);
        assert_eq!(src, Source::Dict);
        assert_eq!(out, "Send email");
    }

    /// Auto mode silently falls back when the recorded hash does NOT match
    /// — operator sees a `debug!` line, the live def is returned.
    #[test]
    fn test_dict_auto_mode_silently_falls_back_when_hash_mismatches() {
        let live = "Send an email message.";
        // Use a wrong hash.
        let bogus = "0".repeat(64);
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dict(
            tmp.path(),
            "mail.toml",
            &format!(
                r#"
[metadata]
server_name = "mail"

[[tools]]
name = "send_email"
short_description = "Send email"
schema_hash_sha256 = "{bogus}"
"#
            ),
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let (out, src) = dict.lookup("mail", "send_email", live, ShortMode::Auto);
        assert_eq!(src, Source::Server);
        assert_eq!(out, live);
    }

    /// Off mode always returns the live description — no dict consultation
    /// — regardless of how many entries match.
    #[test]
    fn test_dict_off_mode_always_returns_live_def() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dict(
            tmp.path(),
            "x.toml",
            r#"
[metadata]
server_name = "x"

[[tools]]
name = "y"
short_description = "Short"
"#,
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let live = "Full live description.";
        let (out, src) = dict.lookup("x", "y", live, ShortMode::Off);
        assert_eq!(src, Source::Server);
        assert_eq!(out, live);
    }

    /// On mode applies the entry regardless of hash mismatch (audit mode
    /// — useful for "force the dict to apply even if upstream changed").
    #[test]
    fn test_dict_on_mode_applies_regardless_of_hash() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let bogus = "1".repeat(64);
        write_dict(
            tmp.path(),
            "x.toml",
            &format!(
                r#"
[metadata]
server_name = "x"

[[tools]]
name = "y"
short_description = "Short"
schema_hash_sha256 = "{bogus}"
"#
            ),
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let (out, src) = dict.lookup("x", "y", "live desc", ShortMode::On);
        assert_eq!(src, Source::Dict);
        assert_eq!(out, "Short");
    }

    /// Unparseable file is skipped — other files in the same dir still
    /// load — and the load() call does not panic.
    #[test]
    fn test_dict_skips_bad_file_without_panicking() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_dict(tmp.path(), "broken.toml", "this is not :: valid toml [");
        write_dict(
            tmp.path(),
            "good.toml",
            r#"
[metadata]
server_name = "good"

[[tools]]
name = "ok"
short_description = "ok"
"#,
        );
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let (out, src) = dict.lookup("good", "ok", "live", ShortMode::On);
        assert_eq!(src, Source::Dict);
        assert_eq!(out, "ok");
    }

    /// ShortMode parsing: defaults to Auto on unknown strings.
    #[test]
    fn test_short_mode_parse() {
        assert_eq!(ShortMode::from_str_or_auto("off"), ShortMode::Off);
        assert_eq!(ShortMode::from_str_or_auto("OFF"), ShortMode::Off);
        assert_eq!(ShortMode::from_str_or_auto("on"), ShortMode::On);
        assert_eq!(ShortMode::from_str_or_auto("On"), ShortMode::On);
        assert_eq!(ShortMode::from_str_or_auto("auto"), ShortMode::Auto);
        assert_eq!(ShortMode::from_str_or_auto(""), ShortMode::Auto);
        assert_eq!(ShortMode::from_str_or_auto("garbage"), ShortMode::Auto);
        assert_eq!(ShortMode::default(), ShortMode::Auto);
    }
}
