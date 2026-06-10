use anyhow::Result;
use serde::{Deserialize, Serialize};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Standard MCP server configuration format
/// Matches claude_desktop_config.json / .mcp.json format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Map of server name -> server configuration
    /// Using "servers" key for UMB, but also supports "mcpServers" for compatibility
    #[serde(alias = "mcpServers")]
    pub servers: IndexMap<String, ServerEntry>,
}

/// Server entry in standard MCP format
/// The name is derived from the key in the servers map, not stored in the entry
///
/// Defect #1 fix: this enum was previously `#[serde(untagged)]`. Untagged
/// serde matches purely STRUCTURALLY and tries variants top-to-bottom,
/// IGNORING the value of the `type` string. `Sse` and `Http` have an
/// identical shape (`type`, `url`, `env`, `enabled`) and `Sse` is declared
/// first, so EVERY `"type":"http"` (or `"streamable-http"`) entry
/// deserialized as `Sse` — `call_http_server` / `discover_tools_from_http_server`
/// were unreachable dead code. The fix is a hand-written `Deserialize`
/// (below) that treats `type` as a REAL discriminator. `Serialize` stays
/// derived; it emits each variant's fields untagged-style and the custom
/// `Deserialize` routes them back by `type`/`command`, so save→load
/// round-trips. Back-compat preserved: stdio (with `command`, `type`
/// absent or `"stdio"`), `sse`, and the bare-string comment form all still
/// parse to the correct variant.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ServerEntry {
    /// Standard stdio-based MCP server
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
        /// Optional type field (defaults to stdio if not present)
        #[serde(rename = "type", default)]
        transport_type: Option<String>,
        /// Optional enabled field (defaults to true if not present)
        #[serde(default = "default_enabled")]
        enabled: bool,
        /// Optional name field (for configs that include name in the entry)
        #[serde(default)]
        name: Option<String>,
    },
    /// SSE-based MCP server (Server-Sent Events)
    Sse {
        #[serde(rename = "type")]
        transport_type: String,
        url: String,
        #[serde(default)]
        env: HashMap<String, String>,
        /// Optional enabled field (defaults to true if not present)
        #[serde(default = "default_enabled")]
        enabled: bool,
    },
    /// HTTP-based MCP server (plain HTTP POST without SSE handshake)
    Http {
        #[serde(rename = "type")]
        transport_type: String,
        url: String,
        #[serde(default)]
        env: HashMap<String, String>,
        /// Optional enabled field (defaults to true if not present)
        #[serde(default = "default_enabled")]
        enabled: bool,
    },
    /// Comment entry (e.g., "_comment_official": "=== Official servers ===")
    /// These are ignored during server registration but allow JSON comments
    Comment(String),
}

/// Default value for enabled field (true)
fn default_enabled() -> bool {
    true
}

/// Defect #1 fix: hand-written `Deserialize` that uses the `type` field as
/// a REAL discriminator (the old `#[serde(untagged)]` ignored `type`'s
/// value, so `"type":"http"` wrongly matched the structurally-identical
/// `Sse` variant first).
///
/// Routing (back-compat preserved):
/// - bare JSON string                       → `Comment`
/// - `type` == "sse"                        → `Sse`
/// - `type` == "http" | "streamable-http"   → `Http`
/// - `type` == "stdio" OR no `type` present → `Stdio` (requires `command`)
///
/// We deserialize into a `serde_json::Value` first (cheap, the config is
/// tiny) and re-route into per-variant helper structs, so existing
/// `servers.json` files (stdio with/without `type`, sse, http,
/// streamable-http, comment string) all parse to the correct variant and
/// save→load round-trips (derived `Serialize` emits each variant's fields;
/// this routes them back).
impl<'de> Deserialize<'de> for ServerEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        use serde_json::Value;

        #[derive(Deserialize)]
        struct StdioShape {
            command: String,
            #[serde(default)]
            args: Vec<String>,
            #[serde(default)]
            env: HashMap<String, String>,
            #[serde(rename = "type", default)]
            transport_type: Option<String>,
            #[serde(default = "default_enabled")]
            enabled: bool,
            #[serde(default)]
            name: Option<String>,
        }

        #[derive(Deserialize)]
        struct UrlShape {
            #[serde(rename = "type")]
            transport_type: String,
            url: String,
            #[serde(default)]
            env: HashMap<String, String>,
            #[serde(default = "default_enabled")]
            enabled: bool,
        }

        let value = Value::deserialize(deserializer)?;

        // Bare string ⇒ Comment (the JSON-comment convention).
        if let Value::String(s) = &value {
            return Ok(ServerEntry::Comment(s.clone()));
        }

        let obj = value
            .as_object()
            .ok_or_else(|| D::Error::custom("ServerEntry must be a string or an object"))?;

        // `type` is the discriminator. Absent ⇒ infer from `command`.
        let type_str = obj
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());

        match type_str.as_deref() {
            Some("sse") => {
                let s: UrlShape = serde_json::from_value(value.clone())
                    .map_err(D::Error::custom)?;
                Ok(ServerEntry::Sse {
                    transport_type: s.transport_type,
                    url: s.url,
                    env: s.env,
                    enabled: s.enabled,
                })
            }
            Some("http") | Some("streamable-http") => {
                let s: UrlShape = serde_json::from_value(value.clone())
                    .map_err(D::Error::custom)?;
                Ok(ServerEntry::Http {
                    transport_type: s.transport_type,
                    url: s.url,
                    env: s.env,
                    enabled: s.enabled,
                })
            }
            // "stdio" OR no type at all ⇒ Stdio (must carry `command`).
            Some("stdio") | None => {
                let s: StdioShape = serde_json::from_value(value.clone())
                    .map_err(|e| {
                        D::Error::custom(format!(
                            "stdio server entry requires a `command` field: {e}"
                        ))
                    })?;
                Ok(ServerEntry::Stdio {
                    command: s.command,
                    args: s.args,
                    env: s.env,
                    transport_type: s.transport_type,
                    enabled: s.enabled,
                    name: s.name,
                })
            }
            Some(other) => Err(D::Error::custom(format!(
                "unknown server entry `type`: {other:?} \
                 (expected one of: stdio, sse, http, streamable-http)"
            ))),
        }
    }
}

impl ServerEntry {
    /// Get the command for stdio servers, None for SSE/HTTP/Comment
    pub fn command(&self) -> Option<&str> {
        match self {
            ServerEntry::Stdio { command, .. } => Some(command),
            ServerEntry::Sse { .. } => None,
            ServerEntry::Http { .. } => None,
            ServerEntry::Comment(_) => None,
        }
    }

    /// Get args for stdio servers, empty vec for SSE/HTTP/Comment
    pub fn args(&self) -> &[String] {
        match self {
            ServerEntry::Stdio { args, .. } => args,
            ServerEntry::Sse { .. } => &[],
            ServerEntry::Http { .. } => &[],
            ServerEntry::Comment(_) => &[],
        }
    }

    /// Get environment variables (empty for Comment)
    pub fn env(&self) -> &HashMap<String, String> {
        // Static empty map for Comment variant
        static EMPTY_MAP: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();
        match self {
            ServerEntry::Stdio { env, .. } => env,
            ServerEntry::Sse { env, .. } => env,
            ServerEntry::Http { env, .. } => env,
            ServerEntry::Comment(_) => EMPTY_MAP.get_or_init(HashMap::new),
        }
    }

    /// Check if this is an SSE server
    pub fn is_sse(&self) -> bool {
        match self {
            ServerEntry::Stdio { transport_type, .. } => {
                transport_type.as_ref().map(|t| t == "sse").unwrap_or(false)
            }
            ServerEntry::Sse { .. } => true,
            ServerEntry::Http { .. } => false,
            ServerEntry::Comment(_) => false,
        }
    }

    /// Check if this is an HTTP server
    pub fn is_http(&self) -> bool {
        match self {
            ServerEntry::Stdio { transport_type, .. } => {
                transport_type.as_ref().map(|t| t == "http").unwrap_or(false)
            }
            ServerEntry::Http { .. } => true,
            ServerEntry::Sse { .. } => false,
            ServerEntry::Comment(_) => false,
        }
    }

    /// Check if this is a comment entry (not a real server)
    pub fn is_comment(&self) -> bool {
        matches!(self, ServerEntry::Comment(_))
    }

    /// Get SSE URL if this is an SSE server
    pub fn sse_url(&self) -> Option<&str> {
        match self {
            ServerEntry::Sse { url, .. } => Some(url),
            _ => None,
        }
    }

    /// Get HTTP URL if this is an HTTP server
    pub fn http_url(&self) -> Option<&str> {
        match self {
            ServerEntry::Http { url, .. } => Some(url),
            _ => None,
        }
    }

    /// Check if this server is enabled (comments are always disabled)
    pub fn is_enabled(&self) -> bool {
        match self {
            ServerEntry::Stdio { enabled, .. } => *enabled,
            ServerEntry::Sse { enabled, .. } => *enabled,
            ServerEntry::Http { enabled, .. } => *enabled,
            ServerEntry::Comment(_) => false, // Comments are never "enabled"
        }
    }
}

impl RegistryConfig {
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            // Create default servers.json with example configuration
            let default_config = Self::create_default();
            default_config.save(path.clone())?;
            tracing::info!("[Registry] Created default servers.json at {:?}", path);
            return Ok(default_config);
        }

        let content = fs::read_to_string(&path)?;

        // PER-ENTRY RESILIENT LOAD (reliability moat). The whole-file
        // `from_str::<Self>` path is fatal-by-entry: with the strict
        // per-entry `Deserialize for ServerEntry`, a SINGLE typo'd /
        // unknown-`type` server entry made serde fail the ENTIRE file,
        // which then discarded every server and moved the user's
        // servers.json to .json.backup — one bad entry silently wiped a
        // user's whole MCP setup. Unacceptable.
        //
        // Fix: parse LENIENTLY first — the servers collection as
        // `IndexMap<String, serde_json::Value>` (SAME map type the rest of
        // the code uses ⇒ insertion order preserved; SAME
        // `#[serde(alias = "mcpServers")]` compat). Then deserialize each
        // entry INDIVIDUALLY: a bad entry is skipped with a clear warning
        // naming its key + the error; every valid entry still loads, in
        // order. The whole-file fallback (backup + empty) is now reserved
        // STRICTLY for a file that is not valid JSON at all (top-level
        // syntax error) — and even then it logs loudly and names the
        // backup path; it NEVER silently empties a valid-JSON file because
        // of one bad entry.
        #[derive(serde::Deserialize)]
        struct LenientRegistry {
            #[serde(alias = "mcpServers", default)]
            servers: IndexMap<String, serde_json::Value>,
        }

        match serde_json::from_str::<LenientRegistry>(&content) {
            Ok(lenient) => {
                let mut servers: IndexMap<String, ServerEntry> = IndexMap::new();
                let mut skipped = 0usize;
                // IndexMap iteration preserves insertion order, so the
                // resulting registry keeps the file's server order.
                for (key, raw) in lenient.servers {
                    match serde_json::from_value::<ServerEntry>(raw) {
                        Ok(entry) => {
                            servers.insert(key, entry);
                        }
                        Err(entry_err) => {
                            skipped += 1;
                            tracing::warn!(
                                "[Registry] Skipping invalid server entry '{}' \
                                 in servers.json ({}). The rest of your \
                                 configuration is unaffected; fix this entry \
                                 and it will load on next start.",
                                key,
                                entry_err
                            );
                        }
                    }
                }
                if skipped > 0 {
                    tracing::warn!(
                        "[Registry] Loaded {} valid server entr{} ({} skipped \
                         due to errors). servers.json was NOT modified.",
                        servers.len(),
                        if servers.len() == 1 { "y" } else { "ies" },
                        skipped
                    );
                }
                Ok(Self { servers })
            }
            Err(e) => {
                // GENUINELY un-parseable JSON (top-level syntax error) —
                // NOT "valid JSON with one bad entry" (that path above
                // never reaches here). Be LOUD; never a silent total loss.
                tracing::error!(
                    "[Registry] servers.json is not valid JSON (top-level \
                     parse error): {}. This is a malformed FILE, not a bad \
                     entry — UMB cannot read ANY servers from it.",
                    e
                );
                let backup_path = path.with_extension("json.backup");
                match fs::copy(&path, &backup_path) {
                    Ok(_) => tracing::error!(
                        "[Registry] Preserved the unreadable file at {:?} \
                         and is starting with an EMPTY server set. Restore \
                         or fix that file to recover your configuration.",
                        backup_path
                    ),
                    Err(backup_err) => tracing::error!(
                        "[Registry] Could not back up the malformed \
                         servers.json ({}); the original file at {:?} is \
                         left in place. Starting with an EMPTY server set.",
                        backup_err,
                        path
                    ),
                }
                Ok(Self { servers: IndexMap::new() })
            }
        }
    }

    /// Create a default configuration with example servers
    fn create_default() -> Self {
        let mut servers = IndexMap::new();

        // Add example servers in standard MCP format
        servers.insert("filesystem".to_string(), ServerEntry::Stdio {
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-filesystem".to_string(),
                dirs::home_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/home".to_string()),
            ],
            env: HashMap::new(),
            transport_type: None,
            enabled: true,
            name: None,
        });

        servers.insert("memory".to_string(), ServerEntry::Stdio {
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-memory".to_string(),
            ],
            env: HashMap::new(),
            transport_type: None,
            enabled: true,
            name: None,
        });

        servers.insert("fetch".to_string(), ServerEntry::Stdio {
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-fetch".to_string(),
            ],
            env: HashMap::new(),
            transport_type: None,
            enabled: true,
            name: None,
        });

        Self { servers }
    }

    pub fn save(&self, path: PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(&self)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)?;
        Ok(())
    }

    pub fn get_server(&self, name: &str) -> Option<&ServerEntry> {
        self.servers.get(name)
    }

    /// List all stdio servers (SSE servers are handled separately)
    pub fn list_stdio_servers(&self) -> Vec<(&String, &ServerEntry)> {
        self.servers
            .iter()
            .filter(|(_, entry)| !entry.is_sse())
            .collect()
    }

    /// List all SSE servers
    pub fn list_sse_servers(&self) -> Vec<(&String, &ServerEntry)> {
        self.servers
            .iter()
            .filter(|(_, entry)| entry.is_sse())
            .collect()
    }
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            servers: IndexMap::new(),
        }
    }
}

/// ServerRegistry tracks the set of configured (enabled) MCP servers.
///
/// All enabled servers are always active — there is no tier-based limit in the
/// open release. The per-server `enabled: false` toggle is a user control and
/// is still honored.
#[derive(Debug, Clone)]
pub struct ServerRegistry {
    /// All enabled servers from config (name -> entry)
    pub all_servers: IndexMap<String, ServerEntry>,
    /// Names of active servers (== all enabled servers)
    pub active_servers: Vec<String>,
}

impl ServerRegistry {
    /// Load registry from config.
    ///
    /// Servers with `enabled: false` are excluded. SSE servers are included but
    /// handled separately for discovery. Every enabled server is active.
    pub fn load(path: PathBuf) -> Result<Self> {
        let config = RegistryConfig::load(path)?;

        let mut all_servers = IndexMap::new();
        let mut active_servers = Vec::new();

        for (name, entry) in config.servers.into_iter() {
            if !entry.is_enabled() {
                tracing::debug!("[Registry] Server '{}' disabled, skipping", name);
                continue;
            }
            let is_sse = entry.is_sse();
            all_servers.insert(name.clone(), entry);
            active_servers.push(name.clone());
            if is_sse {
                tracing::info!("[Registry] SSE server '{}' registered (discovery pending)", name);
            }
        }

        Ok(Self {
            all_servers,
            active_servers,
        })
    }

    pub fn is_active(&self, name: &str) -> bool {
        self.active_servers.contains(&name.to_string())
    }

    pub fn active_count(&self) -> usize {
        self.active_servers.len()
    }

    pub fn total_count(&self) -> usize {
        self.all_servers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ServerEntry {
        serde_json::from_str::<ServerEntry>(json).expect("ServerEntry should parse")
    }

    /// Defect #1: stdio entry with NO `type` field → `Stdio` (back-compat
    /// with every existing `servers.json`).
    #[test]
    fn test_stdio_no_type_field() {
        let e = parse(r#"{"command":"npx","args":["-y","@mcp/fs"]}"#);
        assert!(matches!(e, ServerEntry::Stdio { .. }), "got {e:?}");
        assert_eq!(e.command(), Some("npx"));
        assert!(!e.is_sse());
        assert!(!e.is_http());
    }

    /// stdio entry with explicit `"type":"stdio"` → `Stdio`.
    #[test]
    fn test_stdio_explicit_type() {
        let e = parse(r#"{"type":"stdio","command":"node","args":["s.js"]}"#);
        assert!(matches!(e, ServerEntry::Stdio { .. }), "got {e:?}");
        assert_eq!(e.command(), Some("node"));
    }

    /// `"type":"sse"` → `Sse` (declared before Http; must NOT be shadowed).
    #[test]
    fn test_sse_type() {
        let e = parse(r#"{"type":"sse","url":"https://x/sse"}"#);
        assert!(matches!(e, ServerEntry::Sse { .. }), "got {e:?}");
        assert!(e.is_sse());
        assert_eq!(e.sse_url(), Some("https://x/sse"));
        assert!(!e.is_http());
    }

    /// Defect #1 CORE: `"type":"http"` → `Http` (previously wrongly
    /// matched `Sse` under `#[serde(untagged)]`, making `call_http_server`
    /// / `discover_tools_from_http_server` dead code). Asserting
    /// `is_http()` + `http_url()` proves the HTTP transport is now
    /// REACHABLE from the call/discovery dispatch.
    #[test]
    fn test_http_type_now_reaches_http_variant() {
        let e = parse(r#"{"type":"http","url":"https://x/mcp"}"#);
        assert!(
            matches!(e, ServerEntry::Http { .. }),
            "REGRESSION: http entry must be Http, got {e:?}"
        );
        assert!(e.is_http(), "is_http() must be true → call_http_server reachable");
        assert!(!e.is_sse(), "http must NOT be misclassified as sse");
        assert_eq!(e.http_url(), Some("https://x/mcp"));
    }

    /// `"type":"streamable-http"` also → `Http` (MCP streamable-http alias).
    #[test]
    fn test_streamable_http_type() {
        let e = parse(r#"{"type":"streamable-http","url":"https://y/mcp"}"#);
        assert!(matches!(e, ServerEntry::Http { .. }), "got {e:?}");
        assert!(e.is_http());
        assert_eq!(e.http_url(), Some("https://y/mcp"));
    }

    /// Bare JSON string → `Comment` (the JSON-comment convention).
    #[test]
    fn test_comment_string_form() {
        let e = parse(r#""=== Official servers ===""#);
        assert!(matches!(e, ServerEntry::Comment(_)), "got {e:?}");
        assert!(e.is_comment());
        assert!(!e.is_enabled(), "comments are never enabled");
    }

    /// Unknown `type` is a hard error (no silent mis-route).
    #[test]
    fn test_unknown_type_is_error() {
        let r = serde_json::from_str::<ServerEntry>(r#"{"type":"ftp","url":"x"}"#);
        assert!(r.is_err(), "unknown transport type must error, got {r:?}");
    }

    /// Serialize → deserialize round-trip preserves the variant for ALL
    /// kinds (the derived untagged `Serialize` + the custom discriminating
    /// `Deserialize` must agree).
    #[test]
    fn test_serialize_deserialize_round_trip_all_variants() {
        let cfgs = vec![
            r#"{"command":"npx","args":["a"]}"#,
            r#"{"type":"stdio","command":"node"}"#,
            r#"{"type":"sse","url":"https://s/sse"}"#,
            r#"{"type":"http","url":"https://h/mcp"}"#,
            r#"{"type":"streamable-http","url":"https://sh/mcp"}"#,
            r#""=== a comment ===""#,
        ];
        for c in cfgs {
            let parsed = parse(c);
            let ser = serde_json::to_string(&parsed).expect("serialize");
            let reparsed = parse(&ser);
            // The discriminant (variant) must survive the round-trip.
            assert_eq!(
                std::mem::discriminant(&parsed),
                std::mem::discriminant(&reparsed),
                "round-trip changed variant for {c} (serialized: {ser})"
            );
        }
    }

    /// Defect #1 end-to-end via RegistryConfig: a full servers.json mixing
    /// stdio/sse/http/streamable-http/comment parses each to the right
    /// variant (proves the fix holds through the real load path, not just
    /// the bare enum).
    #[test]
    fn test_registry_config_mixed_transports() {
        let json = r#"{
          "servers": {
            "fs":   {"command":"npx","args":["-y","@mcp/fs"]},
            "note": "=== remote servers ===",
            "sse1": {"type":"sse","url":"https://a/sse"},
            "http1":{"type":"http","url":"https://b/mcp"},
            "sh1":  {"type":"streamable-http","url":"https://c/mcp"}
          }
        }"#;
        let cfg: RegistryConfig =
            serde_json::from_str(json).expect("RegistryConfig parses");
        assert!(matches!(cfg.servers["fs"], ServerEntry::Stdio { .. }));
        assert!(matches!(cfg.servers["note"], ServerEntry::Comment(_)));
        assert!(matches!(cfg.servers["sse1"], ServerEntry::Sse { .. }));
        assert!(
            matches!(cfg.servers["http1"], ServerEntry::Http { .. }),
            "http1 must be Http (Defect #1 core)"
        );
        assert!(matches!(cfg.servers["sh1"], ServerEntry::Http { .. }));
        assert!(cfg.servers["http1"].is_http());
        assert_eq!(cfg.servers["http1"].http_url(), Some("https://b/mcp"));
    }

    // ===================================================================
    // #33 — RegistryConfig::load PER-ENTRY resilience
    // ===================================================================

    /// Helper: write `content` to a temp file, return its path. We use a
    /// distinct dir per test so the `.json.backup` sibling check is
    /// unambiguous. Returned `TempDir` must be kept alive by the caller.
    fn write_cfg(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("servers.json");
        std::fs::write(&path, content).expect("write servers.json");
        (dir, path)
    }

    /// (a) ≥2 valid entries + 1 unknown/typo `type` entry ⇒ exactly the
    /// valid ones load, the bad one is skipped, and the on-disk file is
    /// UNCHANGED (NOT moved to .json.backup). This is the verified repro
    /// from the bug report (`"type":"studio"` typo).
    #[test]
    fn test_load_skips_bad_entry_keeps_valid_and_file_untouched() {
        let original = r#"{"servers":{
            "fs":{"command":"npx","args":["a"]},
            "mem":{"command":"node","args":["m.js"]},
            "x":{"type":"studio","command":"npx"}
        }}"#;
        let (_dir, path) = write_cfg(original);

        let cfg = RegistryConfig::load(path.clone()).expect("load must not fail");

        // Exactly the two valid entries loaded; the typo'd one skipped.
        assert_eq!(cfg.servers.len(), 2, "only the valid entries load");
        assert!(cfg.servers.contains_key("fs"));
        assert!(cfg.servers.contains_key("mem"));
        assert!(
            !cfg.servers.contains_key("x"),
            "the unknown-`type` entry must be skipped, NOT load"
        );
        // Insertion order preserved (fs before mem).
        let keys: Vec<&String> = cfg.servers.keys().collect();
        assert_eq!(keys, vec!["fs", "mem"], "file order preserved");

        // CRITICAL: the user's file is byte-identical and NO backup made.
        let on_disk = std::fs::read_to_string(&path).expect("file still readable");
        assert_eq!(on_disk, original, "servers.json must be UNCHANGED");
        assert!(
            !path.with_extension("json.backup").exists(),
            "#33 REGRESSION: a single bad entry must NOT trigger a \
             whole-file backup/wipe"
        );
    }

    /// (b) valid entries + a legacy stdio+`type:"sse"` WITHOUT `url`
    /// (functionally dead in old code: Stdio.sse_url()==None ⇒ never
    /// registered) ⇒ that entry is skipped+warned, the rest still load.
    /// Honest skip, no resurrection logic.
    #[test]
    fn test_load_skips_legacy_stdio_type_sse_without_url() {
        // `{"type":"sse","command":"npx"}` — has `type:sse` so the strict
        // deserializer routes it to the Sse variant, which REQUIRES `url`;
        // missing `url` ⇒ per-entry error ⇒ skipped (correct honest
        // handling — it was already dead config).
        let original = r#"{"servers":{
            "good":{"command":"npx","args":["x"]},
            "legacy":{"type":"sse","command":"npx"}
        }}"#;
        let (_dir, path) = write_cfg(original);

        let cfg = RegistryConfig::load(path.clone()).expect("load");
        assert_eq!(cfg.servers.len(), 1, "only the valid entry loads");
        assert!(cfg.servers.contains_key("good"));
        assert!(!cfg.servers.contains_key("legacy"), "dead legacy entry skipped");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "file unchanged"
        );
        assert!(!path.with_extension("json.backup").exists());
    }

    /// (c) a genuinely-malformed JSON file (top-level syntax error) ⇒ the
    /// defined LOUD behavior: empty config + a `.json.backup` made
    /// (preserving the file). Distinct from (a): a valid-JSON file with one
    /// bad entry must NEVER hit this path / lose all config — re-asserted.
    #[test]
    fn test_load_malformed_json_file_loud_backup_no_silent_loss() {
        // Not valid JSON at all (trailing garbage / unclosed).
        let (_dir, path) = write_cfg(r#"{"servers": { this is not json "#);
        let cfg = RegistryConfig::load(path.clone()).expect("load returns Ok");
        assert!(cfg.servers.is_empty(), "unparseable file ⇒ empty set");
        assert!(
            path.with_extension("json.backup").exists(),
            "a genuinely malformed FILE is preserved as .json.backup"
        );

        // Re-assert the contrast: valid JSON + one bad entry NEVER behaves
        // like a malformed file (no backup, no total loss).
        let (_d2, p2) = write_cfg(
            r#"{"servers":{"ok":{"command":"npx"},"bad":{"type":"nope","url":"u"}}}"#,
        );
        let cfg2 = RegistryConfig::load(p2.clone()).expect("load");
        assert_eq!(cfg2.servers.len(), 1, "valid entry survives a bad sibling");
        assert!(
            !p2.with_extension("json.backup").exists(),
            "#33 REGRESSION: valid-JSON-one-bad-entry must not be treated \
             as a malformed file"
        );
    }

    /// (d) mcpServers alias + all-valid still loads every entry in order
    /// (back-compat: the lenient parse keeps the `#[serde(alias)]`).
    #[test]
    fn test_load_mcpservers_alias_all_valid_preserved() {
        let (_dir, path) = write_cfg(
            r#"{"mcpServers":{
                "a":{"command":"npx"},
                "b":{"type":"http","url":"https://h/mcp"},
                "c":{"type":"sse","url":"https://s/sse"}
            }}"#,
        );
        let cfg = RegistryConfig::load(path).expect("load");
        let keys: Vec<&String> = cfg.servers.keys().collect();
        assert_eq!(keys, vec!["a", "b", "c"], "alias + order preserved");
        assert!(matches!(cfg.servers["b"], ServerEntry::Http { .. }));
        assert!(matches!(cfg.servers["c"], ServerEntry::Sse { .. }));
    }
}
