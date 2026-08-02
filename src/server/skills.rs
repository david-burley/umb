//! Agent-skills registry with progressive disclosure.
//!
//! UMB can serve agent skills alongside MCP tools through the same stdio
//! interface. A skill is a subdirectory of the configured skills directory
//! containing a `SKILL.md` file with YAML-ish frontmatter:
//!
//! ```text
//! ---
//! name: my-skill
//! description: What the skill does (short)
//! ---
//! Full skill body in Markdown...
//! ```
//!
//! Progressive disclosure: clients fetch the compact index (`skills_list`,
//! name + short description + pinned flag) cheaply, then pull the full
//! frontmatter-stripped body on demand (`skills_read`).
//!
//! Tolerance rules (same posture as the tool-dictionary loader): a malformed
//! skill is skipped with a warning and never aborts the scan; unknown
//! frontmatter keys are accepted and ignored; folded (`>`) and literal (`|`)
//! block scalars and flow lists (`[a, b]`) parse correctly. The cache is
//! invalidated per file by (mtime, len) fingerprint, so edits are picked up
//! without a restart.

use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// One entry of the compact skills index returned by `skills_list`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SkillIndexEntry {
    pub name: String,
    pub description: String,
    pub pinned: bool,
}

/// A parsed and cached skill.
#[derive(Debug, Clone)]
struct CachedSkill {
    name: String,
    description: String,
    /// Full SKILL.md body with the frontmatter block stripped.
    body: String,
}

/// Fingerprint used for cache invalidation: mtime alone can collide on
/// coarse-granularity filesystems, so length rides along (cheap, already
/// available from the same `metadata` call).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fingerprint {
    mtime: SystemTime,
    len: u64,
}

/// Per-file cache record. `skill` is None for a malformed/unreadable file:
/// it is skipped but its fingerprint is still recorded so it is not
/// re-parsed (and re-warned) on every request.
#[derive(Debug, Clone)]
struct FileRecord {
    fingerprint: Fingerprint,
    skill: Option<CachedSkill>,
}

#[derive(Debug, Default)]
struct CacheState {
    /// SKILL.md path -> parse record
    files: HashMap<PathBuf, FileRecord>,
}

/// Registry over a skills directory. Thread-safe via interior mutability;
/// refreshed lazily on every read, re-parsing only changed files.
pub struct SkillsRegistry {
    dir: PathBuf,
    pinned: HashSet<String>,
    cache: Mutex<CacheState>,
}

impl SkillsRegistry {
    pub fn new(dir: PathBuf, pinned: Vec<String>) -> Self {
        Self {
            dir,
            pinned: pinned.into_iter().collect(),
            cache: Mutex::new(CacheState::default()),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Compact index: name + short description + pinned flag, sorted by name.
    /// This is the cheap endpoint; bodies stay in the cache until `read`.
    pub fn index(&self) -> Vec<SkillIndexEntry> {
        let cache = self.refresh();
        self.live_skills(&cache)
            .into_iter()
            .map(|s| SkillIndexEntry {
                pinned: self.pinned.contains(&s.name),
                name: s.name,
                description: s.description,
            })
            .collect()
    }

    /// Full frontmatter-stripped body for one skill, or None if unknown.
    pub fn read(&self, name: &str) -> Option<String> {
        let cache = self.refresh();
        self.live_skills(&cache)
            .into_iter()
            .find(|s| s.name == name)
            .map(|s| s.body)
    }

    /// Name-deduplicated view of the successfully parsed skills, sorted by
    /// name. On a name collision the lexicographically first path wins and
    /// the duplicate is dropped (warned about at insert time).
    fn live_skills(&self, cache: &CacheState) -> Vec<CachedSkill> {
        let mut paths: Vec<&PathBuf> = cache.files.keys().collect();
        paths.sort();
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<CachedSkill> = Vec::new();
        for p in paths {
            if let Some(skill) = cache.files[p].skill.clone() {
                if seen.insert(skill.name.clone()) {
                    out.push(skill);
                }
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Rescan the directory, re-parsing only files whose fingerprint changed
    /// and dropping records for files that disappeared. A missing or
    /// unreadable skills directory yields an empty cache, never an error.
    fn refresh(&self) -> parking_lot::MutexGuard<'_, CacheState> {
        let mut cache = self.cache.lock();

        let mut current: HashMap<PathBuf, Fingerprint> = HashMap::new();
        match fs::read_dir(&self.dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let skill_dir = entry.path();
                    if !skill_dir.is_dir() {
                        continue;
                    }
                    let skill_file = skill_dir.join("SKILL.md");
                    if let Ok(meta) = fs::metadata(&skill_file) {
                        if meta.is_file() {
                            current.insert(
                                skill_file,
                                Fingerprint {
                                    mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                                    len: meta.len(),
                                },
                            );
                        }
                    }
                }
            }
            Err(_) => {
                cache.files.clear();
                return cache;
            }
        }

        // Fast path: nothing added, removed, or modified.
        if current.len() == cache.files.len()
            && current
                .iter()
                .all(|(p, fp)| cache.files.get(p).is_some_and(|r| r.fingerprint == *fp))
        {
            return cache;
        }

        // Drop records for files that are gone.
        cache.files.retain(|p, _| current.contains_key(p));

        // (Re)parse new and changed files.
        for (path, fp) in &current {
            if cache.files.get(path).is_some_and(|r| r.fingerprint == *fp) {
                continue; // unchanged
            }
            let fallback = path
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let skill = match fs::read_to_string(path) {
                Ok(content) => match parse_skill_md(&content, &fallback) {
                    Ok(parsed) => {
                        // Duplicate-name warning; the dedup itself happens in
                        // `live_skills` (first path wins).
                        let clash = cache
                            .files
                            .values()
                            .any(|r| r.skill.as_ref().is_some_and(|s| s.name == parsed.name));
                        if clash {
                            tracing::warn!(
                                "[Skills] duplicate skill name '{}' at {:?}; \
                                 lexicographically first path wins",
                                parsed.name,
                                path
                            );
                        }
                        Some(CachedSkill {
                            name: parsed.name,
                            description: parsed.description,
                            body: parsed.body,
                        })
                    }
                    Err(e) => {
                        // Malformed skill: skip + log, never fatal.
                        tracing::warn!("[Skills] skipping malformed skill at {:?}: {}", path, e);
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("[Skills] skipping unreadable {:?}: {}", path, e);
                    None
                }
            };
            cache.files.insert(
                path.clone(),
                FileRecord {
                    fingerprint: *fp,
                    skill,
                },
            );
        }

        cache
    }
}

/// Parsed SKILL.md: identity fields plus the body with frontmatter stripped.
#[derive(Debug, Clone)]
struct ParsedSkill {
    name: String,
    description: String,
    body: String,
}

/// Parse a SKILL.md file. `fallback_name` (the directory name) is used when
/// the frontmatter has no `name` key. Returns Err for structurally invalid
/// files (no frontmatter block, or unterminated frontmatter).
fn parse_skill_md(content: &str, fallback_name: &str) -> Result<ParsedSkill, String> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let mut lines = content.lines();

    // First line must be the frontmatter fence.
    match lines.next() {
        Some(l) if l.trim() == "---" => {}
        _ => return Err("missing opening '---' frontmatter fence".to_string()),
    }

    let mut fm_lines: Vec<&str> = Vec::new();
    let mut terminated = false;
    let mut consumed = 1; // opening fence
    for line in lines {
        consumed += 1;
        if line.trim() == "---" {
            terminated = true;
            break;
        }
        fm_lines.push(line);
    }
    if !terminated {
        return Err("unterminated frontmatter (no closing '---')".to_string());
    }

    let front = parse_frontmatter(&fm_lines);

    let name = front
        .get("name")
        .and_then(|v| v.as_scalar())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback_name.to_string());
    if name.is_empty() {
        return Err("no skill name (frontmatter 'name' missing and no dir name)".to_string());
    }
    let description = front
        .get("description")
        .and_then(|v| v.as_scalar())
        .unwrap_or_default();

    // Body: everything after the closing fence, frontmatter stripped.
    let body = content
        .lines()
        .skip(consumed)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    Ok(ParsedSkill {
        name,
        description,
        body,
    })
}

/// A frontmatter value: scalars for `name`/`description`; lists are parsed
/// (and tolerated) for arbitrary extra keys.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FrontValue {
    Scalar(String),
    List(Vec<String>),
}

impl FrontValue {
    fn as_scalar(&self) -> Option<String> {
        match self {
            FrontValue::Scalar(s) => Some(s.clone()),
            FrontValue::List(items) => Some(items.join(", ")),
        }
    }
}

/// Tolerant YAML-ish frontmatter parser. Supports:
/// - `key: value` scalars (quotes stripped)
/// - `key: |` literal block scalars (newlines preserved)
/// - `key: >` folded block scalars (lines joined with spaces)
/// - `key: [a, b]` flow lists and `key:` + `- item` block lists
/// - unknown keys are kept but ignored by callers
/// - malformed lines are skipped, never fatal
fn parse_frontmatter(lines: &[&str]) -> HashMap<String, FrontValue> {
    let mut map: HashMap<String, FrontValue> = HashMap::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        i += 1;

        // Skip blanks and comments.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Only top-level (unindented) `key:` lines start an entry.
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some(colon) = trimmed.find(':') else {
            continue; // not a key line; tolerate
        };
        let key = trimmed[..colon].trim().to_string();
        if key.is_empty() {
            continue;
        }
        let rest = trimmed[colon + 1..].trim();

        if rest == "|"
            || rest == ">"
            || rest == "|-"
            || rest == ">-"
            || rest == "|+"
            || rest == ">+"
        {
            // Block scalar: consume following more-indented lines.
            let mut block: Vec<String> = Vec::new();
            while i < lines.len() {
                let bl = lines[i];
                if bl.trim().is_empty() {
                    block.push(String::new());
                    i += 1;
                    continue;
                }
                if bl.starts_with(char::is_whitespace) {
                    block.push(bl.trim().to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            // Trailing blank lines of the block are not content.
            while block.last().is_some_and(|l| l.is_empty()) {
                block.pop();
            }
            let value = if rest.starts_with('|') {
                block.join("\n")
            } else {
                block.join(" ")
            };
            map.insert(key, FrontValue::Scalar(value));
        } else if rest.starts_with('[') {
            // Flow list: [a, b, "c d"]
            let inner = rest.trim_start_matches('[').trim_end_matches(']');
            let items = inner
                .split(',')
                .map(|s| unquote(s.trim()))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            map.insert(key, FrontValue::List(items));
        } else if rest.is_empty() {
            // `key:` with nothing on the line: either a block list
            // (`- item` lines) or an empty value.
            let mut items: Vec<String> = Vec::new();
            let mut j = i;
            while j < lines.len() {
                let il = lines[j].trim();
                if let Some(item) = il.strip_prefix("- ") {
                    items.push(unquote(item.trim()));
                    j += 1;
                } else {
                    break;
                }
            }
            if items.is_empty() {
                map.insert(key, FrontValue::Scalar(String::new()));
            } else {
                map.insert(key, FrontValue::List(items));
                i = j;
            }
        } else {
            map.insert(key, FrontValue::Scalar(unquote(rest)));
        }
    }
    map
}

/// Strip one layer of matching single or double quotes.
fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let (first, last) = (bytes[0], bytes[bytes.len() - 1]);
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

/// Expand a leading `~` to the user's home dir. Pure path mangling, no
/// filesystem access. Returns the original path if no expansion is possible.
pub fn expand_tilde(p: &str) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a skill dir with a SKILL.md under `root/<dir>/`.
    fn write_skill(root: &Path, dir: &str, contents: &str) {
        let d = root.join(dir);
        fs::create_dir_all(&d).expect("create skill dir");
        let mut f = fs::File::create(d.join("SKILL.md")).expect("create SKILL.md");
        f.write_all(contents.as_bytes()).expect("write SKILL.md");
    }

    const BASIC: &str =
        "---\nname: alpha\ndescription: First skill\n---\n# Alpha\n\nDo the thing.\n";

    #[test]
    fn test_parse_basic_frontmatter() {
        let p = parse_skill_md(BASIC, "fallback").unwrap();
        assert_eq!(p.name, "alpha");
        assert_eq!(p.description, "First skill");
        assert_eq!(p.body, "# Alpha\n\nDo the thing.");
        assert!(!p.body.contains("---"), "body must be frontmatter-stripped");
    }

    #[test]
    fn test_parse_folded_block_scalar() {
        let md = "---\nname: folded\ndescription: >\n  A long description\n  that spans lines\n---\nBody\n";
        let p = parse_skill_md(md, "x").unwrap();
        assert_eq!(p.description, "A long description that spans lines");
        assert_eq!(p.body, "Body");
    }

    #[test]
    fn test_parse_literal_block_scalar() {
        let md = "---\nname: literal\ndescription: |\n  line one\n  line two\n---\nBody\n";
        let p = parse_skill_md(md, "x").unwrap();
        assert_eq!(p.description, "line one\nline two");
    }

    #[test]
    fn test_parse_flow_list_and_extra_keys_tolerated() {
        let md = "---\nname: extra\ndescription: Has extras\ntags: [a, b, \"c d\"]\nlicense: MIT\nallowed-tools:\n  - read_file\n  - write_file\n---\nBody\n";
        let p = parse_skill_md(md, "x").unwrap();
        assert_eq!(p.name, "extra");
        assert_eq!(p.description, "Has extras");
        assert_eq!(p.body, "Body");
    }

    #[test]
    fn test_parse_missing_description_tolerated() {
        let md = "---\nname: nodesc\n---\nBody only.\n";
        let p = parse_skill_md(md, "x").unwrap();
        assert_eq!(p.name, "nodesc");
        assert_eq!(p.description, "");
    }

    #[test]
    fn test_parse_missing_name_falls_back_to_dir_name() {
        let md = "---\ndescription: No name key\n---\nBody.\n";
        let p = parse_skill_md(md, "dir-name").unwrap();
        assert_eq!(p.name, "dir-name");
    }

    #[test]
    fn test_parse_quoted_scalars() {
        let md = "---\nname: \"quoted-name\"\ndescription: 'single quoted'\n---\nB\n";
        let p = parse_skill_md(md, "x").unwrap();
        assert_eq!(p.name, "quoted-name");
        assert_eq!(p.description, "single quoted");
    }

    #[test]
    fn test_parse_unterminated_frontmatter_errors() {
        let md = "---\nname: broken\ndescription: never closed\n";
        assert!(parse_skill_md(md, "x").is_err());
    }

    #[test]
    fn test_parse_no_frontmatter_errors() {
        assert!(parse_skill_md("# Just markdown\n", "x").is_err());
    }

    #[test]
    fn test_registry_index_and_read() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", BASIC);
        write_skill(
            tmp.path(),
            "beta",
            "---\nname: beta\ndescription: Second skill\n---\nBeta body.\n",
        );
        let reg = SkillsRegistry::new(tmp.path().to_path_buf(), vec!["beta".to_string()]);

        let index = reg.index();
        assert_eq!(index.len(), 2);
        // Sorted by name.
        assert_eq!(index[0].name, "alpha");
        assert_eq!(index[1].name, "beta");
        // Pinned flag only on the pinned skill.
        assert!(!index[0].pinned);
        assert!(index[1].pinned);
        assert_eq!(index[0].description, "First skill");

        assert_eq!(
            reg.read("alpha").as_deref(),
            Some("# Alpha\n\nDo the thing.")
        );
        assert_eq!(reg.read("beta").as_deref(), Some("Beta body."));
        assert!(reg.read("nope").is_none());
    }

    #[test]
    fn test_registry_malformed_skill_isolated() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "good", BASIC);
        // Unterminated frontmatter: must be skipped, never fatal.
        write_skill(
            tmp.path(),
            "broken",
            "---\nname: broken\nno closing fence\n",
        );
        // No frontmatter at all.
        write_skill(tmp.path(), "plain", "# no frontmatter\n");
        // A directory without SKILL.md.
        fs::create_dir_all(tmp.path().join("empty-dir")).unwrap();
        // A stray file at the top level.
        fs::write(tmp.path().join("README.md"), "not a skill").unwrap();

        let reg = SkillsRegistry::new(tmp.path().to_path_buf(), vec![]);
        let index = reg.index();
        assert_eq!(
            index.len(),
            1,
            "malformed skills must be skipped in isolation"
        );
        assert_eq!(index[0].name, "alpha");
    }

    #[test]
    fn test_registry_missing_dir_is_empty_not_error() {
        let reg = SkillsRegistry::new(PathBuf::from("/nonexistent/skills/dir"), vec![]);
        assert!(reg.index().is_empty());
        assert!(reg.read("anything").is_none());
    }

    #[test]
    fn test_registry_mtime_invalidation() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", BASIC);
        let reg = SkillsRegistry::new(tmp.path().to_path_buf(), vec![]);
        assert_eq!(reg.index()[0].description, "First skill");

        // Rewrite with different content AND different length so the
        // (mtime, len) fingerprint changes on any filesystem granularity.
        write_skill(
            tmp.path(),
            "alpha",
            "---\nname: alpha\ndescription: Updated description v2\n---\nNew body here.\n",
        );
        let index = reg.index();
        assert_eq!(index.len(), 1);
        assert_eq!(index[0].description, "Updated description v2");
        assert_eq!(reg.read("alpha").as_deref(), Some("New body here."));
    }

    #[test]
    fn test_registry_removed_skill_disappears() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "alpha", BASIC);
        write_skill(
            tmp.path(),
            "beta",
            "---\nname: beta\ndescription: Second\n---\nB\n",
        );
        let reg = SkillsRegistry::new(tmp.path().to_path_buf(), vec![]);
        assert_eq!(reg.index().len(), 2);

        fs::remove_dir_all(tmp.path().join("beta")).unwrap();
        let index = reg.index();
        assert_eq!(index.len(), 1);
        assert_eq!(index[0].name, "alpha");
        assert!(reg.read("beta").is_none());
    }

    #[test]
    fn test_expand_tilde() {
        assert_eq!(expand_tilde("/abs/path"), PathBuf::from("/abs/path"));
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expand_tilde("~/x"), home.join("x"));
            assert_eq!(expand_tilde("~"), home);
        }
    }
}
