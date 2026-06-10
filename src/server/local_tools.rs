//! Built-in local tools that execute natively without spawning MCP server processes.
//!
//! These tools provide fast, always-available file and shell operations through UMB,
//! so sub-agents can access file/shell operations without needing separate MCP servers.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::path::Path;
use tokio::process::Command as TokioCommand;

use super::router::Tool;

/// Registry of all local tool definitions
pub fn local_tool_definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "read_file".to_string(),
            description: "Read the contents of a file. Returns the file text content.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    }
                },
                "required": ["path"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "write_file".to_string(),
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "edit_file".to_string(),
            description: "Perform a search-and-replace edit in a file. Replaces the first occurrence of old_string with new_string.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences (default: false)"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "list_directory".to_string(),
            description: "List the contents of a directory, showing files and subdirectories.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list"
                    }
                },
                "required": ["path"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "glob_files".to_string(),
            description: "Find files matching a glob pattern (e.g., '**/*.rs', 'src/**/*.ts').".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files (e.g., '**/*.rs', 'src/**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search from (default: current directory)"
                    }
                },
                "required": ["pattern"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "grep_files".to_string(),
            description: "Search for a text pattern in files. Returns matching lines with file paths and line numbers.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: current directory)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter which files to search (e.g., '*.rs')"
                    }
                },
                "required": ["pattern"]
            }),
            server: "local".to_string(),
        },
        Tool {
            name: "run_command".to_string(),
            description: "Execute a shell command and return its output (stdout + stderr).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command (default: current directory)"
                    },
                    "timeout_secs": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 30)"
                    }
                },
                "required": ["command"]
            }),
            server: "local".to_string(),
        },
    ]
}

/// Execute a local tool by name
pub async fn execute_local_tool(tool_name: &str, args: Value) -> Result<Value> {
    match tool_name {
        "read_file" => execute_read_file(args).await,
        "write_file" => execute_write_file(args).await,
        "edit_file" => execute_edit_file(args).await,
        "list_directory" => execute_list_directory(args).await,
        "glob_files" => execute_glob_files(args).await,
        "grep_files" => execute_grep_files(args).await,
        "run_command" => execute_run_command(args).await,
        _ => Err(anyhow!("Unknown local tool: {}", tool_name)),
    }
}

/// Check if a tool name is a local tool
pub fn is_local_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "write_file" | "edit_file" | "list_directory"
        | "glob_files" | "grep_files" | "run_command"
    )
}

fn mcp_text_content(text: String) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": text
        }]
    })
}

fn mcp_error_content(text: String) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": text
        }],
        "isError": true
    })
}

fn get_string_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("Missing required parameter: {}", key))
}

fn get_optional_string_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

// --- Tool implementations ---

async fn execute_read_file(args: Value) -> Result<Value> {
    let path = get_string_arg(&args, "path")?;

    match tokio::fs::read_to_string(path).await {
        Ok(content) => Ok(mcp_text_content(content)),
        Err(e) => Ok(mcp_error_content(format!("Failed to read file '{}': {}", path, e))),
    }
}

async fn execute_write_file(args: Value) -> Result<Value> {
    let path = get_string_arg(&args, "path")?;
    let content = get_string_arg(&args, "content")?;

    // Create parent directories if needed
    if let Some(parent) = Path::new(path).parent() {
        if !parent.exists() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return Ok(mcp_error_content(format!(
                    "Failed to create directories for '{}': {}", path, e
                )));
            }
        }
    }

    match tokio::fs::write(path, content).await {
        Ok(()) => Ok(mcp_text_content(format!("Successfully wrote to '{}'", path))),
        Err(e) => Ok(mcp_error_content(format!("Failed to write file '{}': {}", path, e))),
    }
}

async fn execute_edit_file(args: Value) -> Result<Value> {
    let path = get_string_arg(&args, "path")?;
    let old_string = get_string_arg(&args, "old_string")?;
    let new_string = get_string_arg(&args, "new_string")?;
    let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

    let content = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => return Ok(mcp_error_content(format!("Failed to read file '{}': {}", path, e))),
    };

    if !content.contains(old_string) {
        return Ok(mcp_error_content(format!(
            "old_string not found in '{}'. Make sure the string matches exactly.", path
        )));
    }

    let new_content = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };

    match tokio::fs::write(path, &new_content).await {
        Ok(()) => {
            let count = if replace_all {
                content.matches(old_string).count()
            } else {
                1
            };
            Ok(mcp_text_content(format!(
                "Successfully edited '{}': replaced {} occurrence(s)", path, count
            )))
        }
        Err(e) => Ok(mcp_error_content(format!("Failed to write file '{}': {}", path, e))),
    }
}

async fn execute_list_directory(args: Value) -> Result<Value> {
    let path = get_string_arg(&args, "path")?;

    let mut entries = match tokio::fs::read_dir(path).await {
        Ok(rd) => rd,
        Err(e) => return Ok(mcp_error_content(format!("Failed to read directory '{}': {}", path, e))),
    };

    let mut items = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let file_type = entry.file_type().await.ok();
        let is_dir = file_type.map(|ft| ft.is_dir()).unwrap_or(false);
        let name = entry.file_name().to_string_lossy().to_string();
        let suffix = if is_dir { "/" } else { "" };
        items.push(format!("{}{}", name, suffix));
    }

    items.sort();
    Ok(mcp_text_content(items.join("\n")))
}

async fn execute_glob_files(args: Value) -> Result<Value> {
    let pattern = get_string_arg(&args, "pattern")?.to_string();
    let base_path = get_optional_string_arg(&args, "path")
        .unwrap_or(".")
        .to_string();

    // Run glob in spawn_blocking since it's sync I/O
    let result = tokio::task::spawn_blocking(move || {
        let full_pattern = if pattern.starts_with('/') || pattern.starts_with('.') {
            pattern
        } else {
            format!("{}/{}", base_path, pattern)
        };

        let mut matches = Vec::new();
        match glob::glob(&full_pattern) {
            Ok(paths) => {
                for entry in paths.flatten() {
                    matches.push(entry.display().to_string());
                }
            }
            Err(e) => return Err(anyhow!("Invalid glob pattern: {}", e)),
        }
        Ok(matches)
    })
    .await
    .map_err(|e| anyhow!("Glob task panicked: {}", e))??;

    if result.is_empty() {
        Ok(mcp_text_content("No files matched the pattern.".to_string()))
    } else {
        Ok(mcp_text_content(result.join("\n")))
    }
}

/// Unix `grep_files`: shell out to the system `grep -rn` (behaviour-identical
/// to the original — no codegen change on Unix). Windows lacks `grep`, so the
/// `#[cfg(windows)]` sibling below does an equivalent in-process recursive
/// substring search producing the SAME `path:lineno:line` output shape.
#[cfg(not(windows))]
async fn execute_grep_files(args: Value) -> Result<Value> {
    let pattern = get_string_arg(&args, "pattern")?.to_string();
    let search_path = get_optional_string_arg(&args, "path")
        .unwrap_or(".")
        .to_string();
    let file_glob = get_optional_string_arg(&args, "glob").map(|s| s.to_string());

    // Use grep/rg via command for robust searching
    let mut cmd = TokioCommand::new("grep");
    cmd.arg("-rn") // recursive, line numbers
        .arg("--color=never");

    if let Some(ref g) = file_glob {
        cmd.arg("--include").arg(g);
    }

    cmd.arg(&pattern).arg(&search_path);

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        cmd.output(),
    )
    .await
    .map_err(|_| anyhow!("grep timed out after 30 seconds"))?
    .map_err(|e| anyhow!("Failed to run grep: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if stdout.is_empty() {
        Ok(mcp_text_content(format!("No matches found for pattern '{}'", pattern)))
    } else {
        // Limit output to prevent huge responses
        let lines: Vec<&str> = stdout.lines().take(200).collect();
        let truncated = if stdout.lines().count() > 200 {
            format!("{}\n... (truncated, {} total matches)", lines.join("\n"), stdout.lines().count())
        } else {
            lines.join("\n")
        };
        Ok(mcp_text_content(truncated))
    }
}

/// Windows `grep_files`: in-process recursive plain-substring search. Windows
/// has no `grep`, and `findstr` differs in flags/output, so we implement the
/// search directly. Output shape matches the Unix `grep -rn` path used here
/// (`path:lineno:matched-line`), the same 200-line truncation, and the same
/// `glob` filename filter (via the in-tree `glob` crate's `Pattern`). This is
/// a literal-substring match (not a regex), mirroring the typical UMB usage of
/// this tool; the Unix path remains real `grep` so its richer matching is
/// unchanged.
#[cfg(windows)]
async fn execute_grep_files(args: Value) -> Result<Value> {
    let pattern = get_string_arg(&args, "pattern")?.to_string();
    let search_path = get_optional_string_arg(&args, "path")
        .unwrap_or(".")
        .to_string();
    let file_glob = get_optional_string_arg(&args, "glob").map(|s| s.to_string());

    // Compile the optional filename glob (matched against the file name only,
    // mirroring `grep --include`).
    let glob_pat = match file_glob.as_ref() {
        Some(g) => Some(
            glob::Pattern::new(g)
                .map_err(|e| anyhow!("Invalid glob '{}': {}", g, e))?,
        ),
        None => None,
    };

    // Bounded, blocking recursive walk on a worker thread so the async runtime
    // is never stalled; same 30s wall-clock budget as the Unix grep path. The
    // closure takes its own clone of `pattern` so the original is still usable
    // for the "no matches" message after the search returns.
    let needle = pattern.clone();
    let search = tokio::task::spawn_blocking(move || {
        let mut matches: Vec<String> = Vec::new();
        let mut stack: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from(&search_path)];
        while let Some(dir) = stack.pop() {
            if matches.len() >= 201 {
                break;
            }
            let read = match std::fs::read_dir(&dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for entry in read.flatten() {
                let path = entry.path();
                let ft = match entry.file_type() {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if ft.is_dir() {
                    stack.push(path);
                    continue;
                }
                if !ft.is_file() {
                    continue;
                }
                if let Some(ref gp) = glob_pat {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    if !gp.matches(&name) {
                        continue;
                    }
                }
                // Read as text; skip files that aren't valid UTF-8 (mirrors
                // grep's default text-line behaviour closely enough for the
                // tool's purpose).
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                for (idx, line) in content.lines().enumerate() {
                    if line.contains(&needle) {
                        matches.push(format!("{}:{}:{}", path.display(), idx + 1, line));
                        if matches.len() >= 201 {
                            break;
                        }
                    }
                }
            }
        }
        matches
    });

    let matches = tokio::time::timeout(std::time::Duration::from_secs(30), search)
        .await
        .map_err(|_| anyhow!("grep timed out after 30 seconds"))?
        .map_err(|e| anyhow!("Failed to run search: {}", e))?;

    if matches.is_empty() {
        Ok(mcp_text_content(format!(
            "No matches found for pattern '{}'",
            pattern
        )))
    } else {
        let total = matches.len();
        let lines: Vec<&str> = matches.iter().take(200).map(|s| s.as_str()).collect();
        let truncated = if total > 200 {
            format!(
                "{}\n... (truncated, {} total matches)",
                lines.join("\n"),
                total
            )
        } else {
            lines.join("\n")
        };
        Ok(mcp_text_content(truncated))
    }
}

async fn execute_run_command(args: Value) -> Result<Value> {
    let command = get_string_arg(&args, "command")?;
    let cwd = get_optional_string_arg(&args, "cwd");
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(30.0) as u64;

    // Shell selection is platform-specific: POSIX `sh -c` on Unix, the
    // Windows command interpreter (`cmd.exe /C`) on Windows. The command
    // string is otherwise passed through unchanged on both platforms.
    #[cfg(windows)]
    let mut cmd = {
        let mut c = TokioCommand::new("cmd");
        c.arg("/C").arg(command);
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = TokioCommand::new("sh");
        c.arg("-c").arg(command);
        c
    };

    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    )
    .await
    .map_err(|_| anyhow!("Command timed out after {} seconds", timeout_secs))?
    .map_err(|e| anyhow!("Failed to execute command: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(-1);

    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str("[stderr]\n");
        result.push_str(&stderr);
    }
    if exit_code != 0 {
        result.push_str(&format!("\n[exit code: {}]", exit_code));
    }

    if exit_code != 0 {
        Ok(mcp_error_content(result))
    } else {
        Ok(mcp_text_content(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_tool_definitions() {
        let tools = local_tool_definitions();
        assert_eq!(tools.len(), 7);
        assert!(tools.iter().all(|t| t.server == "local"));

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"list_directory"));
        assert!(names.contains(&"glob_files"));
        assert!(names.contains(&"grep_files"));
        assert!(names.contains(&"run_command"));
    }

    #[test]
    fn test_is_local_tool() {
        assert!(is_local_tool("read_file"));
        assert!(is_local_tool("write_file"));
        assert!(is_local_tool("run_command"));
        assert!(!is_local_tool("some_other_tool"));
        assert!(!is_local_tool("list_tools"));
    }

    /// Portable temp-file path (uses the OS temp dir; `/tmp` on Unix,
    /// `%TEMP%` on Windows) so these tests run green on every platform.
    fn tmp_path(name: &str) -> String {
        std::env::temp_dir().join(name).to_string_lossy().into_owned()
    }

    #[tokio::test]
    async fn test_read_nonexistent_file() {
        let result = execute_local_tool(
            "read_file",
            json!({"path": tmp_path("nonexistent_umb_test_file_12345.txt")}),
        )
        .await
        .unwrap();

        // Should return error content, not an Err
        assert!(result.get("isError").is_some());
    }

    #[tokio::test]
    async fn test_write_and_read_file() {
        let test_path = tmp_path("umb_local_tool_test.txt");
        let test_path = test_path.as_str();
        let content = "Hello from UMB local tools test";

        // Write
        let write_result = execute_local_tool(
            "write_file",
            json!({"path": test_path, "content": content}),
        )
        .await
        .unwrap();
        assert!(write_result.get("isError").is_none());

        // Read back
        let read_result = execute_local_tool(
            "read_file",
            json!({"path": test_path}),
        )
        .await
        .unwrap();

        let text = read_result["content"][0]["text"].as_str().unwrap();
        assert_eq!(text, content);

        // Cleanup
        let _ = tokio::fs::remove_file(test_path).await;
    }

    #[tokio::test]
    async fn test_edit_file() {
        let test_path = tmp_path("umb_local_tool_edit_test.txt");
        let test_path = test_path.as_str();
        let _ = tokio::fs::write(test_path, "hello world hello").await;

        // Replace first occurrence
        let result = execute_local_tool(
            "edit_file",
            json!({"path": test_path, "old_string": "hello", "new_string": "goodbye"}),
        )
        .await
        .unwrap();
        assert!(result.get("isError").is_none());

        let content = tokio::fs::read_to_string(test_path).await.unwrap();
        assert_eq!(content, "goodbye world hello");

        // Cleanup
        let _ = tokio::fs::remove_file(test_path).await;
    }

    #[tokio::test]
    async fn test_run_command() {
        let result = execute_local_tool(
            "run_command",
            json!({"command": "echo 'UMB test'"}),
        )
        .await
        .unwrap();

        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("UMB test"));
    }
}
