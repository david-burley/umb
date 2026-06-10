//! CLI command handlers
//!
//! Handles --list-servers, --doctor, and --doctor-tools commands.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::UmbConfig;
use crate::registry::ServerRegistry;
use crate::server::ToolRouter;
use crate::server::tool_dictionary::{ShortMode, ToolDictionary};

/// Handle --doctor command.
///
/// This is the daemonless 1-process:1-client build: there is NO daemon/proxy
/// multi-client layer, so the legacy daemon-scanning doctor (orphan-daemon
/// detection, singleton registry reconciliation, socket health probes) no
/// longer has any subject to act on. Every pooled backend MCP child is owned
/// by exactly one short-lived `umb` stdio process and is reaped by that
/// process's own teardown chokepoint (parent-death watchdog / stdin EOF /
/// inactivity / SIGTERM / SIGINT → `reap_tracked_children()` with the /proc
/// descendant-snapshot + ident-revalidation). There is therefore nothing for
/// a cross-process "doctor" to clean.
///
/// `--json` still emits a minimal well-formed object (`daemons: []`,
/// `lock_holder_pid: null`) so any existing JSON consumers keep parsing; the
/// human path prints a one-line explanation. Always exits 0.
pub async fn handle_doctor(_clean: bool, json: bool, _yes: bool) -> Result<()> {
    if json {
        // Minimal, stable, well-formed shape for machine consumers.
        println!(
            "{}",
            serde_json::json!({
                "daemons": [],
                "lock_holder_pid": serde_json::Value::Null,
                "note": "daemonless build — no daemon/proxy layer to scan"
            })
        );
    } else {
        println!("umb doctor: nothing to clean (daemonless build).");
        println!(
            "This build is strictly 1 process : 1 client; pooled backend MCP \
             children are reaped by each umb process's own teardown."
        );
    }
    Ok(())
}

/// Handle --doctor-tools command.
///
/// Dumps every registered tool as JSON `[{name, server, description, source}]`
/// where `source` is `"dict"` if the tool-dictionary overrode the live
/// description, `"server"` otherwise. Useful for catching stale dict
/// entries and for measuring how much of the dict actually fires against
/// the operator's particular server set.
///
/// Synchronously runs discovery (one pass) before dumping — this means a
/// full set of pooled MCP children is spawned, the dump is rendered, and
/// the children are torn down by the same chokepoint as a normal stdio
/// exit. Bounded by discovery's own internal timeouts.
pub async fn handle_doctor_tools() -> Result<()> {
    // Load UMB config + the dictionary first so the dump shows the dict
    // already wired up; failure to load anything is non-fatal (a clean
    // empty section appears in the output).
    let umb_config = UmbConfig::load().unwrap_or_else(|e| {
        tracing::warn!("[doctor-tools] Failed to load config: {}, using defaults", e);
        UmbConfig::default()
    });
    let dict = ToolDictionary::load(
        &umb_config.general.tool_dictionary_paths,
        Some(std::path::Path::new(&umb_config.general.tool_dictionary_user_dir)),
    );
    let mode = ShortMode::from_str_or_auto(&umb_config.general.short_definitions);

    // Build a router with meta-tools + local tools + the dictionary, then
    // run discovery once to register external server tools (so the dump
    // shows the FULL set the operator's agent would see). Children
    // spawned by discovery are reaped by `reap_tracked_children` after we
    // finish dumping; we DON'T call `install_child_subreaper` here
    // because the doctor path is a short-lived single-pass dump.
    let mut router = ToolRouter::new().with_tool_dictionary(dict, mode);
    // Register meta-tools + local tools just like the normal startup path.
    crate::startup::server::register_meta_tools(&mut router);
    let router = Arc::new(RwLock::new(router));

    // Run discovery synchronously (await directly — not a spawned bg task).
    let servers_path = umb_config.get_servers_path();
    crate::startup::discovery::run_background_discovery(
        router.clone(),
        servers_path,
        umb_config.clone(),
    )
    .await;

    // Dump. Apply the dictionary overlay per tool via the router helper so
    // the rendered description matches what `get_tool_info` would emit.
    let r = router.read().await;
    let mut entries: Vec<serde_json::Value> = Vec::new();
    for (server, name, live_desc) in r.iter_tools_for_doctor() {
        let (desc, source) = r.apply_dict_overlay(server, name, live_desc);
        entries.push(serde_json::json!({
            "name": name,
            "server": server,
            "description": desc,
            "source": source,
        }));
    }
    let dict_count = entries.iter().filter(|e| e["source"] == "dict").count();
    let server_count = entries.len() - dict_count;
    let summary = serde_json::json!({
        "tools": entries,
        "summary": {
            "total": dict_count + server_count,
            "dict_applied": dict_count,
            "server_only": server_count,
            "short_definitions_mode": umb_config.general.short_definitions,
        }
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());

    // Drop the router read guard before tearing down pooled children
    // (the existing teardown chokepoint reaps them once the process exits).
    drop(r);
    crate::server::router::reap_tracked_children();
    Ok(())
}

/// Handle --list-servers command: display configured servers and exit.
pub async fn handle_list_servers() -> Result<()> {
    // Load UMB config to get servers path
    let umb_config = UmbConfig::load().unwrap_or_else(|e| {
        tracing::warn!("[Commands] Failed to load config: {}, using defaults", e);
        UmbConfig::default()
    });
    let servers_path = umb_config.get_servers_path();

    let server_registry = ServerRegistry::load(servers_path.clone())?;

    if server_registry.active_count() == 0 {
        println!("\nNo servers configured yet.");
        println!("\nTo add servers, edit {:?}", servers_path);
        println!("Use standard MCP server format (same as .mcp.json or claude_desktop_config.json).\n");
        println!("Example:");
        println!("{{");
        println!("  \"servers\": {{");
        println!("    \"filesystem\": {{");
        println!("      \"command\": \"npx\",");
        println!("      \"args\": [\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/home\"]");
        println!("    }}");
        println!("  }}");
        println!("}}\n");
        return Ok(());
    }

    println!(
        "\nConfigured servers: {}",
        server_registry.active_count()
    );
    println!("========================================\n");

    for name in &server_registry.active_servers {
        if let Some(server) = server_registry.all_servers.get(name) {
            if server.is_sse() {
                println!("  [✓] {} (SSE)", name);
                if let Some(url) = server.sse_url() {
                    println!("      URL: {}", url);
                }
            } else if server.is_http() {
                println!("  [✓] {} (HTTP)", name);
                if let Some(url) = server.http_url() {
                    println!("      URL: {}", url);
                }
            } else {
                println!("  [✓] {}", name);
                if let Some(cmd) = server.command() {
                    println!("      Command: {}", cmd);
                }
                if !server.args().is_empty() {
                    println!("      Args: {:?}", server.args());
                }
            }
        }
    }
    println!();
    println!("Note: Hot-swap is automatic - edit servers.json while umb is running");
    println!("      to see changes live.");

    Ok(())
}
