//! MCP server initialization and request handling
//!
//! Contains the core server startup logic and JSON-RPC request handler.

use anyhow::Result;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::config::UmbConfig;
use crate::registry::ServerRegistry;
use crate::server::{
    error_response, local_tools, success_response, JsonRpcRequest, JsonRpcResponse, McpServer, Tool, ToolRouter,
};

use super::discovery::run_background_discovery;
use super::hot_swap::start_hot_swap_handler;

/// Register the 3 built-in meta-tools and local tools
pub fn register_meta_tools(router: &mut ToolRouter) {
    // Register native local tools (file/shell operations)
    for tool in local_tools::local_tool_definitions() {
        router.register_tool(tool);
    }

    router.register_tool(Tool {
        name: "list_tools".to_string(),
        description: "Discover tools (name + short description only). Pass a query (e.g. 'file ops', 'git') for relevant results; no query returns a server-grouped map. Then call get_tool_info(tool) for the full schema before route_mcp_call.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Describe what you need (e.g., 'send emails', 'manage files'). Highly recommended for accurate results."},
                "limit": {"type": "number", "description": "Max results to return (default: 10)"}
            }
        }),
        server: "builtin".to_string(),
    });

    // P1/A1 — on-demand single-tool full schema. The two-tier counterpart
    // to the now-slim `list_tools`: the agent lists names+short-desc
    // cheaply, then fetches the FULL description + inputSchema for exactly
    // the tool it intends to call (so no agent is ever left blind, while
    // the bulk surface stays small).
    router.register_tool(Tool {
        name: "get_tool_info".to_string(),
        description: "Get one tool's full description + argument inputSchema by name (use after list_tools, before route_mcp_call).".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "Exact tool name (from list_tools) to fetch full details for"}
            },
            "required": ["tool"]
        }),
        server: "builtin".to_string(),
    });

    router.register_tool(Tool {
        name: "list_mcps".to_string(),
        description: "List all MCP servers with their tools grouped by server.".to_string(),
        input_schema: json!({"type": "object", "properties": {}}),
        server: "builtin".to_string(),
    });

    router.register_tool(Tool {
        name: "route_mcp_call".to_string(),
        description: "Execute any MCP tool. Use list_tools first to find the tool. If two servers export the same tool name, pass `server` to disambiguate (UMB returns a clear ambiguous-tool error listing candidate servers otherwise).".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "Tool name to execute"},
                "args": {"type": "object", "description": "Tool arguments"},
                "server": {"type": "string", "description": "Optional: MCP server name when multiple servers export the same tool (see list_mcps / get_tool_info)"}
            },
            "required": ["tool"]
        }),
        server: "builtin".to_string(),
    });
}

/// Start the MCP server in silent mode (for coding agents)
///
/// All output goes to stderr via tracing - NO stdout pollution.
/// The MCP server starts IMMEDIATELY; tool discovery runs in background.
pub async fn start_server_silent(
    shutdown_token: CancellationToken,
    search_threshold: f32,
    search_limit: usize,
) -> Result<()> {
    tracing::info!("Starting Universal MCP Bridge (stdio mode)...");

    // Load UMB configuration
    let umb_config = UmbConfig::load().unwrap_or_else(|e| {
        tracing::warn!("[Server] Failed to load config: {}, using defaults", e);
        UmbConfig::default()
    });

    // Load servers config path
    let servers_path = umb_config.get_servers_path();
    tracing::debug!("[Server] Servers path: {:?}", servers_path);

    // Load server registry (all enabled servers are active)
    let server_registry = ServerRegistry::load(servers_path.clone())?;
    tracing::info!(
        "[Server] Loaded {} servers, {} active",
        server_registry.total_count(),
        server_registry.active_count()
    );

    // Load the tool-dictionary overlay (compile-time fallback +
    // in-repo + config paths + user overlay). Pure data load; failure
    // here is impossible (loader is infallible — bad files are warned
    // and skipped).
    let dict = crate::server::tool_dictionary::ToolDictionary::load(
        &umb_config.general.tool_dictionary_paths,
        Some(std::path::Path::new(&umb_config.general.tool_dictionary_user_dir)),
    );
    let dict_mode = crate::server::tool_dictionary::ShortMode::from_str_or_auto(
        &umb_config.general.short_definitions,
    );
    tracing::info!(
        "[Server] Tool-dictionary: {} entries, mode={:?}",
        dict.len(),
        dict_mode,
    );

    // Initialize tool router - NO TOOLS YET, just meta-tools
    let mut router = ToolRouter::new()
        .with_search_threshold(search_threshold)
        .with_search_limit(search_limit)
        .with_tool_dictionary(dict, dict_mode);
    tracing::info!(
        "[Server] Search config: threshold={:.2}, limit={}",
        search_threshold,
        search_limit
    );
    register_meta_tools(&mut router);

    // Wrap router for shared access
    let router = Arc::new(RwLock::new(router));

    // Task #25 (fix/idle-eviction-runtime) — spawn the periodic pooled-
    // connection idle-eviction sweeper. TTL from config (default 600s /
    // 10 min, ENABLED by default; 0 disables — `spawn_idle_sweeper`
    // no-ops in that case). Eviction reuses the EXISTING hardened
    // `ServerConnection::Drop` teardown (pgid kill + untrack +
    // subreaper/1e/probe-allowlist paths, unchanged) — no parallel killer.
    //
    // The sweep cadence is now TTL-SCALED inside `spawn_idle_sweeper`
    // (clamp(1s, ttl/4, 60s)) instead of a hardcoded 45s. The old fixed
    // 45s made short-TTL configs non-functional at runtime: eviction
    // latency is bounded by the sweep period, so ttl=8s with ~12s idle
    // windows never evicted (real-VM E2E). non-blocking per-conn
    // `try_lock` so a busy connection is simply skipped, never blocking
    // the sweeper or a live call.
    {
        let pool = router.read().await.connection_pool().clone();
        pool.spawn_idle_sweeper(umb_config.general.pool_idle_ttl_secs);
    }

    // Spawn BACKGROUND task for tool discovery (don't block MCP server start)
    let router_for_discovery = router.clone();
    let servers_path_for_discovery = servers_path.clone();
    let umb_config_clone = umb_config.clone();

    tokio::spawn(async move {
        run_background_discovery(
            router_for_discovery,
            servers_path_for_discovery,
            umb_config_clone,
        )
        .await;
    });

    // Initialize hot-swap in background
    start_hot_swap_handler(router.clone(), servers_path.clone());

    // Create and run MCP server IMMEDIATELY - don't wait for discovery
    let mcp_server = McpServer::new();
    tracing::info!("[Server] MCP server ready on stdio (discovery running in background)");

    mcp_server
        .run_with_shutdown(
            move |request: JsonRpcRequest| {
                let router_clone = Arc::clone(&router);
                async move { handle_request(&router_clone, request).await }
            },
            shutdown_token,
        )
        .await?;

    Ok(())
}

/// Handle incoming JSON-RPC requests
pub async fn handle_request(
    router_arc: &Arc<RwLock<ToolRouter>>,
    request: JsonRpcRequest,
) -> Result<JsonRpcResponse> {
    match request.method.as_str() {
        // MCP Protocol: Initialize handshake
        "initialize" => Ok(success_response(
            request.id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "universal-mcp-bridge",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )),

        // MCP Protocol: Initialized notification
        "notifications/initialized" | "initialized" => {
            Ok(success_response(request.id, json!({})))
        }

        // MCP Protocol: Ping
        "ping" => Ok(success_response(request.id, json!({}))),

        "tools/list" => {
            // MCP protocol tools/list: Return ONLY meta-tools (builtin)
            let router = router_arc.read().await;
            let all_tools = router.list_tools(None, false, 1000);

            // Filter to builtin meta-tools and local tools
            let meta_tools: Vec<_> = all_tools
                .iter()
                .filter(|t| t.server == "builtin" || t.server == "local")
                .collect();

            tracing::info!(
                "[MCP] tools/list called - returning {} meta+local tools (hiding {} underlying tools)",
                meta_tools.len(),
                all_tools.len() - meta_tools.len()
            );

            let tools_json: Vec<_> = meta_tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect();

            Ok(success_response(request.id, json!({ "tools": tools_json })))
        }

        "list_tools" => {
            // list_tools meta-tool called directly
            let router = router_arc.read().await;
            let query = request
                .params
                .as_ref()
                .and_then(|p| p.get("query"))
                .and_then(|q| q.as_str())
                .map(String::from);

            let limit = request
                .params
                .as_ref()
                .and_then(|p| p.get("limit"))
                .and_then(|l| l.as_u64())
                .map(|l| l as usize)
                .unwrap_or_else(|| router.default_limit());

            let use_semantic = router.has_semantic_search() && query.is_some();
            let query_for_log = query.clone();
            let tools = router.list_tools(query, use_semantic, limit);

            tracing::info!(
                "[MCP] list_tools meta-tool called - returning {} tools (query: {:?}, limit: {})",
                tools.len(),
                query_for_log,
                limit
            );

            // P1 two-tier: bulk discovery surface is name + SHORT
            // description ONLY — NO inputSchema (the full schema is
            // fetched on demand for the one chosen tool via
            // get_tool_info). Cuts the dominant per-discovery token sink.
            //
            // Tool-dictionary overlay: when an entry applies for this
            // (server, name), render the curated short_description
            // VERBATIM (no further `short_description()` first-sentence
            // truncation — the curator's hand-curated form is already
            // terse). When no entry applies, fall through to the
            // existing `short_description(&desc)` derivation. Pure
            // overlay-at-read — `self.tools` is unmodified.
            let tools_json: Vec<_> = tools
                .iter()
                .map(|t| {
                    let (desc, src) = router.apply_dict_overlay(&t.server, &t.name, &t.description);
                    let final_desc = if src == "dict" {
                        desc.to_string()
                    } else {
                        crate::server::router::short_description(desc)
                    };
                    json!({
                        "name": t.name,
                        "description": final_desc,
                    })
                })
                .collect();

            Ok(success_response(request.id, json!({ "tools": tools_json })))
        }

        "get_tool_info" => {
            // P1/A1 — on-demand single-tool full detail (the two-tier
            // counterpart of the slim list_tools). Read-only over the
            // in-memory tool map; brief read guard, no backend await.
            let router = router_arc.read().await;
            let name = request
                .params
                .as_ref()
                .and_then(|p| p.get("tool").or_else(|| p.get("name")))
                .and_then(|v| v.as_str());
            match name {
                Some(n) => match router.get_tool_info(n) {
                    Some(info) => Ok(success_response(request.id, info)),
                    None => Ok(error_response(
                        request.id,
                        -32000,
                        format!(
                            "Tool not found: {}. Use list_tools({{\"query\": \"<what you need>\"}}) to discover available tool names.",
                            n
                        ),
                    )),
                },
                None => Ok(error_response(
                    request.id,
                    -32602,
                    "get_tool_info requires a 'tool' (tool name) parameter".to_string(),
                )),
            }
        }

        "list_mcps" => {
            let router = router_arc.read().await;
            let servers = router.list_servers();
            let servers_json: Vec<_> = servers
                .iter()
                .map(|(name, tools)| {
                    json!({
                        "name": name,
                        "tools": tools,
                        "tool_count": tools.len(),
                    })
                })
                .collect();

            Ok(success_response(request.id, json!({ "servers": servers_json })))
        }

        "tools/call" | "route_mcp_call" => {
            let params = request
                .params
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Missing params"))?;

            let tool_name = params
                .get("tool")
                .or_else(|| params.get("name"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

            let args = params
                .get("args")
                .or_else(|| params.get("arguments"))
                .cloned()
                .unwrap_or(json!({}));

            // Handle built-in tools
            if tool_name == "list_tools" {
                let router = router_arc.read().await;
                let query = args.get("query").and_then(|q| q.as_str()).map(String::from);
                let limit = args
                    .get("limit")
                    .and_then(|l| l.as_u64())
                    .map(|l| l as usize)
                    .unwrap_or_else(|| router.default_limit());
                let use_semantic = router.has_semantic_search() && query.is_some();
                let tools = router.list_tools(query, use_semantic, limit);

                // A1 fix: the wrapped (via route_mcp_call) list_tools was
                // NAMES-ONLY — the agent could not see descriptions, could
                // not judge which tool to pick, and had no path to the
                // schema. Now it carries name + SHORT description (same
                // slim two-tier shape as the direct list_tools path); the
                // agent then calls get_tool_info(name) for the full
                // schema before route_mcp_call. No agent left blind, bulk
                // surface still small. Tool-dictionary overlay applied
                // identically to the direct path (above).
                let tools_json: Vec<_> = tools
                    .iter()
                    .map(|t| {
                        let (desc, src) =
                            router.apply_dict_overlay(&t.server, &t.name, &t.description);
                        let final_desc = if src == "dict" {
                            desc.to_string()
                        } else {
                            crate::server::router::short_description(desc)
                        };
                        json!({
                            "name": t.name,
                            "description": final_desc,
                        })
                    })
                    .collect();
                return Ok(success_response(
                    request.id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&json!({
                                "tools": tools_json,
                                "total_count": tools.len()
                            })).unwrap_or_else(|_| "[]".to_string())
                        }]
                    }),
                ));
            }

            if tool_name == "get_tool_info" {
                // A1 — on-demand full schema also reachable via
                // route_mcp_call (so an agent confined to the 3 meta-tools
                // surface can still fetch a schema before calling).
                let router = router_arc.read().await;
                let inner = args
                    .get("tool")
                    .or_else(|| args.get("name"))
                    .and_then(|v| v.as_str());
                let body = match inner {
                    Some(n) => match router.get_tool_info(n) {
                        Some(info) => info,
                        None => json!({
                            "error": format!(
                                "Tool not found: {}. Use list_tools({{\"query\": \"<what you need>\"}}) to discover tool names.",
                                n
                            )
                        }),
                    },
                    None => json!({"error": "get_tool_info requires a 'tool' (tool name) parameter"}),
                };
                return Ok(success_response(
                    request.id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&body)
                                .unwrap_or_else(|_| "{}".to_string())
                        }]
                    }),
                ));
            }

            if tool_name == "list_mcps" {
                let router = router_arc.read().await;
                let servers = router.list_servers();
                let servers_json: Vec<_> = servers
                    .iter()
                    .map(|(name, tools)| {
                        json!({
                            "name": name,
                            "tools": tools,
                            "tool_count": tools.len()
                        })
                    })
                    .collect();

                return Ok(success_response(
                    request.id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string_pretty(&json!({
                                "servers": servers_json,
                                "total_servers": servers.len()
                            })).unwrap_or_else(|_| "{}".to_string())
                        }]
                    }),
                ));
            }

            // Handle route_mcp_call: extract inner tool name + optional server hint
            if tool_name == "route_mcp_call" {
                let inner_tool = args
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("route_mcp_call requires 'tool' parameter"))?;

                let inner_args = args.get("args").cloned().unwrap_or(json!({}));
                // (server,name)-keyed registry: optional `server` hint for
                // disambiguation when two servers export the same tool
                // name. Absent → unambiguous-name routing as today; if
                // the name is collided AND no hint is given, the router
                // returns a clear ambiguous-tool error listing candidates.
                let inner_server = args
                    .get("server")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                tracing::debug!(
                    "[Server] route_mcp_call routing to tool '{}' (server hint: {:?}) with args {:?}",
                    inner_tool,
                    inner_server,
                    inner_args
                );

                // R1 portal-stall fix: brief read guard for resolution,
                // dropped BEFORE the backend await. The (server,name)
                // resolution honors the optional hint synchronously.
                let resolved = {
                    let router = router_arc.read().await;
                    router.resolve_call_target_with_hint(inner_tool, inner_server.as_deref())
                    // guard dropped here at end of block — BEFORE dispatch
                };
                let result = match resolved {
                    Ok(target) => target.dispatch(inner_args).await,
                    Err(e) => Err(e),
                };

                return match result {
                    Ok(tool_result) => Ok(success_response(request.id, tool_result)),
                    Err(e) => Ok(error_response(
                        request.id,
                        -32000,
                        format!("Tool call failed: {}", e),
                    )),
                };
            }

            // Call external tool via the router. R1: same brief-resolve →
            // drop-guard → lock-free-dispatch pattern as route_mcp_call.
            let tool_name_owned = tool_name.to_string();
            let resolved = {
                let router = router_arc.read().await;
                router.resolve_call_target(&tool_name_owned)
                // guard dropped here — BEFORE the backend await
            };
            let result = match resolved {
                Ok(target) => target.dispatch(args).await,
                Err(e) => Err(e),
            };

            match result {
                Ok(tool_result) => Ok(success_response(request.id, tool_result)),
                Err(e) => Ok(error_response(
                    request.id,
                    -32000,
                    format!("Tool call failed: {}", e),
                )),
            }
        }

        _ => Ok(error_response(
            request.id,
            -32601,
            format!("Method not found: {}", request.method),
        )),
    }
}
