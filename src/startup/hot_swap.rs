//! Hot-swap handler for live configuration reloading
//!
//! Watches servers.json for changes and updates the router dynamically.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::features::{ConfigChange, HotSwapManager};
use crate::registry::ServerRegistry;
use crate::server::{ServerConfig, SseServerConfig, HttpServerConfig, ToolRouter};

/// Initialize and run hot-swap handler
///
/// Spawns a background task that watches servers.json and updates the router
/// when changes are detected.
pub fn start_hot_swap_handler(
    router: Arc<RwLock<ToolRouter>>,
    servers_path: PathBuf,
) {
    tokio::spawn(async move {
        let hot_swap_result = HotSwapManager::new(servers_path.clone());

        let mut hot_swap = match hot_swap_result {
            Ok(hs) => hs,
            Err(e) => {
                tracing::error!("[HotSwap] Failed to initialize: {}", e);
                return;
            }
        };

        let rx = match hot_swap.start_watching() {
            Ok(rx) => rx,
            Err(e) => {
                tracing::error!("[HotSwap] Failed to start watcher: {}", e);
                return;
            }
        };

        if hot_swap.is_enabled() {
            tracing::info!("[HotSwap] Watcher started and enabled");
        }

        while let Ok(event) = rx.recv() {
            let mut router = router.write().await;

            match event.change {
                ConfigChange::ServerAdded(change) => {
                    handle_server_added(&mut router, &change, &servers_path).await;
                }
                ConfigChange::ServerRemoved(name) => {
                    router.unregister_server(&name);
                    // R2: also evict the server's pooled connection so a
                    // stale old process is reaped (via the EXISTING
                    // hardened ServerConnection::Drop) instead of lingering
                    // until idle-TTL, and the next call cannot route to it.
                    // Reuses the existing pool removal verbatim.
                    router.connection_pool().remove(&name).await;
                    tracing::info!("[HotSwap] Removed server: {}", name);
                }
                ConfigChange::ServerUpdated(change) => {
                    handle_server_updated(&mut router, &change, &servers_path).await;
                }
                ConfigChange::FullReload(new_config) => {
                    handle_full_reload(&mut router, &new_config, &servers_path).await;
                }
            }

            hot_swap.log_stats();
        }
    });
}

async fn handle_server_added(
    router: &mut ToolRouter,
    change: &crate::features::ServerChange,
    servers_path: &PathBuf,
) {
    if let Some(command) = change.entry.command() {
        match ServerRegistry::load(servers_path.clone()) {
            Ok(registry) => {
                if registry.is_active(&change.name) {
                    let config = ServerConfig {
                        name: change.name.clone(),
                        command: command.to_string(),
                        args: change.entry.args().to_vec(),
                        env: change.entry.env().clone(),
                    };

                    if let Err(e) = router.register_server(config.clone()) {
                        tracing::error!("[HotSwap] Failed to register server: {}", e);
                    } else {
                        match router.discover_tools_from_server(&change.name, &config) {
                            Ok(tools) => {
                                let tool_count = tools.len();
                                for tool in tools {
                                    router.register_tool(tool);
                                }
                                tracing::info!("[HotSwap] Added server: {} ({} tools)", change.name, tool_count);
                            }
                            Err(e) => {
                                tracing::warn!("[HotSwap] Added server: {} (tool discovery failed: {})", change.name, e);
                            }
                        }
                    }
                } else {
                    tracing::debug!(
                        "[HotSwap] Server '{}' added but disabled in config.",
                        change.name
                    );
                }
            }
            Err(e) => {
                tracing::error!("[HotSwap] Failed to reload registry: {}", e);
            }
        }
    }
}

async fn handle_server_updated(
    router: &mut ToolRouter,
    change: &crate::features::ServerChange,
    servers_path: &PathBuf,
) {
    // Remove first
    router.unregister_server(&change.name);
    // R2 (primary correctness fix): evict the OLD pooled connection so
    // the re-register below builds a FRESH backend from the new config.
    // Without this, get_or_create finds the still-alive old
    // ServerConnection and silently routes to the OLD binary/env after a
    // config update. Reuses the existing pool removal → existing Drop.
    router.connection_pool().remove(&change.name).await;

    if let Some(command) = change.entry.command() {
        match ServerRegistry::load(servers_path.clone()) {
            Ok(registry) => {
                if registry.is_active(&change.name) {
                    let config = ServerConfig {
                        name: change.name.clone(),
                        command: command.to_string(),
                        args: change.entry.args().to_vec(),
                        env: change.entry.env().clone(),
                    };

                    if let Err(e) = router.register_server(config.clone()) {
                        tracing::error!("[HotSwap] Failed to update server: {}", e);
                    } else {
                        match router.discover_tools_from_server(&change.name, &config) {
                            Ok(tools) => {
                                let tool_count = tools.len();
                                for tool in tools {
                                    router.register_tool(tool);
                                }
                                tracing::info!("[HotSwap] Updated server: {} ({} tools)", change.name, tool_count);
                            }
                            Err(e) => {
                                tracing::warn!("[HotSwap] Updated server: {} (tool discovery failed: {})", change.name, e);
                            }
                        }
                    }
                } else {
                    tracing::debug!(
                        "[HotSwap] Server '{}' updated but disabled in config.",
                        change.name
                    );
                }
            }
            Err(e) => {
                tracing::error!("[HotSwap] Failed to reload registry: {}", e);
            }
        }
    }
}

async fn handle_full_reload(
    router: &mut ToolRouter,
    new_config: &crate::registry::RegistryConfig,
    _servers_path: &PathBuf,
) {
    tracing::info!("[HotSwap] Full reload triggered - reloading all servers");

    // Step 1: Clear all existing servers and tools (preserve meta-tools)
    router.clear_all_servers();
    // R2 (clear-all case): evict ALL pooled connections so a full reload
    // rebuilds every backend fresh from the new config and no stale
    // process lingers. Reuses the EXISTING hardened pool teardown
    // (`shutdown()` drains+kills+reaps every connection) verbatim — no
    // new killer, no edit to Drop/shutdown/reaper.
    router.connection_pool().shutdown().await;

    // Step 2: Re-discover tools from all enabled servers in the new config
    let mut reload_total_tools = 0;
    let mut reload_server_count = 0;

    for (name, entry) in &new_config.servers {
        // Skip disabled servers and comments
        if !entry.is_enabled() || entry.is_comment() {
            continue;
        }

        // Handle stdio servers
        if let Some(command) = entry.command() {
            let config = ServerConfig {
                name: name.clone(),
                command: command.to_string(),
                args: entry.args().to_vec(),
                env: entry.env().clone(),
            };

            if let Err(e) = router.register_server(config.clone()) {
                tracing::error!("[HotSwap] Failed to register server '{}': {}", name, e);
                continue;
            }

            match router.discover_tools_from_server(name, &config) {
                Ok(tools) => {
                    let tool_count = tools.len();
                    for tool in tools {
                        router.register_tool(tool);
                    }
                    reload_total_tools += tool_count;
                    reload_server_count += 1;
                    tracing::info!("[HotSwap] Reloaded server '{}' ({} tools)", name, tool_count);
                }
                Err(e) => {
                    tracing::warn!("[HotSwap] Failed to discover tools from '{}': {}", name, e);
                }
            }
        }
        // Handle SSE servers
        else if entry.is_sse() {
            if let Some(sse_url) = entry.sse_url() {
                let sse_config = SseServerConfig {
                    name: name.clone(),
                    url: sse_url.to_string(),
                    env: entry.env().clone(),
                };

                if let Err(e) = router.register_sse_server(sse_config) {
                    tracing::error!("[HotSwap] Failed to register SSE server '{}': {}", name, e);
                    continue;
                }

                match router.discover_tools_from_sse_server(name, sse_url) {
                    Ok(tools) => {
                        let tool_count = tools.len();
                        for tool in tools {
                            router.register_tool(tool);
                        }
                        reload_total_tools += tool_count;
                        reload_server_count += 1;
                        tracing::info!("[HotSwap] Reloaded SSE server '{}' ({} tools)", name, tool_count);
                    }
                    Err(e) => {
                        tracing::warn!("[HotSwap] Failed to discover tools from SSE '{}': {}", name, e);
                    }
                }
            }
        }
        // Handle HTTP servers
        else if entry.is_http() {
            if let Some(http_url) = entry.http_url() {
                let http_config = HttpServerConfig {
                    name: name.clone(),
                    url: http_url.to_string(),
                    env: entry.env().clone(),
                };

                if let Err(e) = router.register_http_server(http_config) {
                    tracing::error!("[HotSwap] Failed to register HTTP server '{}': {}", name, e);
                    continue;
                }

                match router.discover_tools_from_http_server(name, http_url) {
                    Ok(tools) => {
                        let tool_count = tools.len();
                        for tool in tools {
                            router.register_tool(tool);
                        }
                        reload_total_tools += tool_count;
                        reload_server_count += 1;
                        tracing::info!("[HotSwap] Reloaded HTTP server '{}' ({} tools)", name, tool_count);
                    }
                    Err(e) => {
                        tracing::warn!("[HotSwap] Failed to discover tools from HTTP '{}': {}", name, e);
                    }
                }
            }
        }
    }

    tracing::info!(
        "[HotSwap] Full reload complete: {} servers, {} tools",
        reload_server_count, reload_total_tools
    );
}
