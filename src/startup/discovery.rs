//! Background tool discovery
//!
//! Discovers tools from MCP servers asynchronously to avoid blocking startup.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::UmbConfig;
#[cfg(feature = "embed-onnx")]
use crate::features::{SemanticSearchConfig, SemanticSearchProvider};
use crate::registry::ServerRegistry;
use crate::server::{ServerConfig, SseServerConfig, HttpServerConfig, ToolRouter};

/// Run background tool discovery
///
/// This is spawned as a background task so the MCP server can start immediately.
/// Tools are discovered and registered as they become available.
pub async fn run_background_discovery(
    router: Arc<RwLock<ToolRouter>>,
    servers_path: PathBuf,
    umb_config: UmbConfig,
) {
    tracing::info!("[Discovery] Starting background tool discovery...");

    // Initialize semantic search in background (can happen in parallel with discovery).
    // The open release defaults to the zero-dependency "keyword" backend; the
    // (heavier) ONNX embedding path is only initialized when the user explicitly
    // selects a non-keyword backend in config AND the binary was built with the
    // `embed-onnx` feature. Without that feature the build is keyword-only and a
    // non-keyword config gracefully degrades to keyword search (never panics). (§7.1)
    if umb_config.semantic_search.backend != "keyword" {
        #[cfg(feature = "embed-onnx")]
        {
            let ss_config = SemanticSearchConfig {
                dimension: umb_config.get_embedding_dimension(),
                custom_model_path: umb_config.semantic_search.custom_model_path.as_ref().map(PathBuf::from),
                custom_tokenizer_path: umb_config.semantic_search.custom_tokenizer_path.as_ref().map(PathBuf::from),
                huggingface_repo: umb_config.semantic_search.huggingface_repo.clone(),
                huggingface_model_file: umb_config.semantic_search.huggingface_model_file.clone(),
            };

            match SemanticSearchProvider::from_config(ss_config) {
                Ok(provider) => {
                    tracing::debug!("[Discovery] Semantic search initialized");
                    router.write().await.set_semantic_search(provider);
                }
                Err(e) => {
                    tracing::warn!("[Discovery] Semantic search unavailable: {}", e);
                }
            }
        }

        #[cfg(not(feature = "embed-onnx"))]
        {
            tracing::warn!(
                "[Discovery] semantic_search.backend = '{}' but this build has no \
                 embedding support (built without the `embed-onnx` feature). \
                 Falling back to keyword search.",
                umb_config.semantic_search.backend
            );
        }
    }

    // Reload server registry for discovery
    let server_registry = match ServerRegistry::load(servers_path.clone()) {
        Ok(reg) => reg,
        Err(e) => {
            tracing::error!("[Discovery] Failed to reload registry: {}", e);
            return;
        }
    };

    // Discover tools from active servers
    // IMPORTANT: Discover stdio servers FIRST (fast ~1s each), then SSE (slow ~30s due to embedding)
    tracing::info!(
        "[Discovery] Discovering tools from {} servers (stdio first, then SSE)...",
        server_registry.active_count()
    );

    let mut total_tools = 0;
    let mut stdio_count = 0;
    let mut sse_count = 0;
    // (born-clean) removed write-only `registered_server_count` — it was
    // incremented but NEVER read (genuinely dead; behaviour-neutral).

    // PASS 1: Stdio servers first (fast - typically 1-2 seconds each)
    // Lock strategy: brief write lock for registration, read lock for discovery,
    // brief write lock for tool registration. This prevents blocking client
    // requests during slow I/O (each server discovery spawns a child process).
    for (name, server) in &server_registry.all_servers {
        // Skip non-stdio servers (handled in PASS 2 and 3)
        if server.is_sse() || server.is_http() || server.is_comment() || !server.is_enabled() {
            continue;
        }

        if let Some(command) = server.command() {
            let server_config = ServerConfig {
                name: name.clone(),
                command: command.to_string(),
                args: server.args().to_vec(),
                env: server.env().clone(),
            };

            // Brief write lock: register the server config
            {
                let mut router = router.write().await;
                if let Err(e) = router.register_server(server_config.clone()) {
                    tracing::warn!("[Discovery] Failed to register {}: {}", name, e);
                    continue;
                }
            }
            // Write lock released - clients can now read the router

            // Discover tools using spawn_blocking to avoid starving the tokio runtime.
            // We clone the server_config and server_name so no lock is held during blocking I/O.
            let server_config_clone = server_config.clone();
            let name_clone = name.clone();
            // Defect #2(b): bound EACH server's discovery so one slow/noisy
            // backend (e.g. one whose blocking `read_line` never returns
            // because it emitted junk-without-newline then hung) cannot
            // stall the WHOLE serial discovery loop. The inner
            // `read_jsonrpc_response` has its own ~10s read timeout, but it
            // only checks BETWEEN reads — a truly blocked `read_line`
            // ignores it. This outer async timeout is the hard ceiling: on
            // expiry we abandon this server (the detached spawn_blocking
            // task + its child are cleaned up by the child's own
            // kill_on_drop / the pre-exit reaper — NOT touched here) and
            // move on. 20s comfortably covers a healthy stdio server
            // (typically 1–2s) while bounding a pathological one.
            const PER_SERVER_DISCOVERY_TIMEOUT: std::time::Duration =
                std::time::Duration::from_secs(20);
            let discovery_fut = tokio::task::spawn_blocking(move || {
                // Create a temporary router just for discovery (no lock needed)
                let temp_router = crate::server::ToolRouter::new();
                temp_router.discover_tools_from_server(&name_clone, &server_config_clone)
            });
            let discovered = match tokio::time::timeout(
                PER_SERVER_DISCOVERY_TIMEOUT,
                discovery_fut,
            )
            .await
            {
                Ok(join) => join
                    .unwrap_or_else(|e| Err(anyhow::anyhow!("Discovery task panicked: {}", e))),
                Err(_elapsed) => Err(anyhow::anyhow!(
                    "discovery for '{}' exceeded {:?} (slow/noisy server) — \
                     skipped so it cannot stall the serial discovery loop",
                    name,
                    PER_SERVER_DISCOVERY_TIMEOUT
                )),
            };

            match discovered {
                Ok(tools) => {
                    let count = tools.len();
                    // Brief write lock: register discovered tools
                    {
                        let mut router = router.write().await;
                        for tool in tools {
                            router.register_tool(tool);
                        }
                    }
                    total_tools += count;
                    stdio_count += 1;
                    tracing::info!("[Discovery] {} → {} tools", name, count);
                }
                Err(e) => tracing::warn!("[Discovery] {} failed: {}", name, e),
            }
        }
    }
    tracing::info!(
        "[Discovery] Stdio discovery complete: {} servers, {} tools",
        stdio_count, total_tools
    );

    // PASS 2: SSE servers (slow - embedding 100+ tools can take 30+ seconds)
    for (name, server) in &server_registry.all_servers {
        if !server.is_sse() || !server.is_enabled() {
            continue;
        }

        if let Some(sse_url) = server.sse_url() {
            let sse_config = SseServerConfig {
                name: name.clone(),
                url: sse_url.to_string(),
                env: server.env().clone(),
            };

            // Brief write lock: register SSE server
            {
                let mut router = router.write().await;
                if let Err(e) = router.register_sse_server(sse_config) {
                    tracing::warn!("[Discovery] Failed to register SSE {}: {}", name, e);
                    continue;
                }
            }

            // Discover SSE tools using spawn_blocking (SSE can take 30+ seconds)
            let name_clone = name.clone();
            let sse_url_clone = sse_url.to_string();
            let discovered = tokio::task::spawn_blocking(move || {
                let temp_router = crate::server::ToolRouter::new();
                temp_router.discover_tools_from_sse_server(&name_clone, &sse_url_clone)
            }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("SSE discovery task panicked: {}", e)));

            match discovered {
                Ok(tools) => {
                    let count = tools.len();
                    // Brief write lock: register discovered tools
                    {
                        let mut router = router.write().await;
                        for tool in tools {
                            router.register_tool(tool);
                        }
                    }
                    total_tools += count;
                    sse_count += 1;
                    tracing::info!("[Discovery] {} (SSE) → {} tools", name, count);
                }
                Err(e) => tracing::warn!("[Discovery] {} (SSE) failed: {}", name, e),
            }
        }
    }

    // PASS 3: HTTP servers (if any)
    for (name, server) in &server_registry.all_servers {
        if !server.is_http() || !server.is_enabled() {
            continue;
        }

        if let Some(http_url) = server.http_url() {
            let http_config = HttpServerConfig {
                name: name.clone(),
                url: http_url.to_string(),
                env: server.env().clone(),
            };

            // Brief write lock: register HTTP server
            {
                let mut router = router.write().await;
                if let Err(e) = router.register_http_server(http_config) {
                    tracing::warn!("[Discovery] Failed to register HTTP {}: {}", name, e);
                    continue;
                }
            }

            // Discover HTTP tools using spawn_blocking
            let name_clone = name.clone();
            let http_url_clone = http_url.to_string();
            let discovered = tokio::task::spawn_blocking(move || {
                let temp_router = crate::server::ToolRouter::new();
                temp_router.discover_tools_from_http_server(&name_clone, &http_url_clone)
            }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("HTTP discovery task panicked: {}", e)));

            match discovered {
                Ok(tools) => {
                    let count = tools.len();
                    // Brief write lock: register discovered tools
                    {
                        let mut router = router.write().await;
                        for tool in tools {
                            router.register_tool(tool);
                        }
                    }
                    total_tools += count;
                    tracing::info!("[Discovery] {} (HTTP) → {} tools", name, count);
                }
                Err(e) => tracing::warn!("[Discovery] {} (HTTP) failed: {}", name, e),
            }
        }
    }

    tracing::info!(
        "[Discovery] Complete: {} total tools ({} stdio, {} SSE)",
        total_tools, stdio_count, sse_count
    );
}
