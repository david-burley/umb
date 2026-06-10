//! Hot-Swap Module
//!
//! Enables runtime reloading of MCP server configurations without restart.
//! Uses file watching to detect changes to servers.json.

use anyhow::{anyhow, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::registry::{RegistryConfig, ServerEntry};

/// A server change with its name and configuration
#[derive(Debug, Clone)]
pub struct ServerChange {
    pub name: String,
    pub entry: ServerEntry,
}

/// Types of configuration changes detected
#[derive(Debug, Clone)]
pub enum ConfigChange {
    /// New server added
    ServerAdded(ServerChange),
    /// Existing server removed
    ServerRemoved(String),
    /// Server configuration updated
    ServerUpdated(ServerChange),
    /// Full reload triggered
    FullReload(RegistryConfig),
}

/// Hot-swap event with timestamp
#[derive(Debug, Clone)]
pub struct HotSwapEvent {
    pub change: ConfigChange,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Statistics for hot-swap operations
#[derive(Debug, Default, Clone)]
pub struct HotSwapStats {
    pub reloads_performed: u64,
    pub servers_added: u64,
    pub servers_removed: u64,
    pub servers_updated: u64,
    pub errors_encountered: u64,
    pub last_reload: Option<chrono::DateTime<chrono::Utc>>,
}

/// Hot-Swap Manager
///
/// Watches configuration files and provides reload notifications.
pub struct HotSwapManager {
    config_path: PathBuf,
    current_config: Arc<RwLock<RegistryConfig>>,
    stats: Arc<RwLock<HotSwapStats>>,
    event_sender: Option<Sender<HotSwapEvent>>,
    _watcher: Option<RecommendedWatcher>,
    debounce_ms: u64,
    last_event: Arc<RwLock<Option<Instant>>>,
}

impl HotSwapManager {
    /// Create a new hot-swap manager
    pub fn new(config_path: PathBuf) -> Result<Self> {
        // Load initial configuration
        let initial_config = RegistryConfig::load(config_path.clone())?;

        tracing::info!(
            "[HotSwap] Initialized with {} servers from {:?}",
            initial_config.servers.len(),
            config_path
        );

        Ok(Self {
            config_path,
            current_config: Arc::new(RwLock::new(initial_config)),
            stats: Arc::new(RwLock::new(HotSwapStats::default())),
            event_sender: None,
            _watcher: None,
            debounce_ms: 500, // Debounce file changes by 500ms
            last_event: Arc::new(RwLock::new(None)),
        })
    }

    /// Start watching for configuration changes
    ///
    /// Returns a receiver for hot-swap events
    pub fn start_watching(&mut self) -> Result<Receiver<HotSwapEvent>> {
        let (tx, rx) = channel();
        self.event_sender = Some(tx.clone());

        let config_path = self.config_path.clone();
        let current_config = self.current_config.clone();
        let stats = self.stats.clone();
        let debounce_ms = self.debounce_ms;
        let last_event = self.last_event.clone();

        // Create file watcher
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    // Only handle modify events
                    if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        // Check if we need to debounce
                        let should_process = {
                            let mut last = last_event.write();
                            let now = Instant::now();
                            let should = last.map_or(true, |l| {
                                now.duration_since(l) > Duration::from_millis(debounce_ms)
                            });
                            if should {
                                *last = Some(now);
                            }
                            should
                        };

                        if should_process {
                            tracing::info!("[HotSwap] Configuration change detected");

                            if let Err(e) = handle_config_change(
                                &config_path,
                                &current_config,
                                &stats,
                                &tx,
                            ) {
                                tracing::error!("[HotSwap] Failed to handle config change: {}", e);
                                let mut s = stats.write();
                                s.errors_encountered += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("[HotSwap] Watcher error: {}", e);
                }
            }
        })?;

        // Watch the config file's parent directory
        let watch_path = self.config_path.parent().unwrap_or(Path::new("."));
        watcher.watch(watch_path, RecursiveMode::NonRecursive)?;

        self._watcher = Some(watcher);

        tracing::info!("[HotSwap] Watching for changes at {:?}", watch_path);

        Ok(rx)
    }

    /// Manually trigger a reload
    ///
    /// NOTE: Currently unused because the file watcher auto-detects changes.
    /// Kept for potential future CLI command `umb --reload` or programmatic reload.
    #[allow(dead_code)]
    pub fn reload(&self) -> Result<Vec<ConfigChange>> {
        let tx = self.event_sender.as_ref()
            .ok_or_else(|| anyhow!("Hot-swap not started"))?;

        handle_config_change(
            &self.config_path,
            &self.current_config,
            &self.stats,
            tx,
        )
    }

    /// Get the current configuration
    pub fn get_config(&self) -> RegistryConfig {
        self.current_config.read().clone()
    }

    /// Get hot-swap statistics
    pub fn get_stats(&self) -> HotSwapStats {
        self.stats.read().clone()
    }

    /// Check if hot-swap is enabled
    pub fn is_enabled(&self) -> bool {
        self._watcher.is_some()
    }

    /// Stop watching for changes
    ///
    /// NOTE: Currently unused because the HotSwapManager is consumed by a spawned
    /// tokio task in main.rs, and the MCP server runs until process termination.
    /// Graceful shutdown would require architectural changes (e.g., storing the
    /// manager in an Arc<Mutex> accessible from a shutdown handler, or using
    /// tokio's signal handling to coordinate cleanup).
    #[allow(dead_code)]
    pub fn stop(&mut self) {
        self._watcher = None;
        self.event_sender = None;
        tracing::info!("[HotSwap] Stopped watching for changes");
    }

    /// Log current statistics
    ///
    /// Helper method to log stats after hot-swap events
    pub fn log_stats(&self) {
        let stats = self.get_stats();
        if stats.reloads_performed > 0 {
            tracing::info!(
                "[HotSwap] Stats: {} reloads, {} added, {} removed, {} updated, {} errors, last: {:?}",
                stats.reloads_performed,
                stats.servers_added,
                stats.servers_removed,
                stats.servers_updated,
                stats.errors_encountered,
                stats.last_reload.map(|dt| dt.format("%H:%M:%S").to_string())
            );
        }
    }
}

/// Handle a configuration change
fn handle_config_change(
    config_path: &Path,
    current_config: &Arc<RwLock<RegistryConfig>>,
    stats: &Arc<RwLock<HotSwapStats>>,
    tx: &Sender<HotSwapEvent>,
) -> Result<Vec<ConfigChange>> {
    // Load new configuration
    let new_config = RegistryConfig::load(config_path.to_path_buf())?;

    // Compare with current configuration
    let changes = diff_configs(&current_config.read(), &new_config);

    if changes.is_empty() {
        tracing::debug!("[HotSwap] No configuration changes detected");
        return Ok(changes);
    }

    // Update current configuration
    *current_config.write() = new_config.clone();

    // Update stats and send events
    let mut s = stats.write();
    s.reloads_performed += 1;
    s.last_reload = Some(chrono::Utc::now());

    for change in &changes {
        let event = HotSwapEvent {
            change: change.clone(),
            timestamp: chrono::Utc::now(),
        };

        match change {
            ConfigChange::ServerAdded(_) => s.servers_added += 1,
            ConfigChange::ServerRemoved(_) => s.servers_removed += 1,
            ConfigChange::ServerUpdated(_) => s.servers_updated += 1,
            ConfigChange::FullReload(_) => {}
        }

        if let Err(e) = tx.send(event) {
            tracing::error!("[HotSwap] Failed to send event: {}", e);
        }
    }

    tracing::info!(
        "[HotSwap] Applied {} configuration changes",
        changes.len()
    );

    Ok(changes)
}

/// Compare two configurations and return the differences
fn diff_configs(old: &RegistryConfig, new: &RegistryConfig) -> Vec<ConfigChange> {
    let mut changes = Vec::new();

    let old_servers: HashMap<_, _> = old.servers.iter().collect();
    let new_servers: HashMap<_, _> = new.servers.iter().collect();

    // Find removed servers
    for (name, _) in &old_servers {
        if !new_servers.contains_key(name) {
            changes.push(ConfigChange::ServerRemoved(name.to_string()));
        }
    }

    // Find added and updated servers
    for (name, new_server) in &new_servers {
        match old_servers.get(name) {
            Some(old_server) => {
                if !servers_equal(old_server, new_server) {
                    changes.push(ConfigChange::ServerUpdated(ServerChange {
                        name: name.to_string(),
                        entry: (*new_server).clone(),
                    }));
                }
            }
            None => {
                changes.push(ConfigChange::ServerAdded(ServerChange {
                    name: name.to_string(),
                    entry: (*new_server).clone(),
                }));
            }
        }
    }

    changes
}

/// Compare two server entries for equality
fn servers_equal(a: &ServerEntry, b: &ServerEntry) -> bool {
    // Compare using accessor methods for the enum variants
    a.command() == b.command()
        && a.args() == b.args()
        && a.env() == b.env()
        && a.is_sse() == b.is_sse()
        && a.sse_url() == b.sse_url()
        && a.is_enabled() == b.is_enabled()
        && a.is_http() == b.is_http()
        && a.http_url() == b.http_url()
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use std::collections::HashMap;

    #[test]
    fn test_diff_configs_no_changes() {
        let config = RegistryConfig {
            servers: IndexMap::new(),
        };

        let changes = diff_configs(&config, &config);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_diff_configs_server_added() {
        let old = RegistryConfig {
            servers: IndexMap::new(),
        };

        let mut new = RegistryConfig {
            servers: IndexMap::new(),
        };
        new.servers.insert(
            "test".to_string(),
            ServerEntry::Stdio {
                command: "node".to_string(),
                args: vec![],
                env: HashMap::new(),
                transport_type: None,
                enabled: true,
                name: None,
            },
        );

        let changes = diff_configs(&old, &new);
        assert_eq!(changes.len(), 1);
        matches!(&changes[0], ConfigChange::ServerAdded(_));
    }

    #[test]
    fn test_diff_configs_server_removed() {
        let mut old = RegistryConfig {
            servers: IndexMap::new(),
        };
        old.servers.insert(
            "test".to_string(),
            ServerEntry::Stdio {
                command: "node".to_string(),
                args: vec![],
                env: HashMap::new(),
                transport_type: None,
                enabled: true,
                name: None,
            },
        );

        let new = RegistryConfig {
            servers: IndexMap::new(),
        };

        let changes = diff_configs(&old, &new);
        assert_eq!(changes.len(), 1);
        matches!(&changes[0], ConfigChange::ServerRemoved(_));
    }
}
