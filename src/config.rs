//! UMB Configuration Module
//!
//! Manages user configuration from ~/.umb/config.toml
//! Allows customization of embedding models, dimensions, and other features.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::features::EmbeddingDimension;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UmbConfig {
    /// Semantic search configuration
    pub semantic_search: SemanticSearchConfig,

    /// General settings
    pub general: GeneralConfig,
}

/// Semantic search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticSearchConfig {
    /// Semantic search backend.
    ///
    /// - `"keyword"` (default): zero-dependency substring/keyword matching,
    ///   fully offline, no model download. This is the open-release default.
    /// - any other value: a model-backed embedding backend (e.g. ONNX) is
    ///   initialized at startup.
    #[serde(default = "default_search_backend")]
    pub backend: String,

    /// Embedding dimension to use (128, 256, 512, or 768)
    /// Higher dimensions = better quality but slower
    /// Default: 768 (maximum quality)
    pub dimension: u32,

    /// Custom embedding model path (ONNX format)
    /// Leave empty to use the default EmbeddingGemma model from HuggingFace
    ///
    /// To use a custom model, set this to the full path of your ONNX model:
    /// custom_model_path = "/path/to/your/model.onnx"
    ///
    /// Note: The model must output embeddings compatible with Matryoshka dimensions
    /// and have a tokenizer.json in the same directory.
    #[serde(default)]
    pub custom_model_path: Option<String>,

    /// Custom tokenizer path (optional)
    /// If not specified and custom_model_path is set, looks for tokenizer.json
    /// in the same directory as the model
    #[serde(default)]
    pub custom_tokenizer_path: Option<String>,

    /// HuggingFace model repository (used when custom_model_path is not set)
    /// Default: "onnx-community/embeddinggemma-300m-ONNX"
    #[serde(default = "default_hf_repo")]
    pub huggingface_repo: String,

    /// ONNX model file within the HuggingFace repo
    /// Options: "onnx/model_fp16.onnx" (default, best quality)
    ///          "onnx/model_q4.onnx" (smaller, slightly lower quality)
    ///          "onnx/model_quantized.onnx" (int8 quantized)
    #[serde(default = "default_hf_model_file")]
    pub huggingface_model_file: String,

    /// Similarity threshold for search results (0.0 - 1.0)
    ///
    /// Controls how similar a tool must be to your query to be included in results.
    /// - 0.0 = Include everything (no filtering)
    /// - 0.3 = Default, good balance - filters noise while keeping relevant results
    /// - 0.5 = Stricter - only shows highly relevant tools
    /// - 0.7+ = Very strict - may miss some relevant results
    ///
    /// Lower values: More results, may include less relevant tools
    /// Higher values: Fewer results, but more precisely matched
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Maximum number of search results to return (1-100)
    ///
    /// Limits how many tools are returned from a semantic search query.
    /// - 5-10 = Good for focused queries, faster response
    /// - 20-50 = Good for exploratory queries
    /// - 100 = Maximum, may slow response with large tool sets
    ///
    /// Default: 10
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

/// General configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// Enable debug logging
    #[serde(default)]
    pub debug: bool,

    /// Path to the MCP servers configuration file
    ///
    /// This file contains the list of registered MCP servers.
    /// Default: ~/.umb/servers.json
    #[serde(default)]
    pub servers_path: Option<String>,

    /// Cache directory for models and embeddings
    /// Default: ~/.umb/cache
    #[serde(default)]
    pub cache_dir: Option<String>,

    /// Idle TTL (seconds) for pooled MCP-server connections.
    ///
    /// A pooled connection that has not served a call for longer than this
    /// is evicted by a periodic background sweeper (eviction reuses the
    /// existing hardened teardown — pgid kill + subreaper adopted-orphan
    /// sweep + 1e ident-revalidation + probe allowlist). Bounds idle memory.
    ///
    /// Default: 600 (10 minutes), ENABLED by default. `0` disables idle
    /// eviction entirely (pooled connections live until process exit /
    /// failure). Back-compat: a `config.toml` missing this key
    /// deserializes to the 600 default (no `deny_unknown_fields` anywhere,
    /// so older configs keep parsing unchanged).
    #[serde(default = "default_pool_idle_ttl_secs")]
    pub pool_idle_ttl_secs: u64,

    /// Tool-dictionary short-definition overlay mode.
    ///
    /// Controls whether the community-curated short_descriptions in
    /// `tool-dictionary/*.toml` override the live MCP-server-supplied
    /// descriptions at READ time (`get_tool_info` + the slim `list_tools`
    /// envelope). Underlying tool storage is never mutated; the overlay
    /// is applied per response and flippable at runtime.
    ///
    /// Values (case-insensitive; unknown ⇒ "auto"):
    /// - "off"  — never consult the dict; always use live server def.
    /// - "auto" (default) — apply if a dict entry exists AND either has
    ///   no recorded hash OR its recorded sha256 matches the live
    ///   description's hash. Silent fallback to live on mismatch.
    /// - "on"   — apply if a dict entry exists, regardless of hash.
    ///
    /// Default: "auto" (safe-useful; safer than "on" via hash-guard,
    /// strictly more useful than "off" for the median user).
    /// Back-compat: a `config.toml` missing this key deserializes to "auto".
    #[serde(default = "default_short_definitions")]
    pub short_definitions: String,

    /// Additional directories scanned for tool-dictionary `*.toml` files.
    ///
    /// Loaded AFTER the compile-time fallback + in-repo on-disk dir, and
    /// BEFORE the user overlay (`tool_dictionary_user_dir`). Useful for
    /// org-shared dicts vendored next to the deployed binary.
    ///
    /// Default: empty.
    #[serde(default)]
    pub tool_dictionary_paths: Vec<String>,

    /// User overlay directory for `tool-dictionary/*.toml` files.
    ///
    /// HIGHEST precedence — user wins over compiled-in, in-repo, and
    /// config-path entries (per `(server, tool)` key). Tilde `~` is
    /// expanded at load time.
    ///
    /// Default: `~/.umb/tool-dictionary`.
    #[serde(default = "default_tool_dictionary_user_dir")]
    pub tool_dictionary_user_dir: String,
}

// Default value functions for serde
fn default_search_backend() -> String {
    "keyword".to_string()
}

fn default_hf_repo() -> String {
    "onnx-community/embeddinggemma-300m-ONNX".to_string()
}

fn default_hf_model_file() -> String {
    "onnx/model_fp16.onnx".to_string()
}

fn default_similarity_threshold() -> f32 {
    0.3
}

fn default_max_results() -> usize {
    10
}

/// Default pooled-connection idle TTL: 600s (10 min), ENABLED by default.
/// `0` (explicitly set) disables idle eviction. A missing key →this default.
fn default_pool_idle_ttl_secs() -> u64 {
    600
}

/// Default tool-dictionary mode: `"auto"` (safe-useful default; hash-guarded
/// silent fallback to live def). See `GeneralConfig::short_definitions`.
fn default_short_definitions() -> String {
    "auto".to_string()
}

/// Default user-overlay directory for tool-dictionary entries.
/// Tilde `~` is expanded by the loader.
fn default_tool_dictionary_user_dir() -> String {
    "~/.umb/tool-dictionary".to_string()
}

impl Default for UmbConfig {
    fn default() -> Self {
        Self {
            semantic_search: SemanticSearchConfig::default(),
            general: GeneralConfig::default(),
        }
    }
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            backend: default_search_backend(),
            dimension: 768,
            custom_model_path: None,
            custom_tokenizer_path: None,
            huggingface_repo: default_hf_repo(),
            huggingface_model_file: default_hf_model_file(),
            similarity_threshold: default_similarity_threshold(),
            max_results: default_max_results(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            debug: false,
            servers_path: None,
            cache_dir: None,
            pool_idle_ttl_secs: default_pool_idle_ttl_secs(),
            short_definitions: default_short_definitions(),
            tool_dictionary_paths: Vec::new(),
            tool_dictionary_user_dir: default_tool_dictionary_user_dir(),
        }
    }
}

impl UmbConfig {
    /// Get the config file path
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".umb")
            .join("config.toml")
    }

    /// Load configuration from file, or create default if not exists
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .context("Failed to read config file")?;

            let config: UmbConfig = toml::from_str(&content)
                .context("Failed to parse config file")?;

            tracing::info!("[Config] Loaded from {:?}", config_path);
            Ok(config)
        } else {
            // Create default config file
            let config = UmbConfig::default();
            config.save()?;
            tracing::info!("[Config] Created default config at {:?}", config_path);
            Ok(config)
        }
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();

        // Ensure directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;

        // Add helpful comments to the config file
        let commented_content = Self::add_config_comments(&content);

        fs::write(&config_path, commented_content)
            .context("Failed to write config file")?;

        Ok(())
    }

    /// Add helpful comments to the TOML config
    fn add_config_comments(content: &str) -> String {
        let header = r#"# UMB (Universal MCP Bridge) Configuration
# ==========================================
#
# This file controls UMB behavior. Edit values below to customize.
# Delete this file to reset to defaults.
#
# Documentation: https://universalmcpbridge.app/docs/config

"#;

        let semantic_comment = r#"
# Semantic Search Configuration
# -----------------------------
# Controls the AI-powered tool discovery feature.
#
# BACKEND:
# backend = "keyword" (default) uses zero-dependency offline keyword matching.
# Set backend to a model-backed value to enable embedding-based search.
#
# EMBEDDING MODEL:
# By default, UMB downloads EmbeddingGemma from HuggingFace Hub (~617MB).
# To use your own ONNX embedding model:
#   1. Download an ONNX embedding model
#   2. Set custom_model_path to the full path of the .onnx file
#   3. Ensure tokenizer.json is in the same directory (or set custom_tokenizer_path)
#
# Example:
#   custom_model_path = "/path/to/your/embedding/model.onnx"
#
# DIMENSIONS:
# EmbeddingGemma supports Matryoshka dimensions (128, 256, 512, 768).
# Higher dimensions = better semantic understanding, slightly slower.
# Lower dimensions = faster, good for simple tool matching.
#
# SIMILARITY THRESHOLD (0.0 - 1.0):
# Controls minimum similarity for search results.
#   0.0  = Show all results (no filtering)
#   0.3  = Default - balanced, filters noise
#   0.5  = Stricter - only highly relevant tools
#   0.7+ = Very strict - may miss some matches
#
# MAX RESULTS (1-100):
# Limits search results returned.
#   5-10  = Focused, faster response (default: 10)
#   20-50 = Exploratory queries
#   100   = Maximum, may slow response
#
"#;

        let general_comment = r#"
# General Configuration
# ---------------------
# servers_path: Location of your MCP servers configuration file
#               Default: ~/.umb/servers.json
#
"#;

        format!(
            "{}{}{}{}",
            header,
            semantic_comment,
            general_comment,
            content
        )
    }

    /// Get the embedding dimension as EmbeddingDimension enum
    pub fn get_embedding_dimension(&self) -> EmbeddingDimension {
        EmbeddingDimension::from_size(self.semantic_search.dimension as usize)
            .unwrap_or(EmbeddingDimension::D768)
    }

    /// Check if a custom model is configured
    pub fn has_custom_model(&self) -> bool {
        self.semantic_search.custom_model_path.is_some()
    }

    /// Get the cache directory
    pub fn get_cache_dir(&self) -> PathBuf {
        self.general.cache_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".umb")
                    .join("cache")
            })
    }

    /// Get the servers file path
    pub fn get_servers_path(&self) -> PathBuf {
        self.general.servers_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".umb")
                    .join("servers.json")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = UmbConfig::default();
        assert_eq!(config.semantic_search.dimension, 768);
        assert!(config.semantic_search.custom_model_path.is_none());
        assert_eq!(config.semantic_search.huggingface_repo, "onnx-community/embeddinggemma-300m-ONNX");
    }

    #[test]
    fn test_config_serialization() {
        let config = UmbConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        assert!(toml_str.contains("dimension = 768"));
    }

    #[test]
    fn test_embedding_dimension_conversion() {
        let mut config = UmbConfig::default();

        config.semantic_search.dimension = 128;
        assert_eq!(config.get_embedding_dimension(), EmbeddingDimension::D128);

        config.semantic_search.dimension = 256;
        assert_eq!(config.get_embedding_dimension(), EmbeddingDimension::D256);

        config.semantic_search.dimension = 512;
        assert_eq!(config.get_embedding_dimension(), EmbeddingDimension::D512);

        config.semantic_search.dimension = 768;
        assert_eq!(config.get_embedding_dimension(), EmbeddingDimension::D768);

        // Invalid dimension defaults to 768
        config.semantic_search.dimension = 999;
        assert_eq!(config.get_embedding_dimension(), EmbeddingDimension::D768);
    }
}
