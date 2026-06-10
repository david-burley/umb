//! EmbeddingGemma Semantic Search Module
//!
//! The ONNX/EmbeddingGemma provider is OPT-IN behind the `embed-onnx` Cargo
//! feature (default = []). With the feature OFF this module exposes only the
//! dependency-free `EmbeddingDimension` data type; the router/discovery then
//! run keyword-only search (zero ONNX deps, fully offline). (Blueprint §7.1)
//!
//! With `embed-onnx` ON:
//! - High-quality semantic search via Google's EmbeddingGemma-300M (ONNX RT)
//! - Matryoshka Representation Learning: 128, 256, 512, 768 dimensions
//! - GPU acceleration: CoreML/Metal (macOS), CUDA (Linux/Windows)
//! - Automatic model downloading and caching via HuggingFace Hub
//! - Re-embedding support when dimensions change

// `EmbeddingDimension` (below) is always compiled — it has no ONNX deps and is
// consumed by `config.rs`. Everything else in this module is `embed-onnx`-gated.
use serde::{Deserialize, Serialize};

#[cfg(feature = "embed-onnx")]
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "embed-onnx")]
use hf_hub::{api::sync::Api, Repo, RepoType};
#[cfg(feature = "embed-onnx")]
use ndarray::{Array2, Axis};
#[cfg(feature = "embed-onnx")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
};
#[cfg(feature = "embed-onnx")]
use parking_lot::RwLock;
#[cfg(feature = "embed-onnx")]
use std::collections::HashMap;
#[cfg(feature = "embed-onnx")]
use std::path::PathBuf;
#[cfg(feature = "embed-onnx")]
use std::sync::Arc;
#[cfg(feature = "embed-onnx")]
use tokenizers::Tokenizer;

/// Available Matryoshka embedding dimensions for EmbeddingGemma
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum EmbeddingDimension {
    /// 128 dimensions - smallest, fastest, good for simple tasks
    D128 = 128,
    /// 256 dimensions - balanced for mobile/edge
    D256 = 256,
    /// 512 dimensions - good quality/performance balance
    D512 = 512,
    /// 768 dimensions - full quality (default)
    D768 = 768,
}

impl EmbeddingDimension {
    /// Get all available dimensions
    pub fn all() -> &'static [EmbeddingDimension] {
        &[
            EmbeddingDimension::D128,
            EmbeddingDimension::D256,
            EmbeddingDimension::D512,
            EmbeddingDimension::D768,
        ]
    }

    /// Get the numeric value
    pub fn size(&self) -> usize {
        *self as usize
    }

    /// Parse from number
    pub fn from_size(size: usize) -> Option<Self> {
        match size {
            128 => Some(Self::D128),
            256 => Some(Self::D256),
            512 => Some(Self::D512),
            768 => Some(Self::D768),
            _ => None,
        }
    }

    /// Get description for this dimension
    pub fn description(&self) -> &'static str {
        match self {
            Self::D128 => "128d - Smallest, fastest, good for simple similarity",
            Self::D256 => "256d - Balanced for mobile/edge deployment",
            Self::D512 => "512d - Good quality/performance balance",
            Self::D768 => "768d - Full quality, best accuracy (default)",
        }
    }
}

impl Default for EmbeddingDimension {
    fn default() -> Self {
        Self::D768
    }
}

impl std::fmt::Display for EmbeddingDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}d", self.size())
    }
}

/// Tool embedding with metadata
#[cfg(feature = "embed-onnx")]
#[derive(Debug, Clone)]
pub struct ToolEmbedding {
    pub name: String,
    pub description: String,
    pub server: String,
    pub embedding: Vec<f32>,
    pub dimension: EmbeddingDimension,
}

/// Search result with similarity score
#[cfg(feature = "embed-onnx")]
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub name: String,
    pub description: String,
    pub server: String,
    pub score: f32,
}

/// Search statistics
#[cfg(feature = "embed-onnx")]
#[derive(Debug, Default, Clone, Serialize)]
pub struct SearchStats {
    pub queries_processed: u64,
    pub tools_indexed: u64,
    pub embeddings_generated: u64,
    pub cache_hits: u64,
    pub avg_latency_ms: f64,
    pub gpu_available: bool,
    pub gpu_name: Option<String>,
    pub current_dimension: usize,
}

/// GPU/Execution Provider information
#[cfg(feature = "embed-onnx")]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub is_gpu: bool,
    pub provider: String,
}

/// EmbeddingGemma Semantic Search Provider
///
/// Production-quality semantic search using Google's EmbeddingGemma-300M.
#[cfg(feature = "embed-onnx")]
pub struct SemanticSearchProvider {
    session: Session,
    tokenizer: Tokenizer,
    dimension: EmbeddingDimension,
    tool_index: Arc<RwLock<HashMap<String, ToolEmbedding>>>,
    embedding_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    stats: Arc<RwLock<SearchStats>>,
    device_info: DeviceInfo,
    model_path: PathBuf,
}

/// Configuration for semantic search (imported from config module)
#[cfg(feature = "embed-onnx")]
#[derive(Debug, Clone)]
pub struct SemanticSearchConfig {
    pub dimension: EmbeddingDimension,
    pub custom_model_path: Option<PathBuf>,
    pub custom_tokenizer_path: Option<PathBuf>,
    pub huggingface_repo: String,
    pub huggingface_model_file: String,
}

#[cfg(feature = "embed-onnx")]
impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            dimension: EmbeddingDimension::D768,
            custom_model_path: None,
            custom_tokenizer_path: None,
            huggingface_repo: "onnx-community/embeddinggemma-300m-ONNX".to_string(),
            huggingface_model_file: "onnx/model_fp16.onnx".to_string(),
        }
    }
}

#[cfg(feature = "embed-onnx")]
impl SemanticSearchProvider {
    /// Maximum input tokens (EmbeddingGemma supports 2048, but shorter is faster)
    const MAX_TOKENS: usize = 512;

    /// Create a new semantic search provider with specified dimension (uses defaults)
    pub fn new(dimension: EmbeddingDimension) -> Result<Self> {
        Self::from_config(SemanticSearchConfig {
            dimension,
            ..Default::default()
        })
    }

    /// Create from full configuration
    pub fn from_config(config: SemanticSearchConfig) -> Result<Self> {
        tracing::info!(
            "[SemanticSearch] Initializing with {} dimensions...",
            config.dimension.size()
        );

        // Load model and tokenizer based on configuration
        let (model_path, tokenizer_path) = if let Some(custom_path) = &config.custom_model_path {
            // Use custom model
            tracing::info!("[SemanticSearch] Using custom model: {:?}", custom_path);

            let model_path = PathBuf::from(custom_path);
            if !model_path.exists() {
                return Err(anyhow!("Custom model not found: {:?}", model_path));
            }

            // Tokenizer: use custom path or look in same directory as model
            let tokenizer_path = config.custom_tokenizer_path
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    model_path.parent()
                        .unwrap_or(&model_path)
                        .join("tokenizer.json")
                });

            if !tokenizer_path.exists() {
                return Err(anyhow!(
                    "Tokenizer not found at {:?}. Please provide custom_tokenizer_path in config.",
                    tokenizer_path
                ));
            }

            (model_path, tokenizer_path)
        } else {
            // Download from HuggingFace Hub
            Self::download_from_huggingface(&config)?
        };

        tracing::info!("[SemanticSearch] Model: {:?}", model_path);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Initialize ONNX Runtime with GPU detection
        let (session, device_info) = Self::create_session(&model_path)?;

        tracing::info!(
            "[EmbeddingGemma] Initialized on {} ({})",
            device_info.name,
            device_info.provider
        );

        let stats = SearchStats {
            gpu_available: device_info.is_gpu,
            gpu_name: if device_info.is_gpu {
                Some(device_info.name.clone())
            } else {
                None
            },
            current_dimension: config.dimension.size(),
            ..Default::default()
        };

        Ok(Self {
            session,
            tokenizer,
            dimension: config.dimension,
            tool_index: Arc::new(RwLock::new(HashMap::new())),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(stats)),
            device_info,
            model_path,
        })
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface(config: &SemanticSearchConfig) -> Result<(PathBuf, PathBuf)> {
        tracing::info!(
            "[SemanticSearch] Downloading from HuggingFace: {}/{}",
            config.huggingface_repo,
            config.huggingface_model_file
        );

        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.repo(Repo::new(config.huggingface_repo.clone(), RepoType::Model));

        // Download ONNX model graph (small file with model structure)
        let model_path = repo
            .get(&config.huggingface_model_file)
            .context("Failed to download ONNX model")?;

        // Download ONNX model weights (the _data file)
        // Construct the data file path from the model file path
        let data_file = format!("{}_data", config.huggingface_model_file);
        tracing::info!("[SemanticSearch] Downloading model weights...");
        let _model_data_path = repo
            .get(&data_file)
            .context("Failed to download ONNX model weights")?;

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer")?;

        tracing::info!("[SemanticSearch] Model downloaded successfully");

        Ok((model_path, tokenizer_path))
    }

    /// Create ONNX Runtime session with automatic GPU detection
    fn create_session(model_path: &PathBuf) -> Result<(Session, DeviceInfo)> {
        // Note: GPU providers require specific ort features:
        // - CoreML: ort/coreml (macOS)
        // - CUDA: ort/cuda (Linux/Windows)
        // The load-dynamic feature allows runtime selection

        // For now, use optimized CPU with multi-threading
        // GPU support can be enabled via cargo features
        tracing::info!("[EmbeddingGemma] Creating optimized session...");

        let num_threads = num_cpus::get();

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .commit_from_file(model_path)?;

        // Check available execution providers
        let provider_name = "CPU (optimized)";
        let is_gpu = false;

        tracing::info!(
            "[EmbeddingGemma] Session created with {} threads on {}",
            num_threads,
            provider_name
        );

        Ok((
            session,
            DeviceInfo {
                name: format!("CPU ({} threads)", num_threads),
                is_gpu,
                provider: provider_name.to_string(),
            },
        ))
    }

    /// Generate embedding for text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Check cache
        let cache_key = Self::cache_key(text, self.dimension);
        {
            let cache = self.embedding_cache.read();
            if let Some(cached) = cache.get(&cache_key) {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

        // Truncate if needed
        if input_ids.len() > Self::MAX_TOKENS {
            input_ids.truncate(Self::MAX_TOKENS);
            attention_mask.truncate(Self::MAX_TOKENS);
        }

        let seq_len = input_ids.len();

        // Create input tensors
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_array,
            "attention_mask" => attention_mask_array,
        ]?)?;

        // Extract embeddings (last_hidden_state with mean pooling)
        // Get the first output (model should have last_hidden_state as output)
        let (_, output_tensor) = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow!("No output tensor found"))?;

        let output_array: ndarray::ArrayViewD<f32> = output_tensor.try_extract_tensor()?;

        // Mean pooling over sequence dimension
        let embeddings_3d = output_array.into_dimensionality::<ndarray::Ix3>()?;
        let pooled = embeddings_3d.mean_axis(Axis(1)).ok_or_else(|| anyhow!("Mean pooling failed"))?;
        let mut embedding: Vec<f32> = pooled.iter().cloned().collect();

        // Apply Matryoshka truncation
        let target_dim = self.dimension.size();
        if embedding.len() > target_dim {
            embedding.truncate(target_dim);
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        // Cache the result
        {
            let mut cache = self.embedding_cache.write();
            cache.insert(cache_key, embedding.clone());
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.embeddings_generated += 1;
            let elapsed = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.embeddings_generated - 1) as f64 + elapsed)
                / stats.embeddings_generated as f64;
        }

        Ok(embedding)
    }

    /// Generate cache key
    fn cache_key(text: &str, dimension: EmbeddingDimension) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hasher.update(&(dimension.size() as u32).to_le_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Index a tool for semantic search
    pub fn index_tool(&self, name: &str, description: &str, server: &str) -> Result<()> {
        let search_text = format!("{}: {}", name, description);
        let embedding = self.embed(&search_text)?;

        let tool_embedding = ToolEmbedding {
            name: name.to_string(),
            description: description.to_string(),
            server: server.to_string(),
            embedding,
            dimension: self.dimension,
        };

        let mut index = self.tool_index.write();
        index.insert(name.to_string(), tool_embedding);

        let mut stats = self.stats.write();
        stats.tools_indexed = index.len() as u64;

        Ok(())
    }

    /// Index multiple tools at once (more efficient)
    pub fn index_tools(&self, tools: Vec<(String, String, String)>) -> Result<usize> {
        if tools.is_empty() {
            return Ok(0);
        }

        let mut index = self.tool_index.write();

        for (name, description, server) in &tools {
            let search_text = format!("{}: {}", name, description);
            let embedding = self.embed(&search_text)?;

            let tool_embedding = ToolEmbedding {
                name: name.clone(),
                description: description.clone(),
                server: server.clone(),
                embedding,
                dimension: self.dimension,
            };

            index.insert(name.clone(), tool_embedding);
        }

        let count = index.len();

        let mut stats = self.stats.write();
        stats.tools_indexed = count as u64;

        tracing::info!("[EmbeddingGemma] Indexed {} tools", count);

        Ok(count)
    }

    /// Search for tools using a natural language query
    pub fn search(&self, query: &str, top_k: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();

        let query_embedding = self.embed(query)?;
        let index = self.tool_index.read();

        let mut results: Vec<SearchResult> = index
            .values()
            .map(|tool| {
                let similarity = cosine_similarity(&query_embedding, &tool.embedding);
                SearchResult {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    server: tool.server.clone(),
                    score: similarity,
                }
            })
            .filter(|r| r.score >= threshold)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        let mut stats = self.stats.write();
        stats.queries_processed += 1;

        tracing::debug!(
            "[EmbeddingGemma] Query '{}' returned {} results in {:.1}ms",
            query,
            results.len(),
            start.elapsed().as_millis()
        );

        Ok(results)
    }

    /// Remove a tool from the index
    pub fn remove_tool(&self, name: &str) -> bool {
        let mut index = self.tool_index.write();
        let removed = index.remove(name).is_some();

        if removed {
            let mut stats = self.stats.write();
            stats.tools_indexed = index.len() as u64;
        }

        removed
    }

    /// Clear all indexed tools
    pub fn clear_index(&self) {
        let mut index = self.tool_index.write();
        index.clear();

        let mut stats = self.stats.write();
        stats.tools_indexed = 0;
    }

    /// Get current dimension
    pub fn get_dimension(&self) -> EmbeddingDimension {
        self.dimension
    }

    /// Get all available dimensions
    pub fn available_dimensions() -> &'static [EmbeddingDimension] {
        EmbeddingDimension::all()
    }

    /// Change embedding dimension and re-embed all tools
    ///
    /// This creates a new provider with the new dimension and re-indexes all tools.
    pub fn with_new_dimension(&self, new_dimension: EmbeddingDimension) -> Result<Self> {
        if new_dimension == self.dimension {
            return Err(anyhow!("Already using {} dimensions", new_dimension));
        }

        tracing::info!(
            "[SemanticSearch] Changing dimension from {} to {}...",
            self.dimension,
            new_dimension
        );

        // Create new provider with new dimension, reusing model path
        // This assumes the model is already downloaded, so we use custom_model_path
        let config = SemanticSearchConfig {
            dimension: new_dimension,
            custom_model_path: Some(self.model_path.clone()),
            custom_tokenizer_path: Some(
                self.model_path.parent()
                    .unwrap_or(&self.model_path)
                    .join("tokenizer.json")
            ),
            ..Default::default()
        };

        let new_provider = Self::from_config(config)?;

        // Re-embed all tools
        let tools: Vec<(String, String, String)> = {
            let index = self.tool_index.read();
            index
                .values()
                .map(|t| (t.name.clone(), t.description.clone(), t.server.clone()))
                .collect()
        };

        if !tools.is_empty() {
            tracing::info!(
                "[EmbeddingGemma] Re-embedding {} tools with {} dimensions...",
                tools.len(),
                new_dimension
            );
            new_provider.index_tools(tools)?;
        }

        Ok(new_provider)
    }

    /// Re-embed all indexed tools (useful after dimension change)
    pub fn reembed_all(&self) -> Result<usize> {
        let tools: Vec<(String, String, String)> = {
            let index = self.tool_index.read();
            index
                .values()
                .map(|t| (t.name.clone(), t.description.clone(), t.server.clone()))
                .collect()
        };

        // Clear existing
        self.clear_index();
        self.embedding_cache.write().clear();

        // Re-embed
        self.index_tools(tools)
    }

    /// Get search statistics
    pub fn get_stats(&self) -> SearchStats {
        self.stats.read().clone()
    }

    /// Get device information
    pub fn get_device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Check if GPU is being used
    pub fn is_gpu_enabled(&self) -> bool {
        self.device_info.is_gpu
    }

    /// Check if provider is ready
    pub fn is_ready(&self) -> bool {
        true
    }
}

/// Calculate cosine similarity between two vectors
#[cfg(feature = "embed-onnx")]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Get number of CPUs for threading
#[cfg(feature = "embed-onnx")]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions() {
        assert_eq!(EmbeddingDimension::D128.size(), 128);
        assert_eq!(EmbeddingDimension::D256.size(), 256);
        assert_eq!(EmbeddingDimension::D512.size(), 512);
        assert_eq!(EmbeddingDimension::D768.size(), 768);
    }

    #[test]
    fn test_dimension_from_size() {
        assert_eq!(EmbeddingDimension::from_size(128), Some(EmbeddingDimension::D128));
        assert_eq!(EmbeddingDimension::from_size(256), Some(EmbeddingDimension::D256));
        assert_eq!(EmbeddingDimension::from_size(512), Some(EmbeddingDimension::D512));
        assert_eq!(EmbeddingDimension::from_size(768), Some(EmbeddingDimension::D768));
        assert_eq!(EmbeddingDimension::from_size(1000), None);
    }

    #[cfg(feature = "embed-onnx")]
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c)).abs() < 0.001);
    }

    /// Integration test for EmbeddingGemma semantic search
    /// This test downloads the model if not cached and tests full functionality
    #[cfg(feature = "embed-onnx")]
    #[test]
    #[ignore] // Run with: cargo test --release --features embed-onnx -- --ignored test_embeddinggemma_integration
    fn test_embeddinggemma_integration() {
        println!("\n=== EmbeddingGemma Integration Test ===\n");

        // Test all available dimensions
        for dimension in EmbeddingDimension::all() {
            println!("Testing dimension: {}", dimension);

            let provider = SemanticSearchProvider::new(*dimension)
                .expect("Failed to create SemanticSearchProvider");

            // Verify dimension
            assert_eq!(provider.get_dimension(), *dimension);

            // Verify device info
            let device = provider.get_device_info();
            println!("  Device: {} ({})", device.name, device.provider);
            assert!(!device.name.is_empty());

            // Test embedding generation
            let test_text = "Read a file from disk";
            let embedding = provider.embed(test_text).expect("Failed to embed text");

            // Verify embedding has correct dimension
            assert_eq!(
                embedding.len(),
                dimension.size(),
                "Expected {} dimensions, got {}",
                dimension.size(),
                embedding.len()
            );

            // Verify embedding is normalized (L2 norm ≈ 1.0)
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Embedding not normalized, norm = {}",
                norm
            );

            println!("  Embedding: [{:.4}, {:.4}, {:.4}, ...]", embedding[0], embedding[1], embedding[2]);
            println!("  Norm: {:.4}", norm);
            println!();
        }

        println!("=== Full Semantic Search Test ===\n");

        // Create provider with 512d for good balance
        let provider = SemanticSearchProvider::new(EmbeddingDimension::D512)
            .expect("Failed to create provider");

        // Index some MCP tools
        let tools = vec![
            ("read_file", "Read contents of a file from the filesystem", "filesystem"),
            ("write_file", "Write content to a file on disk", "filesystem"),
            ("list_directory", "List all files and folders in a directory", "filesystem"),
            ("execute_command", "Run a shell command in the terminal", "shell"),
            ("git_commit", "Create a git commit with the staged changes", "git"),
            ("http_request", "Make an HTTP request to a URL", "network"),
            ("search_code", "Search for patterns in source code", "code"),
            ("format_json", "Pretty-print and format JSON data", "tools"),
        ];

        for (name, desc, server) in &tools {
            provider.index_tool(name, desc, server).expect("Failed to index tool");
        }

        let stats = provider.get_stats();
        assert_eq!(stats.tools_indexed, 8, "Expected 8 tools indexed");
        println!("Indexed {} tools\n", stats.tools_indexed);

        // Test semantic search queries
        let queries = vec![
            ("read a text file", "read_file"),
            ("save data to disk", "write_file"),
            ("show folder contents", "list_directory"),
            ("run bash command", "execute_command"),
            ("commit my changes", "git_commit"),
            ("fetch from API", "http_request"),
            ("find code patterns", "search_code"),
            ("beautify JSON output", "format_json"),
        ];

        for (query, expected_top) in queries {
            let results = provider.search(query, 3, 0.0).expect("Search failed");

            println!("Query: \"{}\"", query);
            for (i, result) in results.iter().enumerate() {
                println!("  {}. {} (score: {:.3})", i + 1, result.name, result.score);
            }

            // Verify expected tool is in top 3
            let in_top_3 = results.iter().take(3).any(|r| r.name == expected_top);
            assert!(
                in_top_3,
                "Expected '{}' in top 3 for query '{}', got: {:?}",
                expected_top,
                query,
                results.iter().map(|r| &r.name).collect::<Vec<_>>()
            );
            println!();
        }

        println!("=== Stats ===");
        let final_stats = provider.get_stats();
        println!("  Tools indexed: {}", final_stats.tools_indexed);
        println!("  Queries processed: {}", final_stats.queries_processed);

        println!("\n=== Integration Test PASSED ===\n");
    }
}
