//! Feature modules
//!
//! These modules provide UMB's capabilities. There is no tier gating. The
//! ONNX/EmbeddingGemma semantic-search stack is opt-in behind the `embed-onnx`
//! Cargo feature (default = []); the default build is keyword-only. (§7.1)

pub mod hot_swap;
pub mod semantic_search;

// Re-export main types for convenience
pub use hot_swap::{ConfigChange, HotSwapManager, ServerChange};
// `EmbeddingDimension` is always available (zero ONNX deps; used by config.rs).
pub use semantic_search::EmbeddingDimension;
// The ONNX provider + its init config are opt-in behind `embed-onnx` (§7.1).
#[cfg(feature = "embed-onnx")]
pub use semantic_search::{SemanticSearchConfig, SemanticSearchProvider};
