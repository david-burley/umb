//! Build script for UMB
//!
//! Handles:
//! - ONNX Runtime library setup
//! - Optional model pre-download for offline builds
//!
//! §7.1 NOTE — default build performs ZERO network/download:
//! - The ~88MB `libonnxruntime` is fetched by the `ort` crate's
//!   `download-binaries` feature. `ort` is an OPTIONAL dependency gated behind
//!   the `embed-onnx` Cargo feature (default = []), so on a default build `ort`
//!   (and its `download-binaries`) is absent → no ONNX runtime download.
//! - The HuggingFace model pre-download below (`download_embeddinggemma_model`)
//!   is already opt-in: its only call site (in `main()`) is commented out, so
//!   this build script itself never hits the network in any configuration.
//! No `#[cfg]` gating is required in this file; the gating is structural via
//! the optional `ort` dependency in Cargo.toml.

use std::env;
use std::path::PathBuf;

fn main() {
    // Rerun if build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Get output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Set model cache directory for runtime
    let model_cache_dir = out_dir.join("models");
    println!("cargo:rustc-env=UMB_MODEL_CACHE_BUILD={}", model_cache_dir.display());

    // Detect platform for ONNX Runtime
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    println!("cargo:warning=Building for {}-{}", target_os, target_arch);

    // Platform-specific GPU hints
    match target_os.as_str() {
        "macos" => {
            println!("cargo:warning=macOS detected - CoreML/Metal GPU acceleration available");
            println!("cargo:rustc-cfg=has_coreml");
        }
        "linux" | "windows" => {
            println!("cargo:warning=CUDA GPU acceleration available if CUDA toolkit installed");
            println!("cargo:rustc-cfg=has_cuda");
        }
        _ => {}
    }

    // Optional: Pre-download model during build (for offline/embedded builds)
    // Uncomment to enable model embedding:
    // download_embeddinggemma_model(&model_cache_dir);
}

#[allow(dead_code)]
fn download_embeddinggemma_model(cache_dir: &PathBuf) {
    use std::fs;
    use std::io::Write;

    const MODEL_REPO: &str = "onnx-community/embeddinggemma-300m-ONNX";
    const MODEL_FILES: &[&str] = &[
        "onnx/model_fp16.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ];

    // Create cache directory
    if let Err(e) = fs::create_dir_all(cache_dir) {
        println!("cargo:warning=Failed to create model cache: {}", e);
        return;
    }

    println!("cargo:warning=Downloading EmbeddingGemma model files...");

    for file in MODEL_FILES {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            MODEL_REPO, file
        );
        let dest_path = cache_dir.join(file.replace('/', "_"));

        if dest_path.exists() {
            println!("cargo:warning=  [cached] {}", file);
            continue;
        }

        println!("cargo:warning=  [download] {}", file);

        match reqwest::blocking::get(&url) {
            Ok(response) => {
                if response.status().is_success() {
                    match response.bytes() {
                        Ok(bytes) => {
                            if let Ok(mut f) = fs::File::create(&dest_path) {
                                let _ = f.write_all(&bytes);
                            }
                        }
                        Err(e) => println!("cargo:warning=    Failed to read: {}", e),
                    }
                } else {
                    println!("cargo:warning=    HTTP {}", response.status());
                }
            }
            Err(e) => println!("cargo:warning=    Failed: {}", e),
        }
    }
}
