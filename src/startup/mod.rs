//! Startup module - server initialization, discovery, and hot-swap

pub mod discovery;
pub mod hot_swap;
pub mod server;

// `handle_request` is intentional PUBLIC API surface (a re-export for
// embedders/integration tests); it has no in-crate caller so rustc flags
// it unused. Removing it would change the crate's public API — not allowed
// for a publish-ready repo — so the warning is suppressed, not "fixed".
#[allow(unused_imports)]
pub use server::{handle_request, start_server_silent};
