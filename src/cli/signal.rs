//! Signal handling with graceful shutdown coordination
//!
//! Uses tokio_util::sync::CancellationToken for clean shutdown.
//! This fixes the zombie process issue (Spec 001) by allowing
//! the MCP server loop to be interrupted cleanly.

use tokio_util::sync::CancellationToken;

#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};

/// Set up signal handlers that trigger the cancellation token on SIGTERM/SIGHUP
///
/// This allows graceful shutdown without blocking on stdin reads.
/// The CancellationToken can be passed to the MCP server loop which will
/// use tokio::select! to race between stdin reads and cancellation.
#[cfg(unix)]
pub fn setup_signal_handlers(shutdown_token: CancellationToken) {
    tokio::spawn(async move {
        let mut sigterm = signal(SignalKind::terminate())
            .expect("Failed to register SIGTERM handler");
        let mut sighup = signal(SignalKind::hangup())
            .expect("Failed to register SIGHUP handler");
        let mut sigint = signal(SignalKind::interrupt())
            .expect("Failed to register SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => {
                tracing::info!("[Signal] Received SIGTERM, initiating graceful shutdown");
            }
            _ = sighup.recv() => {
                tracing::info!("[Signal] Received SIGHUP (parent died), initiating graceful shutdown");
            }
            _ = sigint.recv() => {
                tracing::info!("[Signal] Received SIGINT, initiating graceful shutdown");
            }
        }

        // Trigger cancellation - this will cause the MCP server loop to exit cleanly
        shutdown_token.cancel();
        tracing::debug!("[Signal] Cancellation token triggered");
    });
}

/// No-op signal handler for non-Unix platforms
#[cfg(not(unix))]
pub fn setup_signal_handlers(_shutdown_token: CancellationToken) {
    // On Windows, we rely on the inactivity timeout and EOF detection
    tracing::debug!("[Signal] Signal handlers not available on this platform");
}

/// Create a new shutdown token and set up signal handlers
///
/// Returns the token which should be passed to the MCP server.
pub fn create_shutdown_handler() -> CancellationToken {
    let token = CancellationToken::new();
    setup_signal_handlers(token.clone());
    token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cancellation_token_creation() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn test_cloned_token_shares_state() {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        assert!(!token.is_cancelled());
        assert!(!token_clone.is_cancelled());

        token.cancel();

        assert!(token.is_cancelled());
        assert!(token_clone.is_cancelled());
    }
}
