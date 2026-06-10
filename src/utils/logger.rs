use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logger() {
    // DEV BUILD - Default to debug level for more verbose output
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("debug"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter)
        // IMPORTANT: Write to stderr, not stdout, to avoid polluting MCP stdio protocol
        .with(tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_writer(std::io::stderr))
        .init();
}
