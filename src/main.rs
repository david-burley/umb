//! Universal MCP Bridge (UMB)
//!
//! Main entry point that wires together all modules.
//! The actual logic lives in dedicated modules for better maintainability.

use anyhow::Result;
use clap::Parser;

mod cli;
mod config;
mod features;
mod registry;
mod server;
mod startup;
mod utils;

use cli::{commands, Cli};
use utils::init_logger;

fn main() -> Result<()> {
    init_logger();

    // Parse CLI BEFORE creating tokio runtime - clap handles --version/--help and exits early
    // This prevents tokio initialization from blocking simple commands like --version
    let cli = Cli::parse();

    // Now create tokio runtime and run async main
    tokio::runtime::Runtime::new()?.block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> Result<()> {
    // Default to silent mode for MCP protocol compliance (JSON-only on stdout)
    let silent_mode = !cli.verbose && !cli.list_servers;

    // Handle --doctor flag: daemonless build → minimal no-op utility, exit 0
    if cli.doctor {
        commands::handle_doctor(cli.clean, cli.json, cli.yes).await?;
        return Ok(());
    }

    // Handle --doctor-tools flag: synchronous discovery + per-tool JSON
    // dump with {name, server, description, source} provenance from the
    // tool-dictionary overlay. Subreaper armed before child spawn so pooled
    // MCP discovery children reparent under us (reused teardown path).
    if cli.doctor_tools {
        crate::server::router::install_child_subreaper();
        let res = commands::handle_doctor_tools().await;
        // Reap any tracked pooled children spawned by discovery, same
        // chokepoint as a normal stdio exit.
        crate::server::router::reap_tracked_children();
        return res;
    }

    // Handle --list-servers flag: display server info and exit
    if cli.list_servers {
        commands::handle_list_servers().await?;
        return Ok(());
    }

    // Attempt #8 — PR_SET_CHILD_SUBREAPER. Mark umb as a child subreaper
    // BEFORE any backend/pool process is spawned, so EVERY subsequently-
    // spawned subtree (including a setsid+double-fork daemonized grandchild
    // whose intermediate parent dies) reparents to umb instead of init. This
    // is the structural fix for the A2 residual that no process-tree walk can
    // solve. Best-effort: a prctl failure warns and continues (degrades to
    // the prior pgid-only reap). Linux-only; clean no-op elsewhere. MUST be
    // before `start_server_silent` (which lazily spawns pooled children).
    crate::server::router::install_child_subreaper();

    // Set up signal handling with CancellationToken (Spec 001 - Zombie Process Fix)
    // IMPORTANT: Must be set up AFTER Cli::parse() to avoid zombie processes on --version/--help
    let shutdown_token = cli::create_shutdown_handler();

    // Start MCP server (silent mode is default). This build is strictly
    // 1 process : 1 client over stdio — there is no daemon/proxy layer.
    // Verbose mode is currently the same as silent (verbose banner can be added later).
    let _ = silent_mode;
    let serve_result =
        startup::start_server_silent(shutdown_token, cli.search_threshold, cli.search_limit).await;

    // === SINGLE TEARDOWN CHOKEPOINT (Part B) =============================
    // `start_server_silent` returns when the stdio serve loop exits via ANY
    // path: stdin EOF, parent-death watchdog cancel, inactivity timeout,
    // SIGTERM/SIGINT (the signal handler cancels the same shutdown token the
    // serve loop selects on; see src/server/mcp.rs:303-417 and
    // src/cli/signal.rs:18-42). Every one of those paths funnels here, so the
    // unified child-reap below is the ONE place teardown happens — there is
    // no longer any exit path that skips it.
    //
    // BUG#2 (the literal project-killer): a done CLI MCP server MUST exit
    // immediately. Background work spawned by `start_server_silent` includes
    // the hot-swap `notify` filesystem watcher on a dedicated OS thread that
    // NEVER terminates and is not wired to the shutdown token. A clean
    // `return` here would drop the tokio `Runtime`, whose teardown waits on
    // that never-ending thread — hanging the process alive (campaign E2E: a
    // stdio `umb` survived 35+ min orphaned). `exit(0)` is load-bearing: it
    // is the only way to sidestep that notify-thread Runtime-drop hang. Do
    // NOT replace it with `return`.
    //
    // Residual 2a/A2: `exit(0)` bypasses every `Drop`, so the
    // `kill_on_drop(true)` armed on pooled backend MCP children does NOT
    // fire — the OS only *reparents* those Node/Python children (and any
    // grandchild that called `setsid()`) to init, it does NOT reap them. So
    // before exiting we synchronously terminate the tracked pooled children
    // (process-group SIGTERM + /proc descendant-snapshot + ident-revalidated
    // SIGKILL); it is bounded by design and cannot hang the exit. init then
    // reaps the now-dead children. This runs on EVERY exit path because they
    // all return through here.
    crate::server::router::reap_tracked_children();

    // Surface a genuine serve error to the exit code, but ONLY after the
    // children are reaped (teardown must not be skipped by an early `?`).
    if let Err(e) = serve_result {
        tracing::error!("[main] serve loop returned error: {e}");
        std::process::exit(1);
    }
    std::process::exit(0);
}
