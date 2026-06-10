//! CLI argument parsing
//!
//! Defines the command-line interface for umb.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "umb")]
#[command(about = "Universal MCP Bridge - Connect all your MCP servers", long_about = None)]
#[command(version)]
pub struct Cli {
    /// List configured MCP servers
    #[arg(long)]
    pub list_servers: bool,

    /// Show startup banner and verbose output (default: silent for MCP compliance)
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// Minimal safety utility: in this daemonless 1-process:1-client build
    /// there is no daemon/proxy layer to scan, so this is a no-op that exits 0.
    /// Retained so existing tooling/`--doctor --json` callers keep working.
    #[arg(long)]
    pub doctor: bool,

    /// With --doctor: no-op in the daemonless build (nothing to clean).
    #[arg(long)]
    pub clean: bool,

    /// With --doctor: emit machine-readable JSON output
    #[arg(long)]
    pub json: bool,

    /// With --doctor --clean: skip the confirmation prompt (no-op now)
    #[arg(long)]
    pub yes: bool,

    /// [Deprecated] Silent mode is now the default
    #[arg(long, hide = true)]
    pub stdio: bool,

    /// Minimum cosine similarity threshold for semantic search (0.0–1.0, default: 0.7)
    /// Higher values return fewer but more relevant results.
    /// Lower values return more results but may include irrelevant tools.
    #[arg(long, default_value = "0.7", value_name = "THRESHOLD")]
    pub search_threshold: f32,

    /// Maximum number of tools returned by list_tools (default: 10)
    /// Applies to both semantic search and substring/alphabetical results.
    #[arg(long, default_value = "10", value_name = "LIMIT")]
    pub search_limit: usize,

    /// Dump every registered tool as JSON
    /// `[{name, server, description, source}]` and exit. `source` is
    /// `"dict"` if the tool-dictionary overrode the description and
    /// `"server"` otherwise. Useful for auditing which dict entries
    /// actually fire against your registered MCP server set.
    #[arg(long)]
    pub doctor_tools: bool,
}

impl Cli {
    /// Parse CLI arguments from command line
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Determine if we should run in silent mode
    /// Silent mode: No banner, JSON-only on stdout, logs to stderr
    /// Verbose mode: Show banner and info output
    pub fn is_silent_mode(&self) -> bool {
        !self.verbose && !self.list_servers
    }

    /// Check if this is just a status/info command (no MCP server needed)
    pub fn is_info_command(&self) -> bool {
        self.list_servers || self.doctor || self.doctor_tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_silent() {
        let cli = Cli {
            list_servers: false,
            verbose: false,
            doctor: false,
            clean: false,
            json: false,
            yes: false,
            stdio: false,
            search_threshold: 0.7,
            search_limit: 10,
            doctor_tools: false,
        };
        assert!(cli.is_silent_mode());
    }

    #[test]
    fn test_verbose_not_silent() {
        let cli = Cli {
            list_servers: false,
            verbose: true,
            doctor: false,
            clean: false,
            json: false,
            yes: false,
            stdio: false,
            search_threshold: 0.7,
            search_limit: 10,
            doctor_tools: false,
        };
        assert!(!cli.is_silent_mode());
    }

    #[test]
    fn test_list_servers_not_silent() {
        let cli = Cli {
            list_servers: true,
            verbose: false,
            doctor: false,
            clean: false,
            json: false,
            yes: false,
            stdio: false,
            search_threshold: 0.7,
            search_limit: 10,
            doctor_tools: false,
        };
        assert!(!cli.is_silent_mode());
    }

    #[test]
    fn test_doctor_is_info_command() {
        let cli = Cli {
            list_servers: false,
            verbose: false,
            doctor: true,
            clean: false,
            json: false,
            yes: false,
            stdio: false,
            search_threshold: 0.7,
            search_limit: 10,
            doctor_tools: false,
        };
        assert!(cli.is_info_command());
        assert!(cli.is_silent_mode());
    }
}
