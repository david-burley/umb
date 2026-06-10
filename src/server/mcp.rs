use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

/// Inactivity timeout - exit if no messages received for this duration
const INACTIVITY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10 * 60); // 10 minutes

/// Residual 2b — parent-death watchdog debounce threshold.
///
/// The death/None condition must hold for this many *consecutive* watchdog
/// ticks before we conclude the parent is gone. The watchdog ticks every ~1s
/// (see `tokio::time::interval` below), so N=3 ⇒ ~3s of sustained evidence
/// before shutdown. Rationale for the value:
///   * A genuine parent death is *persistent* — the PID change / unresolvable
///     state holds on every subsequent tick, so 3 ticks always confirm it and
///     the orphan still exits within a few seconds (far inside the "no 35-min
///     orphan" guarantee; `tests/stdio_eof.rs` bounds the real binary).
///   * A transient `parent_pid_now() == None` (a one-off sysinfo `/proc/self`
///     read race) clears on the very next tick, so a single blip can never
///     reach 3 and can no longer false-trigger a shutdown of a HEALTHY stdio
///     server. (Pre-fix the predicate acted on a SINGLE tick.)
///   * 3 is the smallest N that tolerates an isolated double-blip while still
///     confirming real death promptly; larger N only delays a guaranteed-true
///     conclusion with no added safety.
/// Fail-safe direction is unchanged: still errs toward exiting on *sustained*
/// ambiguity (N consecutive None) — just not on a 1-tick blip.
const PARENT_DEATH_DEBOUNCE_TICKS: u32 = 3;

/// BUG#2 parent-death decision (pure, unit-testable) — Residual 2b debounced.
///
/// `original` is the parent PID captured at startup; `current` is the parent
/// PID observed this tick (`None` ⇒ unresolvable). `consecutive_failures` is
/// the running count of *prior* consecutive ticks on which the death/None
/// condition held (the caller owns this counter).
///
/// Returns `(parent_dead, new_consecutive_failures)`:
///   * If this tick looks healthy (`current == Some(original)`), the counter
///     RESETS to 0 and `parent_dead` is false — a recovered transient blip
///     cannot accumulate toward a false shutdown.
///   * If this tick looks dead/unresolvable, the counter increments;
///     `parent_dead` is true ONLY once it reaches
///     `PARENT_DEATH_DEBOUNCE_TICKS` (N consecutive bad ticks). A genuine
///     death persists across ticks so it always reaches N within ~Ns; a
///     single transient None reaches only 1 and then resets.
///
/// Isolated from the async watchdog so the exact debounced predicate is
/// tested deterministically with zero process/timing flakiness.
fn parent_died(
    original: u32,
    current: Option<u32>,
    consecutive_failures: u32,
) -> (bool, u32) {
    let condition = match current {
        Some(p) => p != original,
        None => true,
    };
    if condition {
        let n = consecutive_failures.saturating_add(1);
        (n >= PARENT_DEATH_DEBOUNCE_TICKS, n)
    } else {
        // Healthy tick — reset; a recovered blip must not accumulate.
        (false, 0)
    }
}

/// JSON-RPC 2.0 request
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP Server that handles JSON-RPC over stdio
/// Uses async I/O to prevent blocking and support keep-alive
pub struct McpServer {}

impl McpServer {
    pub fn new() -> Self {
        Self {}
    }

    /// Start the MCP server loop reading from stdin using async I/O
    /// This prevents blocking and allows the server to stay alive
    /// Handler is now async to avoid blocking the runtime during tool calls
    #[allow(dead_code)]
    pub async fn run<F, Fut>(&self, handler: F) -> Result<()>
    where
        F: Fn(JsonRpcRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<JsonRpcResponse>> + Send,
    {
        // Create a dummy cancellation token that's never cancelled
        // This allows the old API to work while we migrate to run_with_shutdown
        let dummy_token = CancellationToken::new();
        self.run_with_shutdown(handler, dummy_token).await
    }

    /// Start the MCP server loop with graceful shutdown support and CONCURRENT request handling.
    ///
    /// Architecture:
    /// - Dedicated blocking thread reads stdin lines (prevents macOS UE state)
    /// - Main loop parses requests and spawns a tokio task per request
    /// - Each task runs the handler and sends the response to a channel
    /// - Dedicated writer task serializes responses to stdout
    ///
    /// This means slow tool calls (5-30s) don't block other requests (pings, list_tools, etc.)
    pub async fn run_with_shutdown<F, Fut>(
        &self,
        handler: F,
        shutdown: CancellationToken,
    ) -> Result<()>
    where
        F: Fn(JsonRpcRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<JsonRpcResponse>> + Send,
    {
        // Wrap handler in Arc so it can be shared across spawned tasks
        let handler = Arc::new(handler);

        // Spawn a dedicated blocking thread for stdin reading.
        // This prevents macOS UE (Uninterruptible Event) state that occurs
        // when tokio::io::stdin() is used - the process becomes unkillable.
        let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<String>();

        std::thread::spawn(move || {
            let stdin = io::stdin();
            let reader = stdin.lock();
            let mut buf_reader = io::BufReader::new(reader);
            let mut line = String::new();

            loop {
                line.clear();
                match buf_reader.read_line(&mut line) {
                    Ok(0) => {
                        // EOF - stdin closed
                        tracing::info!("[MCP] stdin thread: EOF detected");
                        break;
                    }
                    Ok(_) => {
                        if stdin_tx.send(line.clone()).is_err() {
                            // Receiver dropped, async side shut down
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("[MCP] stdin thread: read error: {}", e);
                        break;
                    }
                }
            }
        });

        // === BUG#2 fix: parent-death watchdog (the canonical orphan fix) ===
        //
        // Campaign E2E proved a stdio `umb` survived 35+ min after its parent
        // (`timeout 15 umb`) died: when an MCP client spawns `umb` (stdio) and
        // then dies/crashes WITHOUT cleanly closing the pipe — or the pipe's
        // write end is held by an unrelated still-living process — stdin EOF is
        // NEVER delivered, the blocking stdin reader parks forever, the channel
        // never closes, and the loop falls through only to the 10-MINUTE
        // inactivity timeout. A 10-minute orphan IS the project-killing bug.
        //
        // The portable fix (blueprint S3 "parent-death tie", explicitly "no
        // prctl/kqueue"): record the original parent PID at startup and poll
        // it. When the client/parent dies, this process is reparented (to
        // `init`/pid 1 or a subreaper) so the parent PID CHANGES — detect that
        // and trigger the existing shutdown token, exiting promptly and
        // cleanly (the loop's `shutdown.cancelled()` arm returns Ok(())). Uses
        // the existing `sysinfo` dep (same pattern as
        // `registry::parent_pid_of`); no new dependency, blueprint §6 honoured.
        //
        // SIGHUP is NOT relied upon (the blueprint notes it is never delivered
        // here); EOF handling is preserved and still the fast path when the
        // client does close stdin. This watchdog is the safety net for the
        // no-EOF orphan case and bounds orphan lifetime to ~1s.
        let parent_watch_shutdown = shutdown.clone();
        tokio::spawn(async move {
            fn parent_pid_now() -> Option<u32> {
                use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
                let sys = System::new_with_specifics(
                    RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
                );
                sys.process(Pid::from_u32(std::process::id()))
                    .and_then(|p| p.parent())
                    .map(|p| p.as_u32())
            }

            // Original parent at startup (the spawning MCP client). If we
            // cannot resolve it, skip the watchdog rather than risk a false
            // positive — EOF + inactivity timeout remain as fallbacks.
            let original_parent = match parent_pid_now() {
                Some(p) => p,
                None => {
                    tracing::debug!(
                        "[MCP] parent-death watchdog: parent pid unresolved; \
                         relying on EOF/inactivity fallbacks"
                    );
                    return;
                }
            };
            tracing::debug!(
                "[MCP] parent-death watchdog armed (original parent pid={})",
                original_parent
            );

            let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Residual 2b: count of consecutive ticks on which the
            // death/unresolvable condition has held. A genuine death keeps
            // this climbing to N; a single transient `None` bumps it to 1 and
            // the next (healthy) tick resets it to 0 — so a one-off sysinfo
            // `/proc/self` read race can no longer shut down a healthy server.
            let mut consecutive_failures: u32 = 0;
            loop {
                tokio::select! {
                    biased;
                    _ = parent_watch_shutdown.cancelled() => break,
                    _ = tick.tick() => {
                        // Attempt #8 zombie duty (mandatory for a child
                        // subreaper): on this EXISTING low-freq ~1s tick,
                        // drain any adopted orphan that has ALREADY EXITED so
                        // it never lingers as a `<defunct>` zombie. It
                        // discovers the adopted set (ppid==our pid AND exited
                        // AND NOT a tracked child AND NOT an allowlisted
                        // probe) and targeted-`waitpid(pid, WNOHANG)`s only
                        // those — NEVER the `-1` wildcard, so a legitimate
                        // tokio-owned `Child` PID can never be reaped here.
                        //
                        // C2: the discovery half is a synchronous `sysinfo`
                        // /proc scan; run it on `spawn_blocking` (NOT inline)
                        // so a slow scan can never stall this async watchdog
                        // tick. Fire-and-forget: the result is only a debug
                        // count; the scan is idempotent + self-bounded, and
                        // the targeted-`waitpid` semantics are unchanged
                        // (never -1, never a tracked/probe pid). No new
                        // task/timer beyond a short-lived blocking job.
                        let _ = tokio::task::spawn_blocking(|| {
                            crate::server::router::sweep_and_reap_adopted_zombies()
                        });

                        let current = parent_pid_now();
                        // Single source of truth: the pure, unit-tested
                        // DEBOUNCED predicate. Parent changed (reparented to
                        // init/subreaper) or unresolvable ⇒ candidate orphan;
                        // we only conclude death after N consecutive bad
                        // ticks (~Ns), never on a 1-tick blip.
                        let (dead, next) =
                            parent_died(original_parent, current, consecutive_failures);
                        consecutive_failures = next;
                        if dead {
                            tracing::info!(
                                "[MCP] parent process died (parent pid {} -> \
                                 {:?}; confirmed over {} consecutive ticks); \
                                 shutting down to prevent orphan",
                                original_parent, current, consecutive_failures
                            );
                            parent_watch_shutdown.cancel();
                            break;
                        }
                    }
                }
            }
        });

        // Response channel: spawned handler tasks send responses here
        let (resp_tx, mut resp_rx) = mpsc::unbounded_channel::<JsonRpcResponse>();

        // Dedicated stdout writer task - serializes all responses through one writer
        let writer_shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = writer_shutdown.cancelled() => break,
                    resp = resp_rx.recv() => {
                        match resp {
                            Some(response) => {
                                match serde_json::to_string(&response) {
                                    Ok(json) => {
                                        // stdout write must be atomic - use a single write
                                        let line = format!("{}\n", json);
                                        let stdout = io::stdout();
                                        let mut handle = stdout.lock();
                                        if let Err(e) = handle.write_all(line.as_bytes()) {
                                            tracing::error!("[MCP] Failed to write response: {}", e);
                                        }
                                        let _ = handle.flush();
                                    }
                                    Err(e) => {
                                        tracing::error!("[MCP] Failed to serialize response: {}", e);
                                    }
                                }
                            }
                            None => break, // Channel closed
                        }
                    }
                }
            }
        });

        loop {
            // Use tokio::select! to race between:
            // 1. Shutdown signal (CancellationToken)
            // 2. Line from stdin blocking thread (via mpsc channel)
            // 3. Inactivity timeout
            let line = tokio::select! {
                // Check for shutdown first (biased)
                biased;

                _ = shutdown.cancelled() => {
                    tracing::info!("[MCP] Shutdown requested, exiting gracefully");
                    break;
                }

                // Read from stdin via blocking thread channel, with inactivity timeout
                result = timeout(INACTIVITY_TIMEOUT, stdin_rx.recv()) => {
                    match result {
                        Ok(Some(line)) => line,
                        Ok(None) => {
                            // Channel closed - stdin thread exited (EOF or error).
                            // BUG#2: must break (not bare return) so the token
                            // is cancelled below and EVERY token-aware
                            // background task (writer, parent-watchdog) tears
                            // down — a bare return left them running and the
                            // process hung in runtime teardown (orphan).
                            tracing::info!("[MCP] stdin closed (EOF), shutting down gracefully");
                            break;
                        }
                        Err(_) => {
                            // Timeout - no activity for 10 minutes
                            tracing::info!("[MCP] No activity for 10 minutes, shutting down to prevent orphan process");
                            break;
                        }
                    }
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(trimmed) {
                Ok(req) => req,
                Err(e) => {
                    tracing::debug!("[MCP] Parse error for line: {}", trimmed);
                    let error_response = JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {}", e),
                            data: None,
                        }),
                    };
                    let _ = resp_tx.send(error_response);
                    continue;
                }
            };

            // Log the method being called (debug level)
            tracing::debug!("[MCP] Handling method: {}", request.method);

            // Check if this is a notification (no id) - per JSON-RPC 2.0 spec,
            // notifications MUST NOT receive a response
            let is_notification = request.id.is_none();

            // Save request.id before handler consumes the request
            let request_id = request.id.clone();

            // Spawn a task for each request so slow tool calls don't block others
            let handler_clone = Arc::clone(&handler);
            let resp_tx_clone = resp_tx.clone();

            tokio::spawn(async move {
                let response = handler_clone(request).await;

                // Don't send response for notifications (JSON-RPC 2.0 spec compliance)
                if is_notification {
                    tracing::debug!("[MCP] Notification processed, no response sent");
                    return;
                }

                match response {
                    Ok(resp) => {
                        let _ = resp_tx_clone.send(resp);
                    }
                    Err(e) => {
                        tracing::error!("[MCP] Handler error: {}", e);
                        let error_response = JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request_id, // Preserve the original request id
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32603,
                                message: format!("Internal error: {}", e),
                                data: None,
                            }),
                        };
                        let _ = resp_tx_clone.send(error_response);
                    }
                }
            });
        }

        // === BUG#2: tear down ALL token-aware background work on exit ===
        // The serve loop exited (EOF / inactivity / shutdown / parent-death).
        // Cancel the token so the stdout writer task and the parent-death
        // watchdog stop immediately instead of pinning the tokio runtime.
        // (The hot-swap `notify` watcher runs on a non-tokio OS thread that
        // does not observe this token — the caller force-exits the process,
        // see `async_main`, so that thread cannot keep the process alive and
        // re-create the original 35-min orphan.)
        shutdown.cancel();
        Ok(())
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self {}
    }
}

/// Helper to create success response
pub fn success_response(id: Option<Value>, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    }
}

/// Helper to create error response
pub fn error_response(id: Option<Value>, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message,
            data: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: "tools/list".to_string(),
            params: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn test_response_helpers() {
        let resp = success_response(Some(Value::Number(1.into())), serde_json::json!({"status": "ok"}));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());

        let err_resp = error_response(Some(Value::Number(1.into())), -32601, "Method not found".to_string());
        assert!(err_resp.result.is_none());
        assert!(err_resp.error.is_some());
    }

    /// BUG#2 regression (the literal project-killer), deterministic core:
    /// the parent-death predicate that drives the stdio orphan-prevention
    /// watchdog. Pre-fix the stdio server had NO parent-death tie and a stdio
    /// `umb` survived 35+ min orphaned after its parent died. The async
    /// watchdog calls EXACTLY this `parent_died()` (single source of truth),
    /// so asserting its decision table here genuinely guards the fix with zero
    /// timing flakiness. Real end-to-end EOF/orphan exit is covered by the
    /// integration test `tests/stdio_eof.rs` (drives the built binary).
    #[test]
    fn test_parent_died_predicate_drives_orphan_shutdown() {
        // Healthy: parent unchanged ⇒ NOT dead, counter stays/resets to 0.
        assert_eq!(
            parent_died(1000, Some(1000), 0),
            (false, 0),
            "unchanged parent must NOT trigger orphan shutdown"
        );
        // Sustained genuine death persists across ticks; the watchdog feeds
        // the running counter back in. The exact case that left the 35-min
        // orphan (`timeout 15 umb`, parent died → reparented to init).
        let (d1, c1) = parent_died(1000, Some(1), 0);
        assert!(!d1, "1st reparented tick must NOT fire (debounced)");
        let (d2, c2) = parent_died(1000, Some(1), c1);
        assert!(!d2, "2nd reparented tick must NOT fire (debounced)");
        let (d3, c3) = parent_died(1000, Some(1), c2);
        assert!(
            d3 && c3 == PARENT_DEATH_DEBOUNCE_TICKS,
            "reparented-to-init MUST trigger shutdown on the Nth consecutive tick"
        );
        // Any sustained parent-pid change confirms after N ticks too.
        let mut c = 0;
        let mut fired = false;
        for _ in 0..PARENT_DEATH_DEBOUNCE_TICKS {
            let (d, n) = parent_died(1000, Some(4242), c);
            c = n;
            fired = d;
        }
        assert!(fired, "any sustained parent-pid change MUST trigger shutdown");
    }

    /// Residual 2b: the debounce specifically rejects a single transient
    /// `parent_pid_now() == None` (a one-off sysinfo `/proc/self` read race)
    /// while still detecting a genuine, sustained parent death — and resets
    /// on recovery so blips never accumulate toward a false shutdown.
    #[test]
    fn test_parent_death_debounce_rejects_transient_none() {
        // 1 transient None ⇒ NOT dead (pre-fix this shut down a healthy
        // server on a single read-race blip — the false-positive vector).
        let (dead, count) = parent_died(1000, None, 0);
        assert!(!dead, "a single transient None must NOT conclude parent death");
        assert_eq!(count, 1, "one bad tick counted, below the threshold");

        // Recovery: None blip then a valid/healthy read ⇒ counter RESETS to
        // 0, so isolated blips can never accumulate to N over time.
        let (dead2, count2) = parent_died(1000, Some(1000), count);
        assert!(!dead2, "recovered tick must NOT be dead");
        assert_eq!(count2, 0, "a healthy tick must RESET the failure counter");

        // N consecutive None (a sustained unresolvable parent) ⇒ dead, on the
        // Nth tick exactly (fail-safe toward exit on *sustained* ambiguity).
        let mut c = 0;
        for tick in 1..=PARENT_DEATH_DEBOUNCE_TICKS {
            let (dead_n, n) = parent_died(1000, None, c);
            c = n;
            if tick < PARENT_DEATH_DEBOUNCE_TICKS {
                assert!(!dead_n, "must NOT fire before N consecutive bad ticks");
            } else {
                assert!(
                    dead_n,
                    "N consecutive None MUST conclude parent death (fail-safe)"
                );
                assert_eq!(n, PARENT_DEATH_DEBOUNCE_TICKS);
            }
        }

        // Interleaved blips (None, healthy, None, healthy …) never reach N:
        // each healthy tick zeroes the counter, so a flapping reader cannot
        // false-trigger a shutdown of a genuinely-attached server.
        let mut c2 = 0;
        for _ in 0..10 {
            let (d_none, n1) = parent_died(1000, None, c2);
            assert!(!d_none, "isolated None never fires");
            let (d_ok, n2) = parent_died(1000, Some(1000), n1);
            assert!(!d_ok, "recovery never fires");
            c2 = n2;
            assert_eq!(c2, 0, "counter must keep resetting on recovery");
        }
    }
}
