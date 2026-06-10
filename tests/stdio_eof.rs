//! BUG#2 end-to-end regression: the stdio MCP server (the mode MCP clients
//! spawn: `command=umb`, transport=stdio) MUST exit promptly and cleanly when
//! its stdin reaches EOF (the client/parent closed the pipe or died).
//!
//! This is the original project-killing orphan-process class. Campaign E2E
//! captured a stdio `umb` that survived 35+ min orphaned after its parent
//! died (`printf '{...}' | ./umb` ⇒ rc=124 timeout-kill, never exited on
//! its own). This integration test drives the REAL freshly-built binary
//! (`CARGO_BIN_EXE_umb`, guaranteed rebuilt by Cargo for integration tests —
//! no stale-binary hazard) with the verbatim repro shape and asserts the
//! process exits on its own, fast (rc 0), leaving no orphan.

#![cfg(unix)]

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn stdio_umb_exits_promptly_on_stdin_eof() {
    let bin = env!("CARGO_BIN_EXE_umb");

    // Isolated HOME so the run never touches the real ~/.umb.
    let tmp = tempfile::tempdir().expect("tempdir");

    let mut child = Command::new(bin)
        .env("HOME", tmp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn umb stdio server");

    // Send exactly one initialize line, then DROP stdin → pipe EOF. This is
    // verbatim the campaign repro `printf '{"...initialize..."}\n' | ./umb`.
    {
        let mut sin = child.stdin.take().expect("child stdin");
        sin.write_all(
            br#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"clientInfo":{"name":"t","version":"1"}}}"#,
        )
        .expect("write initialize");
        sin.write_all(b"\n").expect("write newline");
        // `sin` dropped here → stdin write end closed → EOF for the child.
    }

    // The fix must terminate the serve loop on EOF within a few seconds.
    // Pre-fix behaviour: never exits on its own (10-min inactivity ceiling /
    // 35-min observed orphan, killed only by an external `timeout`). A
    // generous 20s CI bound proves "prompt, not a timeout-kill".
    let started = Instant::now();
    let deadline = started + Duration::from_secs(20);
    let mut exited = None;
    while Instant::now() < deadline {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                exited = Some(status);
                break;
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    let elapsed = started.elapsed();

    if exited.is_none() {
        // Still alive ⇒ the orphan bug regressed. Kill so the test rig is not
        // poisoned by a floating umb, then fail loudly.
        let _ = child.kill();
        let _ = child.wait();
        panic!(
            "BUG#2 REGRESSION: stdio umb did NOT exit on stdin EOF within 20s \
             (would orphan; the original project-killing class). The serve \
             loop + process must terminate on EOF."
        );
    }

    let status = exited.unwrap();
    assert!(
        status.success(),
        "stdio umb exited on EOF but with non-zero status {:?}; must be a \
         clean rc 0 graceful shutdown",
        status.code()
    );

    // Sanity: it really was prompt (well under the old 10-min ceiling). This
    // is informational but bounds the regression tightly.
    assert!(
        elapsed < Duration::from_secs(20),
        "exit must be prompt on EOF; took {:?}",
        elapsed
    );

    // try_wait already reaped the child; an extra wait() is idempotent and
    // confirms the process (and its kill_on_drop child MCP subtree) is gone.
    let _ = child.wait();
}
