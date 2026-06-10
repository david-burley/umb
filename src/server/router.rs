use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
// `(server, name)`-keyed tool registry (server-keyed feature): IndexMap
// preserves insertion order globally, so the deterministic discovery
// order established by P3 (server-grouped) and the registration order
// from discovery.rs both remain stable. Single map, single key type.
use indexmap::IndexMap;
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;
// `parking_lot::RwLock` is used ONLY by the `embed-onnx`-gated
// `semantic_search` field + its setters; feature-gate the import to match
// so the default (keyword-only) build is warning-clean. Behaviour-neutral
// on both feature sets.
#[cfg(feature = "embed-onnx")]
use parking_lot::RwLock;
use tokio::process::Command as TokioCommand;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader};
use reqwest::blocking::Client;
use reqwest::Client as AsyncClient;

// Semantic-search engine types are opt-in behind `embed-onnx` (§7.1). With the
// feature off the router runs keyword-only and these types are not compiled.
#[cfg(feature = "embed-onnx")]
use crate::features::semantic_search::{SemanticSearchProvider, SearchResult};

/// Residual 2a — process-global registry of pooled child MCP-server PIDs.
///
/// The stdio serve path ends with `std::process::exit(0)` in `main.rs`
/// (load-bearing: it sidesteps BUG#2 — the non-tokio `notify` hot-swap
/// watcher thread blocks tokio `Runtime` drop, hanging the process alive as
/// the original 35-min orphan). But `exit(0)` skips every `Drop`, so the
/// `kill_on_drop(true)` set on pooled children (see `spawn_and_handshake`)
/// does NOT fire — the OS merely *reparents* those Node/Python MCP children
/// to init, it does NOT reap them. Net: we would trade an orphaned `umb` for
/// orphaned children (same project-killing class, one level down).
///
/// Fix: every pooled child PID is registered here at spawn and unregistered
/// when its connection is explicitly killed/reaped. Before `exit(0)`,
/// `main.rs` calls `reap_tracked_children()` to synchronously and
/// best-effort terminate whatever is still tracked. No new dependency
/// (reuses `sysinfo`, already the watchdog's mechanism), bounded so it can
/// never hang the exit.
///
/// Residual 1e: we track `(pid, (start_time, ident))` — NOT a bare pid, and
/// NOT pid+start_time alone — so the pre-exit reaper can re-validate each
/// entry against the *live* process at that pid immediately before signalling
/// it, fully mirroring `doctor.rs::revalidate_for_kill`. That fn requires the
/// live process to satisfy BOTH an identity predicate
/// (`is_umb_daemon_proc(cmd, exe_basename)`) AND `start_time == expected`;
/// start_time alone is only ~1s-granular, so a same-pid + same-1s-bucket
/// recycle could otherwise slip through. `ident` is the live child's
/// `(exe_basename, cmd)` captured at track time from the SAME sysinfo
/// snapshot already taken for `start_time` (no extra snapshot). A stale entry
/// whose pid has since been recycled to an unrelated process now fails on
/// EITHER predicate and is SKIPPED, making a destructive mis-kill of a
/// bystander — including the same-second PID-recycle case — structurally
/// impossible even if an entry ever leaked.
///
/// `ChildIdent` is exactly the pair `doctor.rs:revalidate_for_kill` derives
/// from the live process for its identity check (`exe_basename(p.exe())` and
/// `p.cmd()`); we compare it for *equality* against the value recorded at
/// track time (the MCP-child analogue of doctor.rs's `is_umb_daemon_proc`
/// match — a pooled Node/Python child is not a umb daemon, so the correct
/// identity assertion is "still the very process we spawned", i.e. identical
/// exe basename + argv).
type ChildIdent = (Option<String>, Vec<String>);
static TRACKED_CHILD_PIDS: OnceLock<Mutex<HashMap<u32, (u64, ChildIdent)>>> =
    OnceLock::new();

fn tracked_child_pids() -> &'static Mutex<HashMap<u32, (u64, ChildIdent)>> {
    TRACKED_CHILD_PIDS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Live child's exe basename — the MCP-child analogue of
/// `doctor.rs::exe_basename(p.exe())` (same `file_name().to_string_lossy()`
/// derivation). Kept identical so the recorded ident and the re-validation
/// ident are derived the exact same way.
fn child_exe_basename(exe: Option<&std::path::Path>) -> Option<String> {
    exe.and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
}

/// Resolve a freshly-spawned child's `(start_time, ident)` via sysinfo so it
/// can be recorded alongside the pid for later PID-reuse re-validation. The
/// ident is `(exe_basename, cmd)` — exactly the pair
/// `doctor.rs::revalidate_for_kill` reads from the live process for its
/// identity check — captured here from the SAME single snapshot already used
/// for `start_time` (NO second snapshot). Returns `None` if the process is
/// already gone / not yet visible — the caller then does not track it (an
/// entry we cannot re-validate is worthless and a mis-kill hazard, so we
/// prefer to not track over tracking blindly; this also covers the
/// ident-unresolvable race — a track entry always carries a revalidatable
/// ident or is not created at all).
fn resolve_child_start_time(pid: u32) -> Option<(u64, ChildIdent)> {
    use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
    );
    sys.process(Pid::from_u32(pid)).map(|p| {
        let ident: ChildIdent =
            (child_exe_basename(p.exe()), p.cmd().to_vec());
        (p.start_time(), ident)
    })
}

fn track_child_pid(pid: u32, start_time: u64, ident: ChildIdent) {
    if let Ok(mut map) = tracked_child_pids().lock() {
        map.insert(pid, (start_time, ident));
    }
}

fn untrack_child_pid(pid: u32) {
    if let Ok(mut map) = tracked_child_pids().lock() {
        map.remove(&pid);
    }
}

/// C1 — positive allowlist of UMB-spawned short-lived DISCOVERY/HOT-SWAP
/// PROBE pids (`discover_tools_from_server`'s ephemeral
/// `Command::new(...).spawn()` around the `tools/list` handshake). Unlike a
/// pooled child it is intentionally NOT pgid-led and NOT in
/// `TRACKED_CHILD_PIDS`, yet it IS `ppid == umb` — so without this allowlist
/// the adopted-orphan sweep in `reap_tracked_children()` could SIGKILL an
/// in-flight probe that races teardown/hot-swap.
///
/// This is a POSITIVE allowlist (the set of pids UMB ITSELF spawned as
/// probes), NOT a heuristic on process name/argv. A genuine
/// setsid+double-fork adopted orphan is, by construction, never inserted
/// here (only `ProbePidGuard::new`, called exclusively right after our own
/// probe `.spawn()`, inserts), so it is still swept and killed — the A2
/// guarantee is preserved. Same spirit/shape as `TRACKED_CHILD_PIDS`: a
/// process-global set, RAII-registered at spawn, RAII-unregistered on
/// probe completion / error / kill (every exit path, including the ~8 `?`
/// early returns in `discover_tools_from_server`).
static PROBE_PIDS: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();

fn probe_pids() -> &'static Mutex<HashSet<u32>> {
    PROBE_PIDS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Snapshot of the current probe allowlist (pids UMB spawned as discovery/
/// hot-swap probes). Consulted by every adopted-orphan sweep so a
/// legitimate in-flight probe is treated exactly like a tracked child
/// (excluded from the kill). Poison-tolerant: a poisoned lock yields an
/// empty set (fail toward NOT over-excluding — we never want a poisoned
/// lock to silently spare a real orphan; but a poisoned probe lock is
/// itself near-impossible as the critical sections are tiny + panic-free).
fn probe_pid_allowlist() -> HashSet<u32> {
    match probe_pids().lock() {
        Ok(s) => s.clone(),
        Err(e) => e.into_inner().clone(),
    }
}

/// RAII guard mirroring `TrackedChildGuard`: registers a probe pid in the
/// positive allowlist on construction (right after the probe `.spawn()`),
/// unregisters it on `Drop` — which fires on the normal return, on ANY of
/// `discover_tools_from_server`'s `?` early returns, AND after the explicit
/// `child.kill()`/`wait()`. Net invariant: a pid is in `PROBE_PIDS` iff a
/// live UMB-spawned probe with that pid may still exist, so the window in
/// which the sweep spares it is exactly the probe's lifetime — never longer
/// (so a recycled pid cannot stay spuriously allowlisted).
#[must_use = "dropping the ProbePidGuard unregisters the probe pid; hold it for the probe's whole lifetime"]
struct ProbePidGuard {
    pid: u32,
}

impl ProbePidGuard {
    fn new(pid: u32) -> Self {
        if let Ok(mut s) = probe_pids().lock() {
            s.insert(pid);
        } else if let Ok(mut s) = probe_pids().lock().map_err(|e| e.into_inner()) {
            s.insert(pid);
        }
        Self { pid }
    }
}

impl Drop for ProbePidGuard {
    fn drop(&mut self) {
        match probe_pids().lock() {
            Ok(mut s) => {
                s.remove(&self.pid);
            }
            Err(e) => {
                e.into_inner().remove(&self.pid);
            }
        }
    }
}

/// Residual A2 — canonical race-free process-group reap.
///
/// `kill_on_drop(true)` (tokio) and `sysinfo`'s per-pid `kill_with`/`kill`
/// SIGKILL ONLY the direct tracked child. A pooled MCP server that itself
/// spawns a GRANDCHILD (e.g. `node` → a worker, or `sh -c 'sleep & wait'`)
/// leaves that grandchild reparented to init the instant the direct child
/// dies — the project-killing orphan class, one level deeper (A2: every
/// stdio teardown cycle leaked one `sleep 3600` grandchild monotonically).
/// The 1d parent-link BFS cannot fix this: by the time the pre-exit reaper
/// runs, grandchildren orphaned in *earlier* cycles already have ppid=1, so
/// a downward walk from the tracked child pid can never reach them
/// (kill-then-walk reparent race).
///
/// The race-free fix is process groups: every pooled child is spawned as its
/// OWN process-group leader (`process_group(0)` ⇒ pgid == child pid; see
/// `spawn_and_handshake`). Reaping then signals the *negative pgid* — the
/// kernel delivers the signal to the leader AND every descendant in that
/// group atomically, regardless of any reparenting that already happened.
/// One `kill(-pgid, …)` reaps child + grandchild + great-grandchild in a
/// single shot, immune to the reparent race the BFS lost to.
///
/// No new dependency: `kill(2)` is declared here as a one-line `extern "C"`
/// (std-only FFI; `libc`/`nix` are only *transitive* deps, not direct, and
/// must not be promoted). Returns `Ok` on success, `Err(errno)` otherwise.
#[cfg(unix)]
extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

/// Attempt #8 — PR_SET_CHILD_SUBREAPER. `prctl(2)` and `waitpid(2)` are
/// declared as one-line `extern "C"` exactly like `kill` above (same std-only
/// FFI convention; `libc` is NOT a direct dep — only transitive — and the
/// in-tree precedent is the `kill` decl, so this follows it rather than
/// promoting a dependency). Linux-only call sites are `#[cfg(target_os =
/// "linux")]`; the decls themselves are harmless on any Unix (only ever
/// *called* under the Linux gate). `prctl` is variadic in C but
/// PR_SET_CHILD_SUBREAPER only ever reads the first `unsigned long` arg, so a
/// fixed 5-arg `c_long` signature is ABI-correct for this single use.
#[cfg(target_os = "linux")]
extern "C" {
    fn prctl(
        option: i32,
        arg2: std::os::raw::c_long,
        arg3: std::os::raw::c_long,
        arg4: std::os::raw::c_long,
        arg5: std::os::raw::c_long,
    ) -> i32;
}

/// `waitpid(2)` for the non-blocking adopted-orphan zombie reap. Declared
/// std-only `extern "C"` per the in-tree `kill` precedent (no `libc`
/// promotion). Called ONLY with a SPECIFIC positive `pid` (never the `-1`
/// "any child" wildcard) plus `WNOHANG`, so it is non-blocking AND can only
/// ever reap a pid we positively identified as an adopted orphan — a
/// tokio/std-owned `Child` pid is, by construction, never passed here.
/// `status` is an out-param we discard (we only need the reap, not the exit
/// code).
#[cfg(target_os = "linux")]
extern "C" {
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
}

/// `prctl(2)` option number for "make this process a child subreaper" (from
/// `<linux/prctl.h>`; stable kernel ABI since 3.4). Not exposed by std, so
/// the numeric constant is inlined (same approach as `SIGTERM`/`SIGKILL`).
#[cfg(target_os = "linux")]
const PR_SET_CHILD_SUBREAPER: i32 = 36;

/// `waitpid(2)` `WNOHANG`: return immediately (0) instead of blocking if no
/// child has changed state. Stable POSIX/Linux ABI value `1`.
#[cfg(target_os = "linux")]
const WNOHANG: i32 = 1;

#[cfg(unix)]
const SIGTERM: i32 = 15;
#[cfg(unix)]
const SIGKILL: i32 = 9;
/// `0` is the POSIX "signal validity probe" — delivers no signal, only
/// reports whether the target (here, a process group) still has any member.
#[cfg(unix)]
const SIG_PROBE: i32 = 0;

/// Attempt #8 — mark THIS process (umb) as a child subreaper so the kernel
/// reparents ANY orphaned descendant — including a `setsid()`+double-fork
/// daemonized grandchild whose intermediate parent dies — to umb (the
/// nearest live subreaper) instead of to init/pid 1. The daemonized orphan
/// thereby becomes umb's OWN direct child (`ppid == umb pid`), so the
/// `ppid == std::process::id()` sweep in `reap_tracked_children()` can see
/// and kill it. This is the SAME mechanism `tini`/init systems use to own
/// their whole subtree. It is the structural fix for the A2 residual that no
/// process-tree walk can solve: a double-fork+setsid grandchild severs its
/// PPID link at BIRTH, so by teardown it was never a descendant of the
/// pooled child and is invisible to any descendant enumeration.
///
/// Best-effort: on `prctl` failure (older/locked-down kernel, non-Linux) we
/// warn and continue — behaviour simply degrades to the prior (pgid-only)
/// reap. Idempotent. Call ONCE, EARLY at startup, BEFORE any backend/pool
/// spawn so every subsequently-spawned subtree is owned.
pub fn install_child_subreaper() {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `prctl` is async-signal-safe; PR_SET_CHILD_SUBREAPER reads
        // only arg2 (the flag 1) and ignores arg3..arg5 (passed 0). No
        // Rust-side aliasing — all scalar args.
        let rc = unsafe { prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) };
        if rc == 0 {
            tracing::debug!(
                "[subreaper] PR_SET_CHILD_SUBREAPER set (pid={}); orphaned \
                 descendants now reparent to umb, not init",
                std::process::id()
            );
        } else {
            tracing::warn!(
                "[subreaper] prctl(PR_SET_CHILD_SUBREAPER) failed (rc={}); \
                 continuing — adopted-orphan reap degrades to pgid-only \
                 (prior behaviour, no abort)",
                rc
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        tracing::debug!(
            "[subreaper] PR_SET_CHILD_SUBREAPER is Linux-only — no-op on this \
             platform (pgid reap + kill_on_drop still active)"
        );
    }
}

/// Non-blocking zombie-reaping duty (mandatory once we are a child
/// subreaper): a subreaper becomes responsible for the *exit status* of
/// every descendant the kernel reparents to it. An adopted orphan that then
/// exits would otherwise linger as a `<defunct>` zombie forever.
///
/// CRITICAL — TARGETED, NEVER `waitpid(-1)`: design point 2 requires this to
/// "only mop up reparented/adopted PIDs **not owned by a tokio Child**". A
/// broad `waitpid(-1, WNOHANG)` cannot make that distinction — it would reap
/// ANY exited child of this process, including a legitimate `std::process`/
/// tokio `Child` PID before its owner calls `.wait()` (proven: the broad
/// form destroyed sibling tests' bystander `Child`s across the shared test
/// binary). So we waitpid ONLY the explicit `adopted` PID set the caller
/// positively identified as adopted orphans (ppid == our pid AND NOT in the
/// tracked/owned set). Each `waitpid(pid, WNOHANG)` is per-PID and
/// non-blocking: rc>0 ⇒ that exact orphan was reaped; rc==0 ⇒ still running
/// (skip); rc==-1/ECHILD ⇒ already reaped by its owner / not our child
/// (skip). It NEVER blocks, NEVER busy-loops (one bounded pass over a finite
/// caller-supplied set), and can NEVER touch a tokio-owned `Child` because
/// such a PID is, by construction, never in the adopted set.
///
/// Returns the number of orphans actually reaped (for the debug log / test).
#[cfg(target_os = "linux")]
pub fn reap_adopted_zombies(adopted: &HashSet<u32>) -> usize {
    let mut reaped = 0usize;
    for &pid in adopted {
        if pid <= 1 {
            continue; // never waitpid(-1)/0/1 — the catastrophic wildcards
        }
        let mut status: i32 = 0;
        // SAFETY: `waitpid` on a SPECIFIC positive pid + WNOHANG is
        // non-blocking and async-signal-safe; `status` is a valid local
        // out-pointer. Targeted (never the -1 wildcard) so a tokio/std
        // `Child` PID — which is by construction never in `adopted` — can
        // not be reaped here even by accident.
        let rc = unsafe { waitpid(pid as i32, &mut status as *mut i32, WNOHANG) };
        if rc > 0 {
            reaped += 1;
        }
        // rc == 0  ⇒ still running (not yet a zombie) — leave it
        // rc == -1 ⇒ ECHILD: not our child / already reaped — leave it
    }
    if reaped > 0 {
        tracing::debug!(
            "[subreaper] reaped {} adopted-orphan zombie(s) via targeted \
             waitpid(pid, WNOHANG) ({} candidate pid(s))",
            reaped,
            adopted.len()
        );
    }
    reaped
}

/// Non-Linux: no subreaper ⇒ nothing is reparented to us ⇒ nothing extra to
/// reap. Clean no-op so call sites stay platform-agnostic & build-green.
#[cfg(not(target_os = "linux"))]
pub fn reap_adopted_zombies(_adopted: &HashSet<u32>) -> usize {
    0
}

/// Periodic zombie-duty helper invoked from the parent-death-watchdog ~1s
/// cadence (C2: dispatched on `spawn_blocking`, NOT inline on the async
/// tick, so this synchronous `sysinfo`/proc scan can never stall the
/// watchdog). It discovers the CURRENT set of adopted orphans (processes
/// whose `ppid == our pid`) that have ALREADY exited (Zombie/Dead) and are
/// NOT a tracked pooled child AND NOT an allowlisted UMB discovery/hot-swap
/// probe (C1 positive allowlist), then targeted-`waitpid`s exactly those.
/// Self-contained analogue of the teardown path's explicit-set reap, for
/// the running server: it can NEVER reap a live legitimate pooled child or
/// an in-flight probe (excluded by the tracked-set / probe-allowlist
/// filters; and only Zombie/Dead pids are even considered) and NEVER uses
/// the `-1` wildcard — each `waitpid` is a specific positive pid. Bounded
/// (one process-table scan + one targeted waitpid per already-exited
/// adopted pid), non-blocking.
#[cfg(target_os = "linux")]
pub fn sweep_and_reap_adopted_zombies() -> usize {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessStatus, RefreshKind, System};
    let me = std::process::id();
    let tracked: HashSet<u32> = match tracked_child_pids().lock() {
        Ok(map) => map.keys().copied().collect(),
        Err(_) => return 0,
    };
    // C1: positive probe allowlist — never waitpid a live UMB discovery/
    // hot-swap probe (it is `ppid==me`, untracked; excluded like a tracked
    // child). A real adopted orphan is never allowlisted, so still reaped.
    let probe_set: HashSet<u32> = probe_pid_allowlist();
    let sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
    );
    let mut adopted: HashSet<u32> = HashSet::new();
    for (cpid, cproc) in sys.processes() {
        let cpid = cpid.as_u32();
        if cpid <= 1 || cpid == me || tracked.contains(&cpid) || probe_set.contains(&cpid) {
            continue;
        }
        if cproc.parent() == Some(Pid::from_u32(me)) {
            // Only ALREADY-EXITED adopted orphans are zombies needing a
            // reap; a still-running adopted orphan is left for the teardown
            // path (or the next tick) — we never waitpid a live process here.
            match cproc.status() {
                ProcessStatus::Zombie | ProcessStatus::Dead => {
                    adopted.insert(cpid);
                }
                _ => {}
            }
        }
    }
    reap_adopted_zombies(&adopted)
}

#[cfg(not(target_os = "linux"))]
pub fn sweep_and_reap_adopted_zombies() -> usize {
    0
}

/// Defense-in-depth boundary guard for the catastrophic-class `kill(-pgid,…)`
/// syscall. `kill(-pgid)` targets an ENTIRE process group; a degenerate `pgid`
/// turns it into a project-ending mistake:
///   * `pgid == 0`  ⇒ `kill(0, …)` signals the *caller's own* process group.
///   * `pgid == 1`  ⇒ `kill(-1, …)` signals *every process the uid can reach*.
///   * `pgid == std::process::id()` ⇒ umb's own group (every child is spawned
///     `process_group(0)` ⇒ pgid == child pid, so a legitimate child pgid is
///     ALWAYS `> 1` and NEVER == umb's pid; equality here means a corrupt /
///     recycled / zero pid slipped through upstream invariants).
///
/// Upstream (1e ident-revalidation + 1a guard) already make these structurally
/// unreachable, but this is a belt-and-suspenders ENFORCED boundary check so a
/// self-kill / `kill(-1)` / init-group signal is impossible by construction at
/// the syscall edge, not merely by invariant. Pure + side-effect free so it is
/// unit-testable in isolation. `true` ⇒ safe to signal this pgid.
#[cfg(unix)]
fn pgid_safe_to_signal(pgid: u32) -> bool {
    pgid > 1 && pgid != std::process::id()
}

/// Signal the ENTIRE process group led by `pgid` (child + all descendants,
/// regardless of reparenting). Negative pid ⇒ "process group" per `kill(2)`.
/// Best-effort and non-blocking: never waitpid/sleep here so it is safe on
/// the `Drop` path and the `exit(0)` critical path alike.
#[cfg(unix)]
fn signal_process_group(pgid: u32, sig: i32) -> bool {
    // Defense-in-depth: refuse the catastrophic-class `kill(-pgid)` for any
    // degenerate pgid (0 ⇒ own group, 1 ⇒ kill(-1), == own pid ⇒ self-group).
    // A legitimate child pgid is always > 1 and never == umb's own pid, so
    // this is a strict no-op for every real teardown and an enforced wall
    // (not a mere invariant) against self-/init-group annihilation.
    if !pgid_safe_to_signal(pgid) {
        tracing::warn!(
            "refusing kill(-pgid) for invalid/own pgid={} (own pid={}) — \
             boundary guard blocked catastrophic-class signal",
            pgid,
            std::process::id()
        );
        return false;
    }
    // SAFETY: `kill(2)` is async-signal-safe and has no Rust-side aliasing
    // concerns; a negative first arg is the documented process-group form.
    let rc = unsafe { kill(-(pgid as i32), sig) };
    rc == 0
}

/// True iff the process group led by `pgid` still has at least one live
/// member (POSIX `kill(-pgid, 0)`): rc 0 ⇒ group exists. Used only to gate
/// the SIGKILL escalation (skip it when the group already drained on
/// SIGTERM) — never to block.
#[cfg(unix)]
fn process_group_alive(pgid: u32) -> bool {
    signal_process_group(pgid, SIG_PROBE)
}

/// Attempt #8 — adopted-orphan ident-revalidation (the 1e gate, reused).
///
/// PR_SET_CHILD_SUBREAPER (see `install_child_subreaper`) reparents a
/// `setsid()`+double-fork daemonized orphan to umb, so by teardown it is
/// umb's OWN direct child (`ppid == std::process::id()`). The
/// `reap_tracked_children()` sweep finds those by ppid and SIGTERM/SIGKILLs
/// them. Before the DESTRUCTIVE SIGKILL of any swept orphan we re-validate
/// its identity exactly as the tracked-child 1e gate does — the live process
/// at `pid` must still match the `(exe_basename, cmd)` ident captured for it
/// during the same sweep — so a PID recycled (or vanished) between sweep and
/// kill, or any unrelated bystander, is NEVER force-killed. This is the SAME
/// `(child_exe_basename(p.exe()), p.cmd())` pair `reap_tracked_children`'s 1e
/// gate compares and the exact analogue of
/// `doctor.rs::revalidate_for_kill`'s identity predicate.
#[cfg(unix)]
fn adopted_pid_ident_still_matches(
    sys: &sysinfo::System,
    pid: u32,
    expected: &ChildIdent,
) -> bool {
    use sysinfo::Pid;
    match sys.process(Pid::from_u32(pid)) {
        Some(p) => {
            let live: ChildIdent = (child_exe_basename(p.exe()), p.cmd().to_vec());
            live == *expected
        }
        None => false,
    }
}

/// Residual 1a — RAII leak-guard mirroring `daemon::registry::RegistryGuard`
/// (same `active`-flag + explicit-disarm shape, the in-tree pattern). The
/// pool tracks a child's pid the instant after `.spawn()`, but the `Drop`
/// that *un*tracks lives on `ServerConnection`, which is only built AFTER ~8
/// `?` early-return points in `spawn_and_handshake`. Any of those early
/// returns previously leaked the tracked pid forever (the bare child itself
/// is already SIGKILLed by tokio `kill_on_drop(true)` on its own `Drop` — the
/// *leak* is purely the stale entry in `TRACKED_CHILD_PIDS`). A leaked entry
/// is exactly what residual 1e defends the reaper against, but the correct
/// fix is to not leak it in the first place: this guard untracks on `Drop`
/// (i.e. on ANY early return) unless explicitly disarmed once ownership has
/// transferred to a successfully-constructed `ServerConnection`.
///
/// Net invariant: a `(pid, (start_time, ident))` exists in
/// `TRACKED_CHILD_PIDS` iff a live pooled child it refers to exists.
#[must_use = "dropping the TrackedChildGuard untracks the pid; hold it until ServerConnection is built"]
struct TrackedChildGuard {
    pid: u32,
    /// When true, `Drop` untracks `pid`. Disarmed on successful handoff.
    active: bool,
}

impl TrackedChildGuard {
    /// Track `pid`/`(start_time, ident)` and return an armed guard. Mirrors
    /// `registry::register` returning a `RegistryGuard`.
    fn track(pid: u32, start_time: u64, ident: ChildIdent) -> Self {
        track_child_pid(pid, start_time, ident);
        Self { pid, active: true }
    }

    /// Ownership of the tracked pid has transferred to a constructed
    /// `ServerConnection` (whose own `Drop` untracks). Disarm so this guard's
    /// `Drop` becomes a no-op. Mirrors `RegistryGuard::remove_now` flipping
    /// `active` to false on the success/handoff path.
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for TrackedChildGuard {
    fn drop(&mut self) {
        if self.active {
            // Early-return path: untrack so the pre-exit reaper never chases
            // a pid whose pooled child failed to materialise (and which the
            // OS may recycle to an unrelated process).
            untrack_child_pid(self.pid);
        }
    }
}

/// Synchronously, best-effort terminate every still-tracked pooled child
/// MCP-server process: SIGTERM, a short bounded grace, then SIGKILL anything
/// still alive. Called from `main.rs` immediately before `std::process::exit(0)`
/// on the stdio path so a done `umb` NEVER trades itself for orphaned
/// children. Bounded by design (one fixed `TERM_GRACE` sleep, no waitpid/no
/// per-child blocking) so it cannot hang the exit; reaping (init will reap
/// the now-dead children) is intentionally not awaited — `exit(0)` follows
/// immediately. Idempotent and safe to call when nothing is tracked.
pub fn reap_tracked_children() {
    use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

    // Bound: a single short grace window between SIGTERM and SIGKILL. Keeps
    // the exit prompt (tests/stdio_eof.rs asserts a few-second EOF exit).
    // Deliberately NOT `doctor.rs`'s 2500ms `TERM_GRACE`: this is on the
    // `exit(0)` critical path and must stay well under the stdio_eof budget.
    const TERM_GRACE: Duration = Duration::from_millis(300);

    let tracked: Vec<(u32, u64, ChildIdent)> = match tracked_child_pids().lock() {
        Ok(map) => map
            .iter()
            .map(|(p, (st, ident))| (*p, *st, ident.clone()))
            .collect(),
        Err(_) => return,
    };
    if tracked.is_empty() {
        return;
    }

    // ONE process-table snapshot drives both re-validation and subtree
    // discovery. Refreshed once (not per-pid in a re-scanning loop) so the
    // whole pass stays O(processes), cheap, and bounded — never a hang.
    let sys = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
    );

    // Residual 1e: PID-reuse re-validation BEFORE any signal — fully mirrors
    // `doctor.rs::revalidate_for_kill`, which requires the live process to
    // satisfy BOTH an identity predicate
    // (`is_umb_daemon_proc(p.cmd(), exe_basename(p.exe()))`) AND
    // `p.start_time() == expected_start_time` before it may be signalled.
    // start_time is only ~1s-granular, so start_time alone lets a same-pid +
    // same-1-second-bucket recycle (an unrelated process) slip through; the
    // identity predicate is what closes that. Our pooled children are
    // Node/Python MCP servers, not umb daemons, so the correct identity
    // assertion (the MCP-child analogue of `is_umb_daemon_proc`) is "the live
    // process is byte-for-byte the very child we spawned": its `exe_basename`
    // and `cmd` must EQUAL the ident captured at track time (both derived the
    // exact same way doctor.rs derives them — `exe_basename(p.exe())` /
    // `p.cmd()`). The live process at the tracked pid must therefore exist
    // AND match start_time AND match ident; failing EITHER predicate (or a
    // vanished pid) FAILS re-validation and is SKIPPED entirely (never
    // signalled, logged). This makes a destructive mis-kill of an unrelated
    // bystander process — including the same-second PID-recycle case that
    // start_time-only validation left merely improbable — structurally
    // impossible even with a stale tracking entry. Reads only the SINGLE
    // existing `sys` snapshot (O(1) extra per entry: no new scan/sleep/
    // waitpid; the exit path is not made slower).
    let mut validated: Vec<u32> = Vec::new();
    for (pid, expected_start, expected_ident) in &tracked {
        match sys.process(Pid::from_u32(*pid)) {
            Some(p) => {
                let start_ok = p.start_time() == *expected_start;
                let live_ident: ChildIdent =
                    (child_exe_basename(p.exe()), p.cmd().to_vec());
                let ident_ok = live_ident == *expected_ident;
                if start_ok && ident_ok {
                    validated.push(*pid);
                } else if !start_ok {
                    tracing::warn!(
                        "[reap] skip pid {} — start_time mismatch (PID reused; \
                         NOT our pooled child, not signalled)",
                        pid
                    );
                } else {
                    tracing::warn!(
                        "[reap] skip pid {} — exe/cmd ident mismatch \
                         (same-second PID recycle to an unrelated process; \
                         NOT our pooled child, not signalled)",
                        pid
                    );
                }
            }
            None => {
                tracing::debug!(
                    "[reap] skip pid {} — no live process (already gone)",
                    pid
                );
            }
        }
    }
    if validated.is_empty() {
        if let Ok(mut map) = tracked_child_pids().lock() {
            map.clear();
        }
        return;
    }

    // === Attempt #8 (setsid+double-fork closure): ADOPTED-ORPHAN SWEEP ===
    // === by ppid == umb's own pid =====================================
    // A double-fork+`setsid()` daemonized grandchild severs its PPID link to
    // its spawner at BIRTH — by teardown it was NEVER a descendant of the
    // pooled child, so NO process-tree walk / descendant enumeration can ever
    // see it (this is why the /proc-descendant-snapshot family was proven
    // structurally dead 2×). The structural fix is PR_SET_CHILD_SUBREAPER
    // (set at startup in `install_child_subreaper`): the kernel reparents
    // that orphan to umb (nearest live subreaper) instead of to init, so by
    // the time we reap it, the orphan is umb's OWN DIRECT CHILD —
    // `ppid == std::process::id()`. We therefore find adopted orphans by
    // scanning the process table for `parent() == our pid` and excluding the
    // legitimate tracked children (those go through the pgid path above /
    // tokio Drop). Idents are captured NOW (same `sys` snapshot) so the
    // destructive SIGKILL below can reuse the existing 1e ident-revalidation
    // and never force-kill a recycled/unrelated PID. Linux-only effect (only
    // a subreaper gets reparented orphans); harmless empty set elsewhere.
    let me = std::process::id();
    let tracked_set: HashSet<u32> = tracked.iter().map(|(p, _, _)| *p).collect();
    // C1: the POSITIVE probe allowlist — pids UMB itself spawned as
    // short-lived discovery/hot-swap probes. They are `ppid==me` and NOT in
    // `tracked_set` (not pgid-led, not pooled), so without this they would
    // be mistaken for adopted orphans and SIGKILLed mid-probe. Excluded
    // exactly like tracked children. A genuine setsid+double-fork orphan is
    // never in this set (only `ProbePidGuard::new` inserts, only for our own
    // probe), so it is still swept + killed — A2 intact.
    let probe_set: HashSet<u32> = probe_pid_allowlist();
    // First sweep pass: capture (pid -> ident) of every current adopted
    // orphan. The ppid==me set is re-swept a small bounded number of times
    // across the SINGLE existing grace window (below) to absorb the reparent-
    // timing race — the intermediate fork may not have died yet on this first
    // pass, so the orphan may not have been reparented to us *yet*.
    let mut adopted_idents: HashMap<u32, ChildIdent> = HashMap::new();
    #[cfg(unix)]
    {
        for (cpid, cproc) in sys.processes() {
            let cpid = cpid.as_u32();
            if cproc.parent() == Some(Pid::from_u32(me))
                && cpid != me
                && !tracked_set.contains(&cpid)
                && !probe_set.contains(&cpid)
            {
                adopted_idents.insert(
                    cpid,
                    (child_exe_basename(cproc.exe()), cproc.cmd().to_vec()),
                );
            }
        }
    }

    // Residual A2 — PRIMARY reap: process-group signal. Each validated
    // child was spawned `process_group(0)` (pgid == child pid; see
    // `spawn_and_handshake`), so `kill(-pid, …)` reaches the child AND every
    // descendant in its group ATOMICALLY — including a grandchild orphaned to
    // init in an EARLIER teardown cycle (it kept the inherited pgid even
    // though its ppid is now 1). This is exactly what the old parent-link
    // BFS could not do: by the time the pre-exit reaper runs, prior-cycle
    // grandchildren already have ppid=1, so a downward walk from the tracked
    // pid never reaches them (the kill-then-walk reparent race the A2 E2E
    // caught). The group kill is immune to reparenting and supersedes the BFS
    // for our own children. PID-reuse safety is ALREADY enforced above: a
    // tracked entry only reaches here if its live group leader (pid == pgid)
    // re-validated on BOTH the 1e predicates (start_time AND exe/cmd ident,
    // mirroring `doctor.rs::revalidate_for_kill`); a recycled pgid whose
    // leader is gone/mismatched was SKIPPED and never enters `validated`, so
    // we never blind-kill a recycled process group.
    #[cfg(unix)]
    for pgid in &validated {
        let _ = signal_process_group(*pgid, SIGTERM);
    }

    // Attempt #8: SIGTERM every ADOPTED orphan (a setsid+double-fork
    // grandchild the kernel reparented to umb because we are a child
    // subreaper — `ppid == me`). This is the load-bearing path the
    // pgid-kill above can NEVER reach: a double-fork+setsid orphan left both
    // the pooled child's process group AND its descendant tree at birth, so
    // neither `kill(-pgid)` nor any tree walk touches it; only the subreaper
    // adoption makes it visible (as our own direct child). We do NOT
    // ident-revalidate before this graceful SIGTERM (a polite request, not
    // the destructive SIGKILL); the SIGKILL escalation below DOES reuse the
    // 1e ident-revalidation so a recycled PID is never force-killed.
    #[cfg(unix)]
    for spid in adopted_idents.keys() {
        // SAFETY: single-PID positive `kill(pid, SIGTERM)`; non-blocking.
        unsafe {
            let _ = kill(*spid as i32, SIGTERM);
        }
    }

    // Residual 1d (defense-in-depth BACKSTOP, NOT the primary): on the off
    // chance a descendant escaped the group (it called `setsid`/`setpgid`
    // itself — vanishingly rare for MCP servers, but cheap to cover), also
    // collect the validated children's still-attached descendants via the
    // bounded parent-link BFS over THIS snapshot and SIGTERM them too. This
    // is now a backstop behind the pgid kill, not the load-bearing path.
    let mut targets: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for pid in &validated {
        if seen.insert(*pid) {
            targets.push(*pid);
        }
    }
    let mut frontier: Vec<Pid> = validated.iter().map(|p| Pid::from_u32(*p)).collect();
    while let Some(cur) = frontier.pop() {
        for (cpid, cproc) in sys.processes() {
            if cproc.parent() == Some(cur) && seen.insert(cpid.as_u32()) {
                targets.push(cpid.as_u32());
                frontier.push(*cpid);
            }
        }
    }
    for pid in &targets {
        if let Some(p) = sys.process(Pid::from_u32(*pid)) {
            let _ = p.kill_with(sysinfo::Signal::Term);
        }
    }

    // One bounded grace (the SINGLE existing 300ms `TERM_GRACE`), but spent
    // as a SMALL bounded number of equal sub-slices so the adopted-orphan
    // ppid==me set can be RE-SWEPT each slice. This absorbs the reparent-
    // timing race WITHOUT adding any latency class: the intermediate fork of
    // a double-fork+setsid orphan may not have died yet on our first sweep,
    // so the orphan is not yet reparented to umb; by re-sweeping across the
    // grace we catch it the moment the kernel hands it to us. NOT an
    // unbounded loop and NOT a new sleep budget — total sleep == the exact
    // same `TERM_GRACE`, just subdivided. `RESWEEP_PASSES` is small (3).
    const RESWEEP_PASSES: u32 = 3;
    let slice = TERM_GRACE / RESWEEP_PASSES;
    for _ in 0..RESWEEP_PASSES {
        std::thread::sleep(slice);
        // Re-sweep: any NEW ppid==me orphan (just reparented to us as the
        // intermediate fork finally died) gets recorded + SIGTERMed now so
        // it has had a polite chance before the post-grace SIGKILL. Reuses a
        // fresh table snapshot (bounded, O(processes), no waitpid/no block).
        #[cfg(unix)]
        {
            let s = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            // C1: re-read the probe allowlist each pass (a probe may have
            // started/finished during the grace) so a fresh in-flight probe
            // is still spared; a real orphan is still never allowlisted.
            let probe_set_now: HashSet<u32> = probe_pid_allowlist();
            for (cpid, cproc) in s.processes() {
                let cpid = cpid.as_u32();
                if cproc.parent() == Some(Pid::from_u32(me))
                    && cpid != me
                    && !tracked_set.contains(&cpid)
                    && !probe_set_now.contains(&cpid)
                    && !adopted_idents.contains_key(&cpid)
                {
                    adopted_idents.insert(
                        cpid,
                        (child_exe_basename(cproc.exe()), cproc.cmd().to_vec()),
                    );
                    // SAFETY: single-PID positive SIGTERM; non-blocking.
                    unsafe {
                        let _ = kill(cpid as i32, SIGTERM);
                    }
                }
            }
        }
    }

    // SIGKILL whatever ignored SIGTERM: the whole process group again
    // (primary, pgid) plus any BFS-collected stragglers (backstop). No
    // waitpid: detached pipes have no graceful protocol once umb is gone.
    #[cfg(unix)]
    for pgid in &validated {
        // Skip the SIGKILL syscall if the group already drained on SIGTERM
        // (probe is `kill(-pgid,0)` — no signal delivered, O(1), no block).
        if process_group_alive(*pgid) {
            let _ = signal_process_group(*pgid, SIGKILL);
        }
    }
    let sys2 = System::new_with_specifics(
        RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
    );
    for pid in &targets {
        if let Some(p) = sys2.process(Pid::from_u32(*pid)) {
            let _ = p.kill(); // SIGKILL
        }
    }

    // Attempt #8: SIGKILL any ADOPTED orphan that ignored SIGTERM. This is
    // the destructive escalation, so it REUSES the existing 1e ident-
    // revalidation (`adopted_pid_ident_still_matches`): the live process at
    // this PID must still match the `(exe_basename, cmd)` ident captured for
    // it during the sweep. A PID recycled (or vanished) between sweep and now
    // fails the check and is SKIPPED — a recycled/unrelated bystander is
    // NEVER force-killed, identical to the tracked-pid 1e guarantee above and
    // the analogue of `doctor.rs::revalidate_for_kill`. `sys2` is the fresh
    // post-grace snapshot already taken above (no extra scan).
    #[cfg(unix)]
    for (apid, expected) in &adopted_idents {
        if adopted_pid_ident_still_matches(&sys2, *apid, expected) {
            // SAFETY: single-PID positive `kill(pid, SIGKILL)`.
            unsafe {
                let _ = kill(*apid as i32, SIGKILL);
            }
        } else {
            tracing::debug!(
                "[reap] skip adopted-orphan pid {} — ident mismatch/gone \
                 (recycled or already exited; not force-killed)",
                apid
            );
        }
    }

    // Attempt #8 zombie duty: the SIGKILLed adopted orphans are now exited
    // children of umb (we are their subreaper). Drain ONLY those explicit
    // adopted PIDs (targeted per-pid `waitpid(pid, WNOHANG)`, never the -1
    // wildcard) so none lingers as a `<defunct>` zombie. By construction the
    // legitimate pooled children are NOT in `adopted_idents` (they are
    // tracked + owned/​wait()ed by tokio separately), so this can never reap
    // a tokio Child PID. Non-blocking; bounded by the finite adopted set.
    #[cfg(unix)]
    {
        let adopted_pids: HashSet<u32> = adopted_idents.keys().copied().collect();
        let _ = reap_adopted_zombies(&adopted_pids);
    }

    if let Ok(mut map) = tracked_child_pids().lock() {
        map.clear();
    }
}

/// A persistent connection to an MCP server process.
/// Keeps the process alive across tool calls to avoid spawn+handshake overhead.
struct ServerConnection {
    child: tokio::process::Child,
    stdin: tokio::sync::Mutex<tokio::process::ChildStdin>,
    reader: tokio::sync::Mutex<TokioBufReader<tokio::process::ChildStdout>>,
    next_request_id: AtomicI32,
    server_name: String,
    /// Task #25 — TTL idle eviction: monotonic timestamp of the last
    /// SUCCESSFUL use of this connection. Set at construction and refreshed
    /// on every successful `call_tool`. The periodic pool sweeper evicts a
    /// connection only if `now - last_used > pool_idle_ttl_secs` AND the
    /// connection is not currently in-flight (its per-conn Mutex is free).
    /// `std::sync::Mutex<Instant>` (not the field directly) so the sweeper
    /// can read it WITHOUT taking the per-connection async Mutex (which an
    /// in-flight call holds) — reading idle-age must never block on / race a
    /// live request. Tiny non-async critical sections only.
    last_used: std::sync::Mutex<std::time::Instant>,
}

impl ServerConnection {
    /// Send a tools/call request and read the response
    async fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::SeqCst);

        let call_request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            }
        });

        let call_str = serde_json::to_string(&call_request)?;

        // Lock stdin, write, flush
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(format!("{}\n", call_str).as_bytes()).await?;
            stdin.flush().await?;
        }

        // Lock reader, read response
        let call_response = {
            let mut reader = self.reader.lock().await;
            ToolRouter::read_jsonrpc_response_async(&mut reader, request_id, &self.server_name).await?
        };

        if let Some(result) = call_response.get("result") {
            // Task #25: a SUCCESSFUL call refreshes the idle clock. Tiny
            // non-async critical section; poison-tolerant (a poisoned lock
            // just yields a slightly stale instant — never panics the call).
            self.touch();
            Ok(result.clone())
        } else if let Some(error) = call_response.get("error") {
            Err(anyhow!("MCP server error: {}", error))
        } else {
            Err(anyhow!("Invalid MCP server response"))
        }
    }

    /// Task #25: mark this connection as just-used (refresh the idle clock).
    /// Poison-tolerant: a poisoned lock degrades to a stale timestamp, never
    /// a panic on the hot call path.
    fn touch(&self) {
        let now = std::time::Instant::now();
        match self.last_used.lock() {
            Ok(mut t) => *t = now,
            Err(e) => *e.into_inner() = now,
        }
    }

    /// Task #25: how long this connection has been idle (since last
    /// successful use). Read by the sweeper WITHOUT holding the per-conn
    /// async Mutex, so it never blocks/races a live in-flight request.
    fn idle_for(&self) -> std::time::Duration {
        let last = match self.last_used.lock() {
            Ok(t) => *t,
            Err(e) => *e.into_inner(),
        };
        last.elapsed()
    }

    /// Check if the child process is still alive
    fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false, // Process exited
            Ok(None) => true,     // Still running
            Err(_) => false,      // Error checking = assume dead
        }
    }
}

impl Drop for ServerConnection {
    fn drop(&mut self) {
        // Capture the pid BEFORE start_kill() (which may zero out the handle's
        // id once the child is reaped). With `process_group(0)` at spawn the
        // child's pgid == its pid, so this same value is the group id.
        let pid = self.child.id();

        // Best-effort SIGKILL of the DIRECT child (tokio).
        let _ = self.child.start_kill();

        // Residual A2: the direct child is being killed, but its OWN children
        // (grandchildren of `umb`) are NOT — `start_kill()`/`kill_on_drop`
        // only signal the one pid, so a grandchild reparents to init and
        // ORPHANS on EVERY pool eviction / connection drop (the dominant
        // per-cycle leak in the A2 stress: gc accumulated 1→11). Signal the
        // whole process group so child + all descendants die together.
        //
        // This is the pool-eviction / per-connection-drop path: the
        // connection still owns this live `Child` handle, so there is NO
        // PID-recycle window here — the handle itself is the revalidation
        // (the 1e ident/start_time gate is only for the pre-exit reaper's
        // *tracked* entries, which can outlive their process). Kill the pgid
        // directly. NON-BLOCKING by design: no grace sleep / waitpid in Drop
        // (this runs under the async pool lock and on plain scope exit) —
        // SIGKILL the group outright; init reaps the now-dead members. The
        // direct child already got tokio SIGKILL above; the negative-pgid
        // SIGKILL mops up grandchildren the same instant.
        #[cfg(unix)]
        if let Some(pgid) = pid {
            let _ = signal_process_group(pgid, SIGKILL);
        }

        // Residual 2a: this child is going away via Drop; stop tracking it so
        // the pre-exit reaper does not chase a dead/recycled PID.
        if let Some(pid) = pid {
            untrack_child_pid(pid);
        }
    }
}

/// Pool of persistent MCP server connections.
/// Reuses connections across tool calls to avoid repeated spawn+handshake.
pub struct ServerConnectionPool {
    connections: tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<ServerConnection>>>>,
}

impl ServerConnectionPool {
    pub fn new() -> Self {
        Self {
            connections: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a connection for the given server
    async fn get_or_create(
        &self,
        server: &ServerConfig,
    ) -> Result<Arc<tokio::sync::Mutex<ServerConnection>>> {
        let mut connections = self.connections.lock().await;

        // Check if we have an existing connection that's still alive
        if let Some(conn) = connections.get(&server.name) {
            let mut conn_lock = conn.lock().await;
            if conn_lock.is_alive() {
                drop(conn_lock);
                return Ok(Arc::clone(conn));
            }
            // Dead connection, remove it
            tracing::debug!("[Pool] Removing dead connection for '{}'", server.name);
            drop(conn_lock);
            connections.remove(&server.name);
        }

        // Spawn a new process and do the MCP handshake
        tracing::debug!("[Pool] Creating new connection for '{}'", server.name);
        let conn = Self::spawn_and_handshake(server).await?;
        let conn = Arc::new(tokio::sync::Mutex::new(conn));
        connections.insert(server.name.clone(), Arc::clone(&conn));
        Ok(conn)
    }

    /// Remove a connection from the pool (after a failed call, OR on a
    /// hot-swap server remove/update — R2). Body UNCHANGED: it only drops
    /// the pool-map entry; the existing hardened `ServerConnection::Drop`
    /// reaps child + pgid + grandchild. `pub` so the async hot-swap
    /// callers can invoke this EXISTING removal (no new killer; no edit to
    /// Drop/reaper/sweeper).
    pub async fn remove(&self, server_name: &str) {
        let mut connections = self.connections.lock().await;
        connections.remove(server_name);
    }

    /// Task #25 — TTL idle eviction sweep. Evicts every pooled connection
    /// idle longer than `ttl` that is NOT currently in-flight, then drops
    /// it so the EXISTING hardened `ServerConnection::Drop` teardown fires
    /// (tokio SIGKILL + `kill(-pgid)` + `untrack_child_pid` → the same
    /// path the subreaper adopted-orphan sweep / 1e ident-revalidation /
    /// probe allowlist already harden). NO parallel/competing killer — this
    /// only `remove()`s the map entry; the Drop does the killing.
    ///
    /// RACE-SAFETY (the critical invariant): a connection is evicted ONLY
    /// if `conn.try_lock()` SUCCEEDS. The call path
    /// (`call_mcp_server`) holds `conn.lock().await` for the WHOLE duration
    /// of an in-flight request, so `try_lock()` returning `Err`
    /// (`WouldBlock`) is a definitive "a request is in flight RIGHT NOW" —
    /// we skip it (never evict mid-request). Idle-age itself is read via
    /// `idle_for()` which takes only the tiny inner `std::sync::Mutex`
    /// (NOT the per-conn async Mutex), so an in-flight call cannot even
    /// delay the age read. Ordering: we hold the POOL async Mutex for the
    /// whole sweep, so no `get_or_create` can hand out / re-insert a
    /// connection concurrently; `try_lock()` (non-blocking) on each conn
    /// means the sweeper can NEVER block on / deadlock vs a live call (it
    /// just skips a busy conn and revisits next tick). `ttl == 0` ⇒
    /// disabled (caller must not invoke; defensive early-return here too).
    /// Returns the number of connections evicted (for the debug log/test).
    async fn evict_idle(&self, ttl: std::time::Duration) -> usize {
        if ttl.is_zero() {
            return 0; // disabled — never evict
        }
        let mut connections = self.connections.lock().await;
        // Two-phase to avoid mutating the map while iterating it.
        let mut to_evict: Vec<String> = Vec::new();
        for (name, conn) in connections.iter() {
            // NON-BLOCKING: a held lock ⇒ an in-flight call ⇒ skip (never
            // evict mid-request; never block the sweeper on a live call).
            match conn.try_lock() {
                Ok(guard) => {
                    if guard.idle_for() >= ttl {
                        to_evict.push(name.clone());
                    }
                    // guard dropped here — we did NOT keep the conn locked
                }
                Err(_) => {
                    // In-flight (or otherwise contended) — leave it; the
                    // next sweep will reconsider it once it goes idle.
                }
            }
        }
        for name in &to_evict {
            // #42 (accurate): `remove()` drops the POOL MAP's strong ref to
            // this connection's `Arc`. It is NOT guaranteed to be the LAST
            // ref: there is a benign TOCTOU window between our `try_lock()`
            // (which only proved no in-flight call held the per-conn async
            // Mutex AT THAT INSTANT) and this `remove()`. `call_mcp_server`
            // takes a transient `Arc::clone` from the pool, then
            // `conn.lock().await`s it; a caller that cloned the Arc just
            // before we removed the entry may still hold that clone (e.g.
            // momentarily before its own `.lock()`), so `remove()` may only
            // drop strong-count 2→1. In that case `ServerConnection::Drop`
            // is DEFERRED until that caller's clone drops at its
            // fn-return, and then fires exactly ONCE (Arc guarantees a
            // single Drop on the final ref). Either way the hardened
            // teardown (tokio SIGKILL + kill(-pgid) + untrack_child_pid)
            // runs once — just not necessarily synchronously here. (A
            // caller that already holds the per-conn lock when we scan is
            // separately excluded: `try_lock()` returned Err so the entry
            // was never queued for eviction.) Existing teardown reused, not
            // duplicated.
            connections.remove(name);
            tracing::debug!(
                "[Pool] Evicted idle connection '{}' (idle > {:?}); \
                 hardened Drop teardown fired",
                name,
                ttl
            );
        }
        to_evict.len()
    }

    /// Task #25 (fix/idle-eviction-runtime) — TTL-scaled sweep cadence.
    ///
    /// ROOT CAUSE this replaces: the sweep interval used to be a hardcoded
    /// 45s, totally decoupled from `pool_idle_ttl_secs`. Eviction latency is
    /// bounded by the SWEEP PERIOD, not the TTL — so any TTL shorter than
    /// (and any idle window shorter than) the 45s sweep meant idle backends
    /// were observably NEVER evicted at runtime (real-VM E2E: ttl=8s,
    /// 12s idle windows ⇒ evicted in only 1/20 cycles; child persisted,
    /// RSS crept). Unit tests passed because they called `evict_idle()`
    /// directly and never exercised the live sweep cadence.
    ///
    /// POLICY: `sweep_period = clamp(1s, ttl/4, 60s)`.
    ///  * ttl/4 ⇒ worst-case eviction latency ≈ ttl + ttl/4 = 1.25·ttl
    ///    (a conn that goes idle right after a sweep crosses the TTL and is
    ///    removed at most one sweep-period later) — bounded & proportional.
    ///  * 1s floor ⇒ no pathological tight-spin for tiny TTLs.
    ///  * 60s ceiling ⇒ the long-lived default (ttl=600s ⇒ 60s sweep) keeps
    ///    overhead negligible (≈ the old cadence) while still bounded.
    /// Pure & total (no panics) so the runtime test asserts the SAME policy
    /// production uses.
    pub fn sweep_interval_for(ttl_secs: u64) -> std::time::Duration {
        const FLOOR_SECS: u64 = 1;
        const CEIL_SECS: u64 = 60;
        // ttl/4, then clamp into [FLOOR, CEIL]. ttl_secs==0 is the disabled
        // sentinel (sweeper not spawned) so its value here is irrelevant; we
        // still return a sane FLOOR rather than a zero-period interval.
        let quarter = ttl_secs / 4;
        let secs = quarter.clamp(FLOOR_SECS, CEIL_SECS);
        std::time::Duration::from_secs(secs)
    }

    /// Task #25 — spawn the periodic idle-eviction sweeper. Sweep cadence is
    /// derived from `ttl_secs` via [`sweep_interval_for`] (TTL-scaled, so
    /// eviction actually happens within a bounded multiple of the TTL — see
    /// that fn for the root-cause writeup & policy). Runs until the process
    /// exits. No-op (never spawns) when `ttl_secs == 0` (eviction disabled).
    /// Lightweight: one timer + a non-blocking pool sweep per tick; it
    /// shares the existing teardown and adds NO competing killer. Detached
    /// task (the stdio process `exit(0)`s on teardown, which stops it; the
    /// final reap is the pre-exit `reap_tracked_children()` chokepoint,
    /// unchanged).
    pub fn spawn_idle_sweeper(self: &Arc<Self>, ttl_secs: u64) {
        if ttl_secs == 0 {
            tracing::debug!(
                "[Pool] pool_idle_ttl_secs=0 — idle eviction DISABLED \
                 (sweeper not spawned)"
            );
            return;
        }
        let ttl = std::time::Duration::from_secs(ttl_secs);
        let interval = Self::sweep_interval_for(ttl_secs);
        tracing::debug!(
            "[Pool] idle sweeper: ttl={}s, sweep every {:?} (TTL-scaled)",
            ttl_secs,
            interval
        );
        let pool = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Skip the immediate first tick so a just-started server does
            // not sweep before anything could possibly be idle.
            tick.tick().await;
            loop {
                tick.tick().await;
                let n = pool.evict_idle(ttl).await;
                if n > 0 {
                    tracing::info!(
                        "[Pool] idle sweeper evicted {} connection(s) (ttl={}s)",
                        n,
                        ttl_secs
                    );
                }
            }
        });
    }

    /// Spawn a new MCP server process and complete the handshake
    async fn spawn_and_handshake(server: &ServerConfig) -> Result<ServerConnection> {
        let mut cmd = TokioCommand::new(&server.command);
        cmd.args(&server.args)
            .envs(&server.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // S5 (H7): if the handshake below fails (or this ServerConnection
            // is ever dropped without an explicit kill), tokio SIGKILLs the
            // child instead of orphaning a Node/Python MCP subprocess.
            .kill_on_drop(true);
        // Residual A2: spawn the pooled child as its OWN process-group leader
        // (pgid == child pid). Every descendant the MCP server itself spawns
        // (a grandchild worker, a double-forked helper) inherits this pgid
        // unless it explicitly creates its own group. Teardown then signals
        // `-pgid` and the kernel reaps the WHOLE subtree atomically — immune
        // to the reparent race that defeated the 1d parent-link BFS (an
        // earlier-cycle orphan already at ppid=1 is still in the same process
        // group, so `kill(-pgid)` still reaches it). `process_group(0)` is the
        // tokio mirror of `std::os::unix::process::CommandExt::process_group`
        // (the in-tree `doctor.rs::spawn_fake_daemon` `CommandExt` pattern);
        // it `setpgid(0,0)`s the child pre-exec — no new dependency.
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn MCP server '{}': {}", server.name, e))?;

        // Residual 2a: track this pooled child so the stdio path can reap it
        // before `std::process::exit(0)` (which bypasses Drop/kill_on_drop).
        //
        // Residual 1a: hold the tracking in an RAII `TrackedChildGuard`
        // (mirrors `RegistryGuard`). It is armed here, the instant after
        // `.spawn()`; the ~8 `?` early returns below all run its `Drop`,
        // which untracks the pid — closing the handshake-failure leak. On
        // success we `disarm()` it just before constructing
        // `ServerConnection`, transferring untrack-on-`Drop` ownership to the
        // connection (unchanged). Residual 1e: record the child's spawn
        // `start_time` AND its identity (`(exe_basename, cmd)` — the same pair
        // `doctor.rs::revalidate_for_kill` checks) so the pre-exit reaper can
        // PID-reuse re-validate it on BOTH predicates; if the just-spawned pid
        // is somehow not resolvable (start_time OR ident unresolvable in the
        // sysinfo race) we do NOT track it (an un-revalidatable entry is a
        // mis-kill hazard, not a help — `resolve_child_start_time` returns
        // `None` unless BOTH are captured from its single snapshot).
        let mut _track_guard: Option<TrackedChildGuard> = match child.id() {
            Some(pid) => resolve_child_start_time(pid)
                .map(|(start_time, ident)| {
                    TrackedChildGuard::track(pid, start_time, ident)
                }),
            None => None,
        };

        let stdin = child.stdin.take()
            .ok_or_else(|| anyhow!("Failed to get stdin for {}", server.name))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow!("Failed to get stdout for {}", server.name))?;

        let mut reader = TokioBufReader::new(stdout);

        // Step 1: Send initialize request
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "universal-mcp-bridge",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        });

        let init_str = serde_json::to_string(&init_request)?;
        let mut stdin_writer = stdin;
        stdin_writer.write_all(format!("{}\n", init_str).as_bytes()).await?;
        stdin_writer.flush().await?;

        // Step 2: Wait for initialize response
        let _init_response = ToolRouter::read_jsonrpc_response_async(&mut reader, 1, &server.name).await?;

        // Step 3: Send initialized notification
        let initialized_notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });
        let notif_str = serde_json::to_string(&initialized_notification)?;
        stdin_writer.write_all(format!("{}\n", notif_str).as_bytes()).await?;
        stdin_writer.flush().await?;

        // Small delay for the server to process notification
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Residual 1a: handshake succeeded — ownership of the tracked pid now
        // transfers to the `ServerConnection` below, whose `Drop` untracks
        // it. Disarm the leak-guard so its `Drop` is a no-op (mirrors calling
        // `RegistryGuard::remove_now` on the graceful path). Any early return
        // ABOVE this line still runs the guard's `Drop` and untracks.
        if let Some(g) = _track_guard.as_mut() {
            g.disarm();
        }

        Ok(ServerConnection {
            child,
            stdin: tokio::sync::Mutex::new(stdin_writer),
            reader: tokio::sync::Mutex::new(reader),
            next_request_id: AtomicI32::new(2), // 1 was used for initialize
            server_name: server.name.clone(),
            // Task #25: a freshly-built+handshaked connection starts its
            // idle clock NOW (it has just been "used" to handshake).
            last_used: std::sync::Mutex::new(std::time::Instant::now()),
        })
    }

    /// Kill all pooled connections (for shutdown)
    pub async fn shutdown(&self) {
        let mut connections = self.connections.lock().await;
        for (name, conn) in connections.drain() {
            let mut conn = conn.lock().await;
            let pid = conn.child.id();
            if let Some(pid) = pid {
                untrack_child_pid(pid);
            }
            let _ = conn.child.start_kill();
            // Residual A2: SIGKILL the whole process group (pgid == child pid
            // via `process_group(0)`) so a grandchild the MCP server spawned
            // dies with its parent instead of orphaning to init. The conn
            // owns the live handle (no PID-recycle window); kill the group
            // directly. Non-blocking; init reaps the dead members.
            #[cfg(unix)]
            if let Some(pgid) = pid {
                let _ = signal_process_group(pgid, SIGKILL);
            }
            // Reap the (direct) child process to prevent zombies
            let _ = conn.child.wait().await;
            tracing::debug!("[Pool] Killed and reaped connection for '{}'", name);
        }
    }
}

/// Tool definition
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub server: String,
}

/// P2 — deterministic, pure, zero-LLM short-description derivation for the
/// always-on / discovery surface.
///
/// The bulk `list_tools` surface must stay small (P1 two-tier): per tool we
/// emit `{name, short_description(desc)}` only — the full prose + schema is
/// fetched on demand via `get_tool_info`. This takes the FIRST sentence
/// (up to the first `. ` or end), trims trailing whitespace/period, and
/// hard-caps length so one verbose tool cannot bloat the list. Idempotent
/// and order-stable (same input ⇒ same output, always).
pub fn short_description(full: &str) -> String {
    let trimmed = full.trim();
    // First sentence: split on the first ". " (sentence boundary) or take
    // the whole string if there is none.
    let first = match trimmed.find(". ") {
        Some(i) => &trimmed[..i],
        None => trimmed.trim_end_matches('.'),
    };
    let first = first.trim().trim_end_matches('.').trim();
    // Hard cap (chars, not bytes split — never panic on a multibyte char):
    // 120 chars is ample for a one-line picker hint.
    const MAX: usize = 120;
    if first.chars().count() <= MAX {
        first.to_string()
    } else {
        let cut: String = first.chars().take(MAX - 1).collect();
        format!("{}…", cut.trim_end())
    }
}

/// P2 — deterministic JSON-Schema minifier. Strips zero-signal /
/// default-equivalent fields so the (on-demand) schema payload is smaller
/// WITHOUT losing anything an agent needs to construct valid args:
/// - drops `additionalProperties:false` (the JSON-Schema default behaviour
///   agents assume anyway is `true`, but MCP servers rarely rely on it for
///   arg *construction*; removing the explicit `false` does not change
///   which args are valid for the agent's purpose of building a call) —
///   ONLY when it is literally `false` (a `true`/object value is kept);
/// - drops an empty `"required": []` (semantically identical to absent);
/// - drops `"title"` (display-only, never needed to build args);
/// - recurses into `properties` and nested object schemas.
/// Pure + total (never panics; unknown shapes pass through untouched);
/// idempotent (minify(minify(x)) == minify(x)); order-preserving for
/// `serde_json::Map` (it is insertion-ordered via the `preserve_order`
/// feature already enabled through indexmap-backed serde_json? — we sort
/// nothing, so input key order is whatever serde_json yields, stable per
/// build). Schema VALIDITY is preserved: only default-equivalent /
/// display-only keys are removed.
pub fn minify_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                match k.as_str() {
                    // Display-only — never needed to construct args.
                    "title" => continue,
                    // Drop ONLY an explicit `false` (default-equivalent for
                    // arg construction); keep any other value verbatim.
                    "additionalProperties" if v == &Value::Bool(false) => continue,
                    // Empty required list == no required list.
                    "required" => {
                        if v.as_array().map(|a| a.is_empty()).unwrap_or(false) {
                            continue;
                        }
                        out.insert(k.clone(), v.clone());
                    }
                    // Recurse into the property map and any nested schema.
                    "properties" => {
                        if let Value::Object(props) = v {
                            let mut np = serde_json::Map::new();
                            for (pk, pv) in props {
                                np.insert(pk.clone(), minify_schema(pv));
                            }
                            out.insert(k.clone(), Value::Object(np));
                        } else {
                            out.insert(k.clone(), minify_schema(v));
                        }
                    }
                    _ => {
                        out.insert(k.clone(), minify_schema(v));
                    }
                }
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(minify_schema).collect()),
        other => other.clone(),
    }
}

/// MCP server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

/// SSE server configuration
#[derive(Debug, Clone)]
pub struct SseServerConfig {
    pub name: String,
    pub url: String,
    pub env: HashMap<String, String>,
}

/// HTTP server configuration (MCP Streamable HTTP transport endpoint)
#[derive(Debug, Clone)]
pub struct HttpServerConfig {
    pub name: String,
    pub url: String,
    pub env: HashMap<String, String>,
}

/// R1 portal-stall fix — an OWNED, lock-free tool-call dispatch target.
///
/// Produced by `ToolRouter::resolve_call_target` while the router read
/// guard is briefly held (a consistent snapshot), then `dispatch`ed by the
/// caller AFTER the guard is dropped. Every field is owned: configs are
/// `Clone`, the pool is an `Arc<ServerConnectionPool>` (its connections and
/// teardown are independent of the router lock). Dispatch performs the same
/// backend logic the in-`ToolRouter` helpers do — it routes by transport
/// exactly as `call_tool` did; ONLY the lock scope changed.
pub enum ResolvedCall {
    /// Native in-process tool (no spawn, no router state needed).
    Local { tool_name: String },
    /// Stdio MCP server via the shared connection pool.
    Stdio {
        server: ServerConfig,
        pool: Arc<ServerConnectionPool>,
        tool_name: String,
    },
    /// SSE MCP server (legacy HTTP+SSE transport).
    Sse {
        server: SseServerConfig,
        tool_name: String,
    },
    /// Streamable-HTTP MCP server.
    Http {
        server: HttpServerConfig,
        tool_name: String,
    },
}

impl ResolvedCall {
    /// Perform the backend call WITHOUT holding any router lock. Same
    /// routing/retry semantics as the original `ToolRouter::call_tool`
    /// path; the only difference is no `RwLock<ToolRouter>` guard is held
    /// across this (now portal-non-blocking) await.
    pub async fn dispatch(self, args: Value) -> Result<Value> {
        match self {
            ResolvedCall::Local { tool_name } => {
                super::local_tools::execute_local_tool(&tool_name, args).await
            }
            ResolvedCall::Stdio {
                server,
                pool,
                tool_name,
            } => {
                // Identical to `ToolRouter::call_mcp_server`: pooled call,
                // on failure evict the stale connection and retry once
                // with a fresh process. Uses the shared pool `Arc` (not
                // the router lock) — connection lifecycle is the pool's.
                tracing::debug!(
                    "[Router] Calling tool '{}' on server '{}' (lock-free dispatch)",
                    tool_name,
                    server.name
                );
                let conn = pool.get_or_create(&server).await?;
                let result = {
                    let conn_lock = conn.lock().await;
                    conn_lock.call_tool(&tool_name, args.clone()).await
                };
                match result {
                    Ok(val) => return Ok(val),
                    Err(e) => {
                        tracing::warn!(
                            "[Router] Pooled call to '{}' on '{}' failed: {}. \
                             Retrying with fresh connection.",
                            tool_name,
                            server.name,
                            e
                        );
                        pool.remove(&server.name).await;
                    }
                }
                let conn = pool.get_or_create(&server).await?;
                let conn_lock = conn.lock().await;
                conn_lock.call_tool(&tool_name, args).await
            }
            ResolvedCall::Sse { server, tool_name } => {
                ToolRouter::dispatch_sse(&server, &tool_name, args).await
            }
            ResolvedCall::Http { server, tool_name } => {
                ToolRouter::dispatch_http(&server, &tool_name, args).await
            }
        }
    }
}

/// Defect #3 — MCP **Streamable HTTP** client transport (spec 2025-03-26).
///
/// The previous `Http` backend did a plain `Content-Type: application/json`
/// POST with NO `text/event-stream` in `Accept`. Per the Streamable HTTP
/// spec the client **MUST** send `Accept: application/json, text/event-stream`;
/// modern remote MCP servers (huggingface.co/mcp, docker-docs) reply
/// **HTTP 406 Not Acceptable** to the old request, so UMB could not talk to
/// any modern remote MCP server. This implements the spec faithfully:
///
/// * Every JSON-RPC message is a fresh HTTP POST to the MCP endpoint with
///   `Content-Type: application/json` and `Accept: application/json,
///   text/event-stream` (spec §"Sending Messages to the Server" 1–2).
/// * The server replies with EITHER `Content-Type: application/json` (a
///   single JSON object — parse directly) OR `Content-Type:
///   text/event-stream` (an SSE stream; read `data:` frames, JSON-parse
///   each, return the one whose `id` matches the request, spec §5–6). The
///   client MUST support both.
/// * `Mcp-Session-Id`: if the server sets this header on the
///   `InitializeResult` response, the client MUST echo it on ALL
///   subsequent requests for the session (spec §"Session Management" 1–2).
/// * Notifications get HTTP 202 with no body (spec §4) — no response read.
///
/// Bounded by the caller's reqwest client `.timeout(30s)` (the project's
/// existing convention), so a never-completing stream cannot hang.
#[derive(Debug, Default, Clone)]
struct StreamableHttpSession {
    /// Captured from the `Mcp-Session-Id` response header on initialize.
    session_id: Option<String>,
}

/// Parse an MCP Streamable-HTTP SSE body, returning the first JSON-RPC
/// message whose `id` equals `expected_id` (or, if `expected_id` is None,
/// the first JSON object parsed). SSE framing per the WHATWG spec: lines;
/// `data:` (optionally `data: `) carries the payload; an event ends at a
/// blank line; multiple `data:` lines in one event are joined with `\n`.
/// Non-`data:` fields (`event:`, `id:`, `retry:`, comments `:`) are
/// ignored for our purposes. Pure + standalone so it is unit-testable
/// without a network.
fn parse_sse_for_jsonrpc(body: &str, expected_id: Option<i64>) -> Option<Value> {
    let mut data_buf: Vec<String> = Vec::new();
    let mut flush = |buf: &mut Vec<String>| -> Option<Value> {
        if buf.is_empty() {
            return None;
        }
        let payload = buf.join("\n");
        buf.clear();
        let v: Value = serde_json::from_str(payload.trim()).ok()?;
        match expected_id {
            Some(want) => {
                if v.get("id").and_then(|i| i.as_i64()) == Some(want) {
                    Some(v)
                } else {
                    None
                }
            }
            None => Some(v),
        }
    };
    for raw in body.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            // Event boundary — try to dispatch the accumulated data.
            if let Some(found) = flush(&mut data_buf) {
                return Some(found);
            }
            continue;
        }
        if line.starts_with(':') {
            continue; // SSE comment
        }
        if let Some(rest) = line.strip_prefix("data:") {
            // Spec: a single leading space after the colon is stripped.
            data_buf.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        }
        // event:/id:/retry: are irrelevant to JSON-RPC payload assembly.
    }
    // Stream ended without a trailing blank line — dispatch remainder.
    flush(&mut data_buf)
}

impl StreamableHttpSession {
    fn new() -> Self {
        Self { session_id: None }
    }

    /// Extract a JSON-RPC result/object from a response, handling BOTH
    /// `application/json` (single object) and `text/event-stream` (SSE).
    fn body_to_jsonrpc(
        content_type: &str,
        body: &str,
        expected_id: i64,
        server_name: &str,
    ) -> Result<Value> {
        if content_type.contains("text/event-stream") {
            parse_sse_for_jsonrpc(body, Some(expected_id))
                .or_else(|| parse_sse_for_jsonrpc(body, None))
                .ok_or_else(|| {
                    anyhow!(
                        "no JSON-RPC message in SSE stream from '{}'",
                        server_name
                    )
                })
        } else {
            // application/json (or unspecified) ⇒ single JSON object.
            serde_json::from_str::<Value>(body.trim()).map_err(|e| {
                anyhow!(
                    "failed to parse JSON response from '{}': {} (body: {:.200})",
                    server_name,
                    e,
                    body
                )
            })
        }
    }

    /// Blocking POST of a JSON-RPC *request*; returns the matching
    /// JSON-RPC message. Captures `Mcp-Session-Id` on the first response
    /// that carries it (the initialize response) and sends it thereafter.
    fn post_request(
        &mut self,
        client: &Client,
        url: &str,
        server_name: &str,
        id: i64,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let body = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let mut req = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid.clone());
        }
        let resp = req
            .json(&body)
            .send()
            .map_err(|e| anyhow!("Streamable-HTTP POST to '{}' failed: {}", server_name, e))?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Streamable-HTTP server '{}' returned status {} for {}",
                server_name,
                resp.status(),
                method
            ));
        }
        // Capture the session id from whichever response sets it (spec:
        // the InitializeResult response). Once set, keep it for the session.
        if self.session_id.is_none() {
            if let Some(sid) = resp
                .headers()
                .get("mcp-session-id")
                .and_then(|h| h.to_str().ok())
            {
                self.session_id = Some(sid.to_string());
                tracing::debug!(
                    "[Streamable-HTTP] '{}' assigned Mcp-Session-Id",
                    server_name
                );
            }
        }
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = resp
            .text()
            .map_err(|e| anyhow!("failed to read body from '{}': {}", server_name, e))?;
        Self::body_to_jsonrpc(&ctype, &text, id, server_name)
    }

    /// Blocking POST of a JSON-RPC *notification* (no `id`, no response;
    /// spec: server replies 202 Accepted, no body). Best-effort; session
    /// id is attached if known.
    fn post_notification(
        &self,
        client: &Client,
        url: &str,
        server_name: &str,
        method: &str,
        params: Value,
    ) {
        let body = json!({"jsonrpc":"2.0","method":method,"params":params});
        let mut req = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid.clone());
        }
        if let Err(e) = req.json(&body).send() {
            tracing::debug!(
                "[Streamable-HTTP] notification '{}' to '{}' failed (non-fatal): {}",
                method,
                server_name,
                e
            );
        }
    }

    /// Async variant of `post_request` for the `call_http_server` path
    /// (runs on the tokio runtime). Same transport semantics.
    async fn post_request_async(
        &mut self,
        client: &AsyncClient,
        url: &str,
        server_name: &str,
        id: i64,
        method: &str,
        params: Value,
    ) -> Result<Value> {
        let body = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let mut req = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid.clone());
        }
        let resp = req
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Streamable-HTTP POST to '{}' failed: {}", server_name, e))?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "Streamable-HTTP server '{}' returned status {} for {}",
                server_name,
                resp.status(),
                method
            ));
        }
        if self.session_id.is_none() {
            if let Some(sid) = resp
                .headers()
                .get("mcp-session-id")
                .and_then(|h| h.to_str().ok())
            {
                self.session_id = Some(sid.to_string());
            }
        }
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = resp
            .text()
            .await
            .map_err(|e| anyhow!("failed to read body from '{}': {}", server_name, e))?;
        Self::body_to_jsonrpc(&ctype, &text, id, server_name)
    }

    async fn post_notification_async(
        &self,
        client: &AsyncClient,
        url: &str,
        server_name: &str,
        method: &str,
        params: Value,
    ) {
        let body = json!({"jsonrpc":"2.0","method":method,"params":params});
        let mut req = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid.clone());
        }
        if let Err(e) = req.json(&body).send().await {
            tracing::debug!(
                "[Streamable-HTTP] async notification '{}' to '{}' failed (non-fatal): {}",
                method,
                server_name,
                e
            );
        }
    }
}

/// Tool router - routes tool calls to appropriate MCP servers
pub struct ToolRouter {
    /// `(server, name)`-keyed tool registry. Two external servers can both
    /// export the same `name` and both remain reachable (each gets its own
    /// `(server, name)` slot). Builtin/local precedence still wins over a
    /// later external registration of the same name (the guard in
    /// `register_tool` short-circuits before insertion). `IndexMap`
    /// preserves insertion order, so the P3 deterministic discovery order
    /// and the registration order from `discovery.rs` both stay stable.
    tools: IndexMap<(String, String), Tool>,
    servers: HashMap<String, ServerConfig>,
    sse_servers: HashMap<String, SseServerConfig>,
    http_servers: HashMap<String, HttpServerConfig>,
    #[cfg(feature = "embed-onnx")]
    semantic_search: Option<Arc<RwLock<SemanticSearchProvider>>>,
    connection_pool: Arc<ServerConnectionPool>,
    /// Minimum cosine similarity threshold for semantic search (default: 0.7)
    search_threshold: f32,
    /// Maximum tools returned by list_tools (default: 10)
    search_limit: usize,
    /// Tool-dictionary overlay (overlay-at-READ in `get_tool_info` and the
    /// slim `list_tools` envelope). `None` means no overlay (always
    /// returns live server def). Storage of `self.tools` is NEVER mutated
    /// by the overlay; provenance is exposed via `_source: "server|dict"`
    /// in `get_tool_info` and `--doctor-tools`.
    dict: Option<Arc<crate::server::tool_dictionary::ToolDictionary>>,
    /// Tool-dictionary mode (Off / Auto / On). When `dict` is `None` this
    /// field is unused.
    dict_mode: crate::server::tool_dictionary::ShortMode,
}

impl ToolRouter {
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            servers: HashMap::new(),
            sse_servers: HashMap::new(),
            http_servers: HashMap::new(),
            #[cfg(feature = "embed-onnx")]
            semantic_search: None,
            connection_pool: Arc::new(ServerConnectionPool::new()),
            search_threshold: 0.7,
            search_limit: 10,
            dict: None,
            dict_mode: crate::server::tool_dictionary::ShortMode::Auto,
        }
    }

    /// Install the tool-dictionary overlay + its mode. Read-only data;
    /// the overlay is consulted at `get_tool_info` and slim `list_tools`
    /// render time. Passing an empty dictionary is equivalent to no
    /// overlay (every lookup falls through to the live server def).
    pub fn with_tool_dictionary(
        mut self,
        dict: crate::server::tool_dictionary::ToolDictionary,
        mode: crate::server::tool_dictionary::ShortMode,
    ) -> Self {
        self.dict = Some(Arc::new(dict));
        self.dict_mode = mode;
        self
    }

    /// Setter for use after construction (mirrors `set_semantic_search`).
    /// PUBLIC API surface (intentional re-init handle for embedders +
    /// future hot-reload hook); no in-crate caller today.
    #[allow(dead_code)]
    pub fn set_tool_dictionary(
        &mut self,
        dict: crate::server::tool_dictionary::ToolDictionary,
        mode: crate::server::tool_dictionary::ShortMode,
    ) {
        self.dict = Some(Arc::new(dict));
        self.dict_mode = mode;
    }

    /// Apply the tool-dictionary overlay to one `(server, tool, live_desc)`
    /// triple. Returns `(description_to_emit, source_label)` where
    /// `source_label` is `"dict"` when the dict overrode the description
    /// and `"server"` otherwise (the operand for the `_source` provenance
    /// field). When no dict is installed, behaves as `Off` (always
    /// returns the live description with `"server"`).
    pub fn apply_dict_overlay<'a>(
        &'a self,
        server: &str,
        tool_name: &str,
        live_description: &'a str,
    ) -> (&'a str, &'static str) {
        match self.dict.as_ref() {
            None => (live_description, "server"),
            Some(d) => {
                let (out, src) = d.lookup(server, tool_name, live_description, self.dict_mode);
                let label = match src {
                    crate::server::tool_dictionary::Source::Dict => "dict",
                    crate::server::tool_dictionary::Source::Server => "server",
                };
                (out, label)
            }
        }
    }

    /// Set the semantic search similarity threshold (0.0–1.0)
    pub fn with_search_threshold(mut self, threshold: f32) -> Self {
        self.search_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the default maximum tools returned by list_tools
    pub fn with_search_limit(mut self, limit: usize) -> Self {
        self.search_limit = limit.max(1);
        self
    }

    /// Get reference to the connection pool (for shutdown)
    pub fn connection_pool(&self) -> &Arc<ServerConnectionPool> {
        &self.connection_pool
    }

    /// Enable semantic search
    #[cfg(feature = "embed-onnx")]
    pub fn with_semantic_search(mut self, provider: SemanticSearchProvider) -> Self {
        self.semantic_search = Some(Arc::new(RwLock::new(provider)));
        self
    }

    /// Set semantic search provider (for use after construction)
    #[cfg(feature = "embed-onnx")]
    pub fn set_semantic_search(&mut self, provider: SemanticSearchProvider) {
        self.semantic_search = Some(Arc::new(RwLock::new(provider)));
    }

    /// Check if semantic search is enabled.
    ///
    /// Always present so callers (daemon/stdio handlers) compile regardless of
    /// the `embed-onnx` feature. Without the feature there is no embedding
    /// engine, so this is always `false` and callers fall back to keyword. (§7.1)
    #[cfg(feature = "embed-onnx")]
    pub fn has_semantic_search(&self) -> bool {
        self.semantic_search.is_some()
    }

    /// See above — keyword-only build never has a semantic engine.
    #[cfg(not(feature = "embed-onnx"))]
    pub fn has_semantic_search(&self) -> bool {
        false
    }

    /// Default maximum tools returned when caller doesn't specify a limit
    pub fn default_limit(&self) -> usize {
        self.search_limit
    }

    /// P1 — single-tool full-detail accessor for the `get_tool_info`
    /// meta-tool (the on-demand half of the two-tier discovery surface).
    /// Returns the FULL (minified-but-complete) `{name, description,
    /// inputSchema, server}` for ONE named tool, read from the
    /// already-populated in-memory `self.tools` map. Pure read — no
    /// connection spawn/drop, no lock beyond the caller's brief read
    /// guard. `None` ⇒ unknown tool (caller returns a clean, actionable
    /// error). The schema is `minify_schema`'d (default-equivalent noise
    /// stripped) but COMPLETE — sufficient to construct a valid call.
    pub fn get_tool_info(&self, name: &str) -> Option<Value> {
        // `(server, name)`-keyed: scan for ALL entries with this name.
        // One match → same shape as before (back-compat). Two or more →
        // return a `{matches: [...]}` envelope so the agent can read each
        // candidate's distinct description + server and pick the right
        // one to call via `route_mcp_call(tool, server: <name>, args)`.
        let mut matches: Vec<&Tool> = self
            .tools
            .iter()
            .filter(|((_srv, n), _)| n == name)
            .map(|(_, t)| t)
            .collect();
        // Deterministic order on collision: by server name.
        matches.sort_by(|a, b| a.server.cmp(&b.server));
        match matches.len() {
            0 => None,
            1 => {
                let t = matches[0];
                // Tool-dictionary overlay-at-read: replace description with
                // the curated short form when the dict applies. Provenance
                // is emitted as `_source: "dict"` ONLY when the dict
                // actually overrode (back-compat — pure server defs stay
                // shaped exactly as before).
                let (desc, src) = self.apply_dict_overlay(&t.server, &t.name, &t.description);
                if src == "dict" {
                    Some(json!({
                        "name": t.name,
                        "description": desc,
                        "inputSchema": minify_schema(&t.input_schema),
                        "server": t.server,
                        "_source": "dict",
                    }))
                } else {
                    Some(json!({
                        "name": t.name,
                        "description": desc,
                        "inputSchema": minify_schema(&t.input_schema),
                        "server": t.server,
                    }))
                }
            }
            _ => {
                let arr: Vec<Value> = matches
                    .iter()
                    .map(|t| {
                        // Per-match overlay so each candidate carries its
                        // own (server-scoped) provenance label.
                        let (desc, src) =
                            self.apply_dict_overlay(&t.server, &t.name, &t.description);
                        if src == "dict" {
                            json!({
                                "name": t.name,
                                "description": desc,
                                "inputSchema": minify_schema(&t.input_schema),
                                "server": t.server,
                                "_source": "dict",
                            })
                        } else {
                            json!({
                                "name": t.name,
                                "description": desc,
                                "inputSchema": minify_schema(&t.input_schema),
                                "server": t.server,
                            })
                        }
                    })
                    .collect();
                Some(json!({
                    "ambiguous": true,
                    "name": name,
                    "matches": arr,
                    "hint": "Multiple servers export this tool name. Call \
                             route_mcp_call with `server: <name>` to disambiguate.",
                }))
            }
        }
    }

    /// Register a tool. Storage is `(server, name)`-keyed: two external
    /// servers can both export the same `name` and both remain reachable
    /// (each gets its own `(server, name)` slot). Builtin/local
    /// precedence still wins over a later external registration of the
    /// same name (short-circuits before insertion).
    pub fn register_tool(&mut self, tool: Tool) {
        let name = tool.name.clone();
        let description = tool.description.clone();
        let server = tool.server.clone();

        // Builtin/local precedence: if any builtin/local entry with this
        // name already exists, a later EXTERNAL registration of the same
        // name is skipped (existing intended behavior). Scan by-name
        // across `(server, name)` keys (n small; O(n) sub-µs).
        let already_builtin_or_local = self
            .tools
            .iter()
            .any(|((srv, n), _)| n == &name && (srv == "builtin" || srv == "local"));
        if already_builtin_or_local && server != "builtin" && server != "local" {
            tracing::debug!(
                "[Router] Skipping external tool '{}' from '{}' - already registered as builtin/local",
                name, server
            );
            return;
        }

        // A3 (existing) — external↔external collision warn. With
        // `(server, name)`-keyed storage BOTH tools now COEXIST (no
        // silent last-wins); the warn still fires so operators see the
        // clash and learn to disambiguate via the new `server` arg on
        // `route_mcp_call`.
        let other_external = self
            .tools
            .iter()
            .find(|((srv, n), _)| {
                n == &name
                    && srv != "builtin"
                    && srv != "local"
                    && srv != &server
            })
            .map(|((srv, _), _)| srv.clone());
        if let Some(other) = other_external {
            if server != "builtin" && server != "local" {
                tracing::warn!(
                    "[Router] Tool '{}' now exists on BOTH '{}' and '{}' \
                     (both reachable); callers must disambiguate by passing \
                     `server: <name>` in route_mcp_call when calling by name",
                    name, server, other
                );
            }
        }

        // Index for semantic search if enabled (embed-onnx builds only;
        // the keyword-only default has no embedding index — §7.1). The
        // semantic index is keyed by name only (legacy); collided names
        // therefore index once per registration — the search returns the
        // name, and the (server,name) lookup happens at dispatch.
        #[cfg(feature = "embed-onnx")]
        if let Some(ref search) = self.semantic_search {
            if let Err(e) = search.write().index_tool(&name, &description, &server) {
                tracing::warn!("[Router] Failed to index tool for semantic search: {}", e);
            }
        }

        self.tools.insert((server, name), tool);
    }

    /// Register a stdio server (limit enforcement now done at load time)
    pub fn register_server(&mut self, config: ServerConfig) -> Result<()> {
        self.servers.insert(config.name.clone(), config);
        Ok(())
    }

    /// Register an SSE server
    pub fn register_sse_server(&mut self, config: SseServerConfig) -> Result<()> {
        self.sse_servers.insert(config.name.clone(), config);
        Ok(())
    }

    /// Register an HTTP server
    pub fn register_http_server(&mut self, config: HttpServerConfig) -> Result<()> {
        self.http_servers.insert(config.name.clone(), config);
        Ok(())
    }

    /// Unregister a server and its tools (for hot-swap). With
    /// `(server, name)`-keyed storage this filters on the first key
    /// component directly — collided same-name tools from OTHER servers
    /// are untouched (only this server's entries are removed).
    pub fn unregister_server(&mut self, name: &str) {
        // Remove server from all server types
        self.servers.remove(name);
        self.sse_servers.remove(name);
        self.http_servers.remove(name);

        // Remove tools belonging to this server (by (server, name) key —
        // collided same-name tools from OTHER servers stay reachable).
        let keys_to_remove: Vec<(String, String)> = self.tools
            .iter()
            .filter(|((srv, _), _)| srv == name)
            .map(|(k, _)| k.clone())
            .collect();

        let removed = keys_to_remove.len();
        for key in &keys_to_remove {
            self.tools.shift_remove(key);

            // Remove from semantic search index (embed-onnx builds only — §7.1)
            #[cfg(feature = "embed-onnx")]
            if let Some(ref search) = self.semantic_search {
                search.write().remove_tool(&key.1);
            }
        }

        tracing::info!(
            "[Router] Unregistered server '{}' and {} tools",
            name,
            removed
        );
    }

    /// Clear all non-builtin servers and tools (for full reload).
    /// Preserves meta-tools (list_tools, list_mcps, route_mcp_call,
    /// get_tool_info) and local tools. `(server, name)`-keyed filter
    /// matches on the first key component.
    pub fn clear_all_servers(&mut self) {
        // Clear all server registrations
        self.servers.clear();
        self.sse_servers.clear();
        self.http_servers.clear();

        // Remove all non-builtin and non-local tools
        let keys_to_remove: Vec<(String, String)> = self.tools
            .iter()
            .filter(|((srv, _), _)| srv != "builtin" && srv != "local")
            .map(|(k, _)| k.clone())
            .collect();

        let removed_count = keys_to_remove.len();

        for key in &keys_to_remove {
            self.tools.shift_remove(key);

            // Remove from semantic search index (embed-onnx builds only — §7.1)
            #[cfg(feature = "embed-onnx")]
            if let Some(ref search) = self.semantic_search {
                search.write().remove_tool(&key.1);
            }
        }

        tracing::info!(
            "[Router] Cleared all servers and {} non-builtin tools",
            removed_count
        );
    }

    /// Discover tools from an MCP server by spawning it and calling tools/list
    /// This performs the proper MCP handshake: initialize → initialized → tools/list
    pub fn discover_tools_from_server(&self, server_name: &str, config: &ServerConfig) -> Result<Vec<Tool>> {
        tracing::info!("[Discovery] Querying {} for tools...", server_name);

        // Spawn the MCP server process
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn MCP server '{}': {}", server_name, e))?;

        // C1: register this UMB-spawned probe pid in the POSITIVE allowlist
        // the instant after `.spawn()`. The adopted-orphan sweep in
        // `reap_tracked_children()` excludes allowlisted pids exactly like
        // tracked children, so a discovery/hot-swap probe that races
        // teardown is NEVER SIGKILLed. RAII: `_probe_guard`'s `Drop`
        // unregisters on the normal return, on EVERY `?` early return below,
        // and after the explicit `child.kill()`/`wait()` — the allowlist
        // window is exactly this probe's lifetime, no longer (no
        // recycled-pid staleness). A real setsid+double-fork orphan is
        // never inserted here (only this guard, only for our own probe), so
        // it is still swept + killed — A2 preserved.
        let _probe_guard = ProbePidGuard::new(child.id());

        let mut stdin = child.stdin.take()
            .ok_or_else(|| anyhow!("Failed to get stdin for {}", server_name))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| anyhow!("Failed to get stdout for {}", server_name))?;

        let mut reader = BufReader::new(stdout);
        let mut request_id = 1;

        // Step 1: Send initialize request
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "universal-mcp-bridge",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        });

        let init_str = serde_json::to_string(&init_request)?;
        writeln!(stdin, "{}", init_str)?;
        stdin.flush()?;

        // Step 2: Wait for initialize response
        let init_response = Self::read_jsonrpc_response(&mut reader, request_id, server_name)?;
        tracing::debug!("[Discovery] {} initialize response: {:?}", server_name, init_response);
        request_id += 1;

        // Step 3: Send initialized notification (no id, no response expected)
        let initialized_notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });
        let notif_str = serde_json::to_string(&initialized_notification)?;
        writeln!(stdin, "{}", notif_str)?;
        stdin.flush()?;

        // Small delay to let server process the notification
        std::thread::sleep(Duration::from_millis(50));

        // Step 4: Send tools/list request
        let tools_request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {}
        });
        let tools_str = serde_json::to_string(&tools_request)?;
        writeln!(stdin, "{}", tools_str)?;
        stdin.flush()?;

        // Step 5: Wait for tools/list response
        let tools_response = Self::read_jsonrpc_response(&mut reader, request_id, server_name)?;

        // Kill the process and reap it to prevent zombie
        let _ = child.kill();
        let _ = child.wait();

        // Parse tools from response
        let tools_result = tools_response.get("result")
            .ok_or_else(|| anyhow!("No result in tools/list response from {}", server_name))?;

        let tools_array = tools_result.get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow!("No tools array in response from {}", server_name))?;

        let mut discovered_tools = Vec::new();
        for tool_value in tools_array {
            let name = tool_value.get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow!("Tool missing name field"))?
                .to_string();

            let description = tool_value.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();

            let input_schema = tool_value.get("inputSchema")
                .cloned()
                .unwrap_or(json!({"type": "object"}));

            discovered_tools.push(Tool {
                name,
                description,
                input_schema,
                server: server_name.to_string(),
            });
        }

        tracing::info!(
            "[Discovery] {} returned {} tools",
            server_name,
            discovered_tools.len()
        );

        Ok(discovered_tools)
    }

    /// Discover tools from an SSE-based MCP server
    /// MCP-over-SSE protocol: GET /sse for connection, POST to message endpoint, responses come via SSE stream
    pub fn discover_tools_from_sse_server(&self, server_name: &str, sse_url: &str) -> Result<Vec<Tool>> {
        tracing::info!("[Discovery] Querying SSE server {} at {}...", server_name, sse_url);

        // MCP-over-SSE protocol:
        // 1. Open SSE connection to get message endpoint
        // 2. POST JSON-RPC requests to message endpoint
        // 3. Responses come back on the SSE stream (NOT as HTTP response body)

        // Use ureq for simpler blocking HTTP with streaming support
        let sse_response = ureq::get(sse_url)
            .set("Accept", "text/event-stream")
            .set("Cache-Control", "no-cache")
            .call()
            .map_err(|e| anyhow!("Failed to connect to SSE server '{}': {}", server_name, e))?;

        let mut reader = std::io::BufReader::new(sse_response.into_reader());

        // Step 1: Read endpoint URL from SSE stream
        let message_url = Self::read_sse_endpoint(&mut reader, sse_url, server_name)?;
        tracing::info!("[Discovery] SSE {} message endpoint: {}", server_name, message_url);

        // Step 2: Send initialize request via POST (response comes on SSE stream)
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "universal-mcp-bridge",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        });

        // POST request (response will come on SSE stream)
        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&init_request)
            .map_err(|e| anyhow!("Failed to send initialize to SSE server '{}': {}", server_name, e))?;

        // Read initialize response from SSE stream
        let init_response = Self::read_sse_jsonrpc_response(&mut reader, 1, server_name)?;
        tracing::debug!("[Discovery] {} SSE initialize response: {:?}", server_name, init_response);

        // Step 3: Send initialized notification
        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }));

        // Step 4: Send tools/list request
        let tools_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&tools_request)
            .map_err(|e| anyhow!("Failed to send tools/list to SSE server '{}': {}", server_name, e))?;

        // Read tools/list response from SSE stream
        let tools_response = Self::read_sse_jsonrpc_response(&mut reader, 2, server_name)?;

        // Parse tools from response
        let tools_result = tools_response.get("result")
            .ok_or_else(|| anyhow!("No result in tools/list response from SSE server {}", server_name))?;

        let tools_array = tools_result.get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow!("No tools array in response from SSE server {}", server_name))?;

        let mut discovered_tools = Vec::new();
        for tool_value in tools_array {
            let name = tool_value.get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow!("Tool missing name field"))?
                .to_string();

            let description = tool_value.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();

            let input_schema = tool_value.get("inputSchema")
                .cloned()
                .unwrap_or(json!({"type": "object"}));

            discovered_tools.push(Tool {
                name,
                description,
                input_schema,
                server: server_name.to_string(),
            });
        }

        tracing::info!(
            "[Discovery] SSE {} returned {} tools",
            server_name,
            discovered_tools.len()
        );

        Ok(discovered_tools)
    }

    /// Read the message endpoint URL from SSE stream
    fn read_sse_endpoint<R: std::io::BufRead>(reader: &mut R, base_url: &str, server_name: &str) -> Result<String> {
        let base = base_url.trim_end_matches("/sse").trim_end_matches('/');
        let mut line = String::new();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(30);

        loop {
            if start.elapsed() > timeout {
                return Err(anyhow!("Timeout waiting for endpoint from SSE server {}", server_name));
            }

            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Ok(_) => {
                    let trimmed = line.trim();

                    // Look for "data: /path" or "data: http://..."
                    if trimmed.starts_with("data:") {
                        let data = trimmed.strip_prefix("data:").unwrap().trim();

                        if data.starts_with('/') {
                            return Ok(format!("{}{}", base, data));
                        } else if data.starts_with("http") {
                            return Ok(data.to_string());
                        }

                        // Try parsing as JSON
                        if data.starts_with('{') {
                            if let Ok(json) = serde_json::from_str::<Value>(data) {
                                if let Some(endpoint) = json.get("endpoint").and_then(|e| e.as_str()) {
                                    if endpoint.starts_with('/') {
                                        return Ok(format!("{}{}", base, endpoint));
                                    } else if endpoint.starts_with("http") {
                                        return Ok(endpoint.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => return Err(anyhow!("Error reading SSE stream from {}: {}", server_name, e)),
            }
        }
    }

    /// Read a JSON-RPC response from the SSE stream
    fn read_sse_jsonrpc_response<R: std::io::BufRead>(reader: &mut R, expected_id: i32, server_name: &str) -> Result<Value> {
        let mut line = String::new();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(30);

        loop {
            if start.elapsed() > timeout {
                return Err(anyhow!("Timeout waiting for response from SSE server {}", server_name));
            }

            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Ok(_) => {
                    let trimmed = line.trim();

                    // Look for "data: {...}" containing JSON-RPC response
                    if trimmed.starts_with("data:") {
                        let data = trimmed.strip_prefix("data:").unwrap().trim();

                        if data.starts_with('{') {
                            if let Ok(response) = serde_json::from_str::<Value>(data) {
                                // Check if this is the response we're waiting for
                                if let Some(id) = response.get("id") {
                                    if id.as_i64() == Some(expected_id as i64) {
                                        if let Some(error) = response.get("error") {
                                            return Err(anyhow!(
                                                "SSE MCP error from {}: {}",
                                                server_name,
                                                error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error")
                                            ));
                                        }
                                        return Ok(response);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => return Err(anyhow!("Error reading SSE stream from {}: {}", server_name, e)),
            }
        }
    }

    /// Discover tools from an HTTP-based MCP server (plain HTTP POST)
    /// These servers don't require the SSE handshake - just POST JSON-RPC directly
    pub fn discover_tools_from_http_server(&self, server_name: &str, http_url: &str) -> Result<Vec<Tool>> {
        tracing::info!("[Discovery] Querying HTTP server {} at {} (MCP Streamable HTTP)...", server_name, http_url);

        // Defect #3: MCP Streamable HTTP transport (spec 2025-03-26). The
        // old code did plain JSON-POST with no `text/event-stream` Accept;
        // modern remote servers (huggingface.co/mcp, docker-docs) reply
        // HTTP 406 to that. Run the proper handshake over a blocking reqwest
        // client and a session, branching json-vs-eventstream per response.
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        let mut session = StreamableHttpSession::new();

        // Step 1: initialize (capture Mcp-Session-Id from this response).
        let init_result = session.post_request(
            &client,
            http_url,
            server_name,
            1,
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "universal-mcp-bridge",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )?;
        tracing::debug!("[Discovery] {} Streamable-HTTP initialize result: {:?}", server_name, init_result);

        // Step 2: notifications/initialized (session id auto-attached).
        session.post_notification(
            &client,
            http_url,
            server_name,
            "notifications/initialized",
            json!({}),
        );

        // Step 3: tools/list.
        let tools_result = session.post_request(
            &client,
            http_url,
            server_name,
            2,
            "tools/list",
            json!({}),
        )?;

        let tools_array = tools_result
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow!("No tools array in response from HTTP server {}", server_name))?;

        let mut discovered_tools = Vec::new();
        for tool_value in tools_array {
            let name = tool_value.get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow!("Tool missing name field"))?
                .to_string();

            let description = tool_value.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();

            let input_schema = tool_value.get("inputSchema")
                .cloned()
                .unwrap_or(json!({"type": "object"}));

            discovered_tools.push(Tool {
                name,
                description,
                input_schema,
                server: server_name.to_string(),
            });
        }

        tracing::info!(
            "[Discovery] HTTP {} returned {} tools",
            server_name,
            discovered_tools.len()
        );

        Ok(discovered_tools)
    }

    /// Read a JSON-RPC response from the MCP server stdout
    fn read_jsonrpc_response(reader: &mut BufReader<std::process::ChildStdout>, expected_id: i32, server_name: &str) -> Result<Value> {
        let mut line = String::new();
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(10);

        loop {
            if start.elapsed() > timeout {
                return Err(anyhow!("Timeout waiting for response from {}", server_name));
            }

            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF - wait a bit and retry
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Defect #2(a): MCP stdio is JSON-RPC ONLY on stdout, but
                    // some backends (e.g. `local/universal-bridge`) print
                    // banners / log chatter to stdout. We must be RESILIENT:
                    // skip + debug-log a non-JSON-parseable line instead of
                    // treating it as a protocol error or stalling. (The
                    // bounded per-server discovery timeout in
                    // startup/discovery.rs is the second half of the fix —
                    // it stops a server that NEVER emits valid JSON-RPC, or
                    // one whose blocking `read_line` hangs, from blocking
                    // the whole serial discovery loop.)
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(response) => {
                            // Check if this is the response we're waiting for
                            if let Some(id) = response.get("id") {
                                if id.as_i64() == Some(expected_id as i64) {
                                    // Check for error
                                    if let Some(error) = response.get("error") {
                                        return Err(anyhow!(
                                            "MCP error from {}: {}",
                                            server_name,
                                            error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error")
                                        ));
                                    }
                                    return Ok(response);
                                }
                            }
                            // Valid JSON-RPC but not our response id → keep reading.
                        }
                        Err(_) => {
                            // Non-JSON stdout chatter — skip it (do NOT error,
                            // do NOT stall) and keep reading for the real
                            // JSON-RPC response. Truncate the log so a noisy
                            // server cannot flood logs.
                            let preview: String = trimmed.chars().take(120).collect();
                            tracing::debug!(
                                "[Discovery] {} wrote non-JSON-RPC stdout line (skipped): {}",
                                server_name,
                                preview
                            );
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow!("Error reading from {}: {}", server_name, e));
                }
            }
        }
    }

    /// Read a JSON-RPC response from the MCP server stdout using async I/O
    /// This version doesn't block the tokio runtime
    async fn read_jsonrpc_response_async(
        reader: &mut TokioBufReader<tokio::process::ChildStdout>,
        expected_id: i32,
        server_name: &str
    ) -> Result<Value> {
        let mut line = String::new();
        let timeout = Duration::from_secs(30); // Increased timeout for slow tools

        let result = tokio::time::timeout(timeout, async {
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        // EOF - wait a bit and retry
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Try to parse as JSON
                        if let Ok(response) = serde_json::from_str::<Value>(trimmed) {
                            // Check if this is the response we're waiting for
                            if let Some(id) = response.get("id") {
                                if id.as_i64() == Some(expected_id as i64) {
                                    // Check for error
                                    if let Some(error) = response.get("error") {
                                        return Err(anyhow!(
                                            "MCP error from {}: {}",
                                            server_name,
                                            error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error")
                                        ));
                                    }
                                    return Ok(response);
                                }
                            }
                            // Not our response, continue reading
                        }
                    }
                    Err(e) => {
                        return Err(anyhow!("Error reading from {}: {}", server_name, e));
                    }
                }
            }
        }).await;

        match result {
            Ok(inner) => inner,
            Err(_) => Err(anyhow!("Timeout waiting for response from {}", server_name)),
        }
    }

    /// List tools with optional semantic search and limit
    ///
    /// When no query: returns first `limit` tools alphabetically (default 10)
    /// When query + semantic: returns top `limit` semantic matches
    /// When query + substring: returns first `limit` substring matches
    pub fn list_tools(&self, query: Option<String>, use_semantic: bool, limit: usize) -> Vec<Tool> {
        // P3 — no-query path: NOT an alphabetical junk-slice. Return a
        // DETERMINISTIC, server-GROUPED ordering so the agent gets a
        // navigable capability map instead of an arbitrary A–C slice:
        //   1. local tools first (always present, the agent's primary set),
        //   2. then stdio/sse/http servers in stable name order,
        //   3. then builtin meta-tools,
        // with tools alphabetical WITHIN each group. Fully deterministic:
        // the group key is derived from server identity (not HashMap
        // iteration order), the secondary key is the tool name, so the
        // same router state yields the same order on every run/OS.
        let query = match query {
            Some(q) if !q.is_empty() => q,
            _ => {
                // `(server, name)`-keyed: emit all entries (collided
                // same-name pairs from different servers BOTH appear with
                // their distinct descriptions). Same P3 deterministic
                // ordering: group-rank → server → name (the existing
                // tertiary by-server already disambiguates the
                // collided-pair order so it is stable).
                let mut tools: Vec<Tool> = self.tools.values().cloned().collect();
                tools.sort_by(|a, b| {
                    Self::server_group_rank(&a.server)
                        .cmp(&Self::server_group_rank(&b.server))
                        .then_with(|| a.server.cmp(&b.server))
                        .then_with(|| a.name.cmp(&b.name))
                });
                return tools.into_iter().take(limit).collect();
            }
        };

        // Try semantic search first if enabled and requested.
        //
        // The whole semantic path is `embed-onnx`-only. On the keyword-only
        // default build there is no embedding engine, callers always pass
        // `use_semantic = false` (has_semantic_search() == false), and this
        // block compiles out entirely — execution falls straight through to
        // the substring/keyword path below. (§7.1)
        #[cfg(feature = "embed-onnx")]
        if use_semantic {
            if let Some(ref search) = self.semantic_search {
                // Use limit for semantic search (min of requested limit and 20)
                let search_limit = limit.min(20);
                match search.read().search(&query, search_limit, self.search_threshold) {
                    Ok(results) if !results.is_empty() => {
                        tracing::debug!(
                            "[Router] Semantic search returned {} results for '{}'",
                            results.len(),
                            query
                        );

                        // Map search results back to tools. The semantic
                        // index is keyed by name only (legacy); under
                        // `(server, name)` storage a name can map to >1
                        // entry (collision). Return ALL matches (each
                        // server's distinct entry) so the agent sees both
                        // candidates — same shape contract as the
                        // keyword/no-query branches under collision.
                        return results
                            .into_iter()
                            .flat_map(|r| {
                                let mut hits: Vec<Tool> = self
                                    .tools
                                    .iter()
                                    .filter(|((_srv, n), _)| n == &r.name)
                                    .map(|(_, t)| t.clone())
                                    .collect();
                                // Deterministic order within a collided
                                // name (by server).
                                hits.sort_by(|a, b| a.server.cmp(&b.server));
                                hits.into_iter()
                            })
                            .take(limit)
                            .collect();
                    }
                    Ok(_) => {
                        tracing::debug!("[Router] Semantic search returned no results, falling back");
                    }
                    Err(e) => {
                        tracing::warn!("[Router] Semantic search failed: {}, falling back", e);
                    }
                }
            }
        }

        // Keyword-only default build: `use_semantic` is unused (no engine).
        #[cfg(not(feature = "embed-onnx"))]
        let _ = use_semantic;

        // A2 (accuracy #1 + #6) — keyword path: SCORE then STABLE-SORT.
        // The old code did a flat `.values().filter().take()` over a
        // HashMap: (a) a NAME match and a DESCRIPTION-only match were
        // unranked (a tool whose *name* is the query could lose to one
        // that merely mentions it in prose), and (b) `HashMap::values()`
        // order is per-process random ⇒ the same query gave different
        // top results across runs. Fix: rank name-match (0) strictly
        // above description-only (1), with a deterministic secondary sort
        // on the tool name (eliminates the HashMap nondeterminism). Same
        // query + same router state ⇒ identical order, every run/OS.
        let q_lower = query.to_lowercase();
        let mut scored: Vec<(u8, Tool)> = self
            .tools
            .values()
            .filter_map(|t| {
                let name_match = t.name.to_lowercase().contains(&q_lower);
                let desc_match = t.description.to_lowercase().contains(&q_lower);
                match (name_match, desc_match) {
                    (true, _) => Some((0u8, t.clone())),     // name-match: top
                    (false, true) => Some((1u8, t.clone())), // desc-only: lower
                    _ => None,
                }
            })
            .collect();
        scored.sort_by(|(ra, ta), (rb, tb)| ra.cmp(rb).then_with(|| ta.name.cmp(&tb.name)));
        scored
            .into_iter()
            .map(|(_, t)| t)
            .take(limit)
            .collect()
    }

    /// P3 — deterministic group rank for the no-query capability map.
    /// `local` (the agent's primary always-on set) first, then external
    /// transports (stdio/sse/http servers), then `builtin` meta-tools
    /// (the agent already has these from `tools/list`, lowest browse
    /// priority). Pure, total, order-stable.
    fn server_group_rank(server: &str) -> u8 {
        match server {
            "local" => 0,
            "builtin" => 2,
            _ => 1, // any external server
        }
    }

    /// Semantic search with detailed results including similarity scores.
    ///
    /// NOTE: This method is intentionally unused - kept for potential future use.
    /// Currently, semantic search is accessed through `list_tools(query, true)` which
    /// provides a simpler interface without exposing similarity scores.
    ///
    /// If we later want to add a `semantic_search` 4th meta-tool that exposes scores
    /// (e.g., for analytics or debugging), this method is ready.
    ///
    /// Decision rationale (2025-12-19):
    /// - Current list_tools approach is sufficient and transparent
    /// - Similarity scores are internal implementation detail
    /// - Adding a 4th meta-tool would increase token usage (against UMB's value prop)
    ///
    /// Only compiled with the `embed-onnx` feature (returns `SearchResult`,
    /// which is part of the opt-in embedding stack — §7.1).
    #[cfg(feature = "embed-onnx")]
    #[allow(dead_code)]
    pub fn semantic_search(&self, query: &str, limit: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        let search = self.semantic_search.as_ref()
            .ok_or_else(|| anyhow!("Semantic search not enabled."))?;

        search.read().search(query, limit, threshold)
    }

    /// Iterate `(server, name, description)` over every registered tool
    /// (live descriptions — pre-overlay). Read-only borrow; used by
    /// `--doctor-tools` to render the per-tool provenance dump WITHOUT
    /// needing direct field access. Order matches the underlying
    /// IndexMap insertion order (stable per registration sequence).
    pub fn iter_tools_for_doctor(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.tools.values().map(|t| {
            (t.server.as_str(), t.name.as_str(), t.description.as_str())
        })
    }

    /// List all MCP servers (local, stdio, SSE, and HTTP)
    pub fn list_servers(&self) -> Vec<(String, Vec<String>)> {
        let mut result = Vec::new();

        // Add local tools as a virtual server
        let local_tools: Vec<String> = self
            .tools
            .values()
            .filter(|t| t.server == "local")
            .map(|t| t.name.clone())
            .collect();
        if !local_tools.is_empty() {
            result.push(("local".to_string(), local_tools));
        }

        // Add stdio servers
        for (server_name, _) in &self.servers {
            let tools: Vec<String> = self
                .tools
                .values()
                .filter(|t| &t.server == server_name)
                .map(|t| t.name.clone())
                .collect();

            result.push((server_name.clone(), tools));
        }

        // Add SSE servers
        for (server_name, _) in &self.sse_servers {
            let tools: Vec<String> = self
                .tools
                .values()
                .filter(|t| &t.server == server_name)
                .map(|t| t.name.clone())
                .collect();

            result.push((server_name.clone(), tools));
        }

        // Add HTTP servers
        for (server_name, _) in &self.http_servers {
            let tools: Vec<String> = self
                .tools
                .values()
                .filter(|t| &t.server == server_name)
                .map(|t| t.name.clone())
                .collect();

            result.push((server_name.clone(), tools));
        }

        result
    }

    /// Get tool count
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Get server count (stdio, SSE, and HTTP)
    pub fn server_count(&self) -> usize {
        self.servers.len() + self.sse_servers.len() + self.http_servers.len()
    }

    /// Check if a server is an SSE server
    pub fn is_sse_server(&self, name: &str) -> bool {
        self.sse_servers.contains_key(name)
    }

    /// Check if a server is an HTTP server
    pub fn is_http_server(&self, name: &str) -> bool {
        self.http_servers.contains_key(name)
    }

    /// R1 portal-stall fix — resolve a tool call to an OWNED dispatch
    /// target WITHOUT awaiting anything.
    ///
    /// The bug: callers held the `RwLock<ToolRouter>` READ guard across the
    /// up-to-30s backend `call_tool().await`. tokio `RwLock` is
    /// write-preferring, so a hot-swap / background-discovery `.write()`
    /// blocks behind an in-flight long read and then starves EVERY
    /// subsequent reader — a portal-wide stall (health pings included).
    ///
    /// This method does ONLY the synchronous map lookups and returns an
    /// owned `ResolvedCall` (configs are `Clone`; the pool is an
    /// `Arc<ServerConnectionPool>` whose lifecycle is independent of the
    /// router lock — connections live in the pool, NOT under this lock).
    /// The caller acquires the read guard briefly, calls this, DROPS the
    /// guard, then `await`s `ResolvedCall::dispatch` lock-free.
    ///
    /// Hot-swap-mid-call safety: resolution is a consistent snapshot taken
    /// under the lock. If the server is hot-swapped/removed between resolve
    /// and dispatch, dispatch operates on the cloned config + the shared
    /// pool `Arc` — never a freed/stale handle: the pool's own
    /// `get_or_create` spawns a fresh process or its dead-connection
    /// handling returns a clean `Err` (no panic / no UB). Removed-server
    /// pooled processes are reaped by the pool's OWN teardown (unchanged).
    pub fn resolve_call_target(&self, tool_name: &str) -> Result<ResolvedCall> {
        self.resolve_call_target_with_hint(tool_name, None)
    }

    /// Server-keyed resolution (additive). `server_hint`:
    /// * `Some(s)` → resolve EXACTLY `(s, tool_name)`; error if absent.
    /// * `None` AND name is unambiguous → resolve as today.
    /// * `None` AND name is collided across N servers → clean ambiguous
    ///   error listing candidate servers (the agent retries with a hint).
    /// Builtin/local precedence: a builtin/local entry shadows any
    /// external entry with the same name and is selected if no hint is
    /// given (existing intended behavior; `register_tool` already
    /// short-circuits the external registration in that case).
    pub fn resolve_call_target_with_hint(
        &self,
        tool_name: &str,
        server_hint: Option<&str>,
    ) -> Result<ResolvedCall> {
        // Gather every entry matching this name (by-name scan over the
        // `(server, name)` keys). n is small.
        let matches: Vec<&Tool> = self
            .tools
            .iter()
            .filter(|((_srv, n), _)| n == tool_name)
            .map(|(_, t)| t)
            .collect();

        // Pick the resolved Tool:
        // - server_hint Some → require exact (hint, name) match.
        // - server_hint None + 1 match → that one.
        // - server_hint None + 0 matches → not found.
        // - server_hint None + >1 matches → clean ambiguous error.
        let tool: &Tool = match (server_hint, matches.len()) {
            (Some(hint), _) => {
                match matches.iter().find(|t| t.server == hint) {
                    Some(t) => t,
                    None => {
                        let candidates: Vec<String> =
                            matches.iter().map(|t| t.server.clone()).collect();
                        if candidates.is_empty() {
                            return Err(anyhow!(
                                "Tool not found: '{}' (no tool with this name on any server)",
                                tool_name
                            ));
                        } else {
                            return Err(anyhow!(
                                "Tool '{}' not found on server '{}'; \
                                 known servers exporting it: {:?}",
                                tool_name, hint, candidates
                            ));
                        }
                    }
                }
            }
            (None, 0) => {
                return Err(anyhow!("Tool not found: {}", tool_name));
            }
            (None, 1) => matches[0],
            (None, _) => {
                // Builtin/local would have been the sole entry by the
                // register guard, so >1 matches here means external↔
                // external collision with no hint.
                let mut servers: Vec<String> =
                    matches.iter().map(|t| t.server.clone()).collect();
                servers.sort();
                return Err(anyhow!(
                    "ambiguous tool '{}'; candidate servers: {:?}; \
                     pass `server: <one>` in route_mcp_call args to disambiguate",
                    tool_name, servers
                ));
            }
        };

        if tool.server == "builtin" {
            return Err(anyhow!(
                "Tool '{}' is a meta-tool and cannot be called via route_mcp_call. Call it directly.",
                tool_name
            ));
        }
        if tool.server == "local" {
            return Ok(ResolvedCall::Local {
                tool_name: tool_name.to_string(),
            });
        }
        if let Some(server) = self.servers.get(&tool.server) {
            return Ok(ResolvedCall::Stdio {
                server: server.clone(),
                pool: Arc::clone(&self.connection_pool),
                tool_name: tool_name.to_string(),
            });
        }
        if let Some(sse_server) = self.sse_servers.get(&tool.server) {
            return Ok(ResolvedCall::Sse {
                server: sse_server.clone(),
                tool_name: tool_name.to_string(),
            });
        }
        if let Some(http_server) = self.http_servers.get(&tool.server) {
            return Ok(ResolvedCall::Http {
                server: http_server.clone(),
                tool_name: tool_name.to_string(),
            });
        }
        Err(anyhow!("Server not found: {}", tool.server))
    }

    // NOTE (born-clean): the former `ToolRouter::call_tool` and
    // `ToolRouter::call_mcp_server` were removed here as compiler-confirmed
    // dead code introduced by the R1 portal-stall refactor chain. R1
    // replaced the lock-held-across-await call path with
    // `resolve_call_target()` → `ResolvedCall::dispatch()` (see those
    // items + server.rs); the stdio pooled-call + one-shot-retry logic
    // those fns held now lives byte-equivalent inside
    // `ResolvedCall::dispatch`'s `Stdio` arm, and SSE/HTTP go through the
    // associated `dispatch_sse`/`dispatch_http`. Verified dead: no live
    // caller, no test reference (grep), and a clean release build of BOTH
    // feature sets after removal (the compiler proves it).

    /// Call an SSE-based MCP server using direct SSE protocol
    /// Uses the same approach as discovery - open SSE connection, POST to message endpoint
    ///
    /// R1: this never used `&self` (only the passed `server` config), so it
    /// is an associated fn — callable lock-free by `ResolvedCall::dispatch`.
    /// Logic byte-unchanged from the original `call_sse_server`.
    async fn dispatch_sse(server: &SseServerConfig, tool_name: &str, args: Value) -> Result<Value> {
        tracing::debug!("[Router] Calling tool '{}' on SSE server '{}' via direct SSE", tool_name, server.name);

        let sse_url = server.url.clone();
        let server_name = server.name.clone();
        let tool_name = tool_name.to_string();
        let args = args.clone();

        // Run blocking SSE I/O in spawn_blocking to avoid blocking the tokio runtime
        let result = tokio::task::spawn_blocking(move || {
            Self::call_sse_server_blocking(&sse_url, &server_name, &tool_name, args)
        }).await
        .map_err(|e| anyhow!("SSE call task panicked: {}", e))??;

        Ok(result)
    }

    /// Blocking implementation of SSE tool call (runs in spawn_blocking)
    fn call_sse_server_blocking(sse_url: &str, server_name: &str, tool_name: &str, args: Value) -> Result<Value> {
        // Open SSE connection
        let sse_response = ureq::get(sse_url)
            .set("Accept", "text/event-stream")
            .set("Cache-Control", "no-cache")
            .call()
            .map_err(|e| anyhow!("Failed to connect to SSE server '{}': {}", server_name, e))?;

        let mut reader = std::io::BufReader::new(sse_response.into_reader());

        // Step 1: Read message endpoint from SSE stream
        let message_url = Self::read_sse_endpoint(&mut reader, sse_url, server_name)?;
        tracing::debug!("[Router] SSE {} message endpoint: {}", server_name, message_url);

        // Step 2: Send initialize request
        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "universal-mcp-bridge",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }))
            .map_err(|e| anyhow!("Failed to send initialize to SSE server '{}': {}", server_name, e))?;

        // Read initialize response from SSE stream
        let _init_response = Self::read_sse_jsonrpc_response(&mut reader, 1, server_name)?;

        // Step 3: Send initialized notification
        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }));

        // Step 4: Send tools/call request
        let _ = ureq::post(&message_url)
            .set("Content-Type", "application/json")
            .send_json(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args
                }
            }))
            .map_err(|e| anyhow!("Failed to send tools/call to SSE server '{}': {}", server_name, e))?;

        // Step 5: Read tools/call response from SSE stream
        let call_response = Self::read_sse_jsonrpc_response(&mut reader, 2, server_name)?;

        // Extract result from JSON-RPC response
        if let Some(result) = call_response.get("result") {
            Ok(result.clone())
        } else if let Some(error) = call_response.get("error") {
            Err(anyhow!("SSE MCP server error: {}", error))
        } else {
            Err(anyhow!("Invalid SSE MCP server response"))
        }
    }

    /// Call a tool on an `Http` backend via the MCP **Streamable HTTP**
    /// transport (spec 2025-03-26). Sends `Accept: application/json,
    /// text/event-stream`, handles BOTH a single `application/json`
    /// response and a `text/event-stream` SSE stream, and carries the
    /// `Mcp-Session-Id` captured at initialize through the whole handshake
    /// (initialize → notifications/initialized → tools/call). Bounded by
    /// the 30s reqwest timeout (existing convention) so it cannot hang.
    ///
    /// R1: never used `&self` — an associated fn, callable lock-free by
    /// `ResolvedCall::dispatch`. Logic byte-unchanged from `call_http_server`.
    async fn dispatch_http(server: &HttpServerConfig, tool_name: &str, args: Value) -> Result<Value> {
        tracing::debug!("[Router] Calling tool '{}' on Streamable-HTTP server '{}'", tool_name, server.name);

        let client = AsyncClient::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        let mut session = StreamableHttpSession::new();

        // Step 1: initialize (capture Mcp-Session-Id from this response).
        let _init = session
            .post_request_async(
                &client,
                &server.url,
                &server.name,
                1,
                "initialize",
                json!({
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "universal-mcp-bridge",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;

        // Step 2: notifications/initialized (session id auto-attached).
        session
            .post_notification_async(
                &client,
                &server.url,
                &server.name,
                "notifications/initialized",
                json!({}),
            )
            .await;

        // Step 3: tools/call.
        let response_json = session
            .post_request_async(
                &client,
                &server.url,
                &server.name,
                2,
                "tools/call",
                json!({ "name": tool_name, "arguments": args }),
            )
            .await?;

        // Extract result from the JSON-RPC response.
        if let Some(result) = response_json.get("result") {
            Ok(result.clone())
        } else if let Some(error) = response_json.get("error") {
            Err(anyhow!("Streamable-HTTP MCP server error: {}", error))
        } else {
            Err(anyhow!("Invalid Streamable-HTTP MCP server response"))
        }
    }
}

impl Default for ToolRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only serialization guard. The PRODUCTION `reap_tracked_children()`
    /// adopted-orphan sweep targets every process whose `ppid ==
    /// std::process::id()` that is not a tracked pooled child. That is correct
    /// in the shipped binary (umb's only ppid==self processes ARE its own
    /// children + genuine adopted orphans). But cargo's default test runner
    /// executes all `#[test]`s IN ONE PROCESS, so that one test binary is the
    /// shared parent of EVERY child-spawning test's helpers — a
    /// `reap_tracked_children()` call in test A will SIGKILL test B's still-
    /// needed untracked `sleep`/`sh` helpers (and the shared `TRACKED_CHILD_PIDS`
    /// global is `clear()`ed by the reaper). This is a test-harness artifact
    /// ONLY (production is unaffected; verified by single-threaded green). Every
    /// test that calls `reap_tracked_children()`/the sweeps OR spawns an
    /// untracked child OR mutates `TRACKED_CHILD_PIDS` acquires this ONE mutex
    /// at entry, so no two ever run concurrently. Zero new dependency (std
    /// `Mutex`; `Mutex::new` is const since Rust 1.63). New child-spawning /
    /// reaper tests MUST take this lock too.
    static REAPER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper: resolve a live pid's `(start_time, ident)` via sysinfo
    /// (test-side mirror of the production `resolve_child_start_time`, used to
    /// register a tracked child with a *correct* start_time AND ident the
    /// reaper's two-predicate re-validation will accept).
    #[cfg(unix)]
    fn live_start_time(pid: u32) -> Option<(u64, ChildIdent)> {
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
        let sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        sys.process(Pid::from_u32(pid)).map(|p| {
            let ident: ChildIdent =
                (child_exe_basename(p.exe()), p.cmd().to_vec());
            (p.start_time(), ident)
        })
    }

    /// Residual 2a + Residual 1d regression: the stdio shutdown path must
    /// SYNCHRONOUSLY terminate pooled child MCP-server processes BEFORE
    /// `std::process::exit(0)` (which bypasses `Drop`/`kill_on_drop`),
    /// **including a grandchild** the pooled server itself double-forks /
    /// spawns outside our process group. Without the 1d subtree walk the
    /// direct child dies but the grandchild reparents to init and ORPHANS
    /// (project-killing class, one level down). This spawns a real `sh` that
    /// in turn spawns a real grandchild `sleep`, tracks ONLY the direct child
    /// (exactly as the pool does), runs the REAL pre-exit reaper, and asserts
    /// BOTH the child AND the grandchild are gone — genuinely exercising the
    /// BFS-subtree SIGTERM→grace→SIGKILL path (no `assert!(true)`); fails if
    /// the reaper signalled only the tracked pid.
    ///
    /// CI/RUNNER NOTE — `#[ignore]`d (run isolated): this drives the REAL
    /// `reap_tracked_children()`, whose adopted-orphan sweep
    /// (`ppid == std::process::id()`) is only sound when umb is NOT a
    /// process-group/session LEADER — the production invariant (an MCP client
    /// launches umb as a child inside the client's own group). On a bare CI
    /// runner the `cargo test` binary IS its own session/group leader, so its
    /// OWN worker threads/helper processes appear as `ppid == self` AND share
    /// umb's session; the sweep then SIGTERMs them and the test binary dies
    /// with signal 15 (the exact CI failure). This is identical to why
    /// `test_subreaper_adopts_and_reaps_setsid_doublefork_grandchild_no_zombie`
    /// is already `#[ignore]`d. Run isolated, where it passes:
    /// `cargo test -- --ignored --test-threads=1`.
    #[cfg(unix)]
    #[test]
    #[ignore = "drives the real adopted-orphan sweep (ppid==self); unsound when the cargo-test binary is its own session/group leader (CI runners) — its own worker threads become sweep targets. Run isolated: cargo test -- --ignored --test-threads=1"]
    fn test_stdio_shutdown_reaps_tracked_child_and_grandchild_before_exit() {
        // Serialize vs all other reaper/child-spawning tests (see
        // REAPER_TEST_LOCK): cargo runs tests in one shared process.
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::process::Command as StdCommand;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // A real child that spawns a real (direct) grandchild and waits on
        // it: argv is `sh -c 'sleep 3600 & echo <gpid>; wait'`. The
        // grandchild's parent() IS the tracked child (BFS-reachable). The
        // child is spawned `process_group(0)` (pgid == its pid) exactly as
        // `spawn_and_handshake` now does, so the primary process-group reap
        // covers it; the parent-link BFS remains as the backstop. (The
        // genuinely double-forked ppid=1 worker — which the BFS structurally
        // CANNOT reach — is exercised by
        // `test_reap_kills_doubleforked_grandchild_via_process_group`.)
        use std::os::unix::process::CommandExt;
        let mut child = {
            let mut c = StdCommand::new("sh");
            c.arg("-c")
                .arg("sleep 3600 & echo $!; wait")
                .stdout(std::process::Stdio::piped());
            c.process_group(0);
            c.spawn().expect("spawn child that spawns a grandchild")
        };
        let child_pid = child.id();

        // Read the grandchild pid the shell printed (`echo $!`).
        let gpid: u32 = {
            use std::io::Read;
            let mut out = child.stdout.take().expect("child stdout");
            let mut buf = String::new();
            // Read just the first line; grandchild pid is printed immediately.
            for _ in 0..200 {
                let mut b = [0u8; 1];
                match out.read(&mut b) {
                    Ok(1) => {
                        if b[0] == b'\n' {
                            break;
                        }
                        buf.push(b[0] as char);
                    }
                    _ => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
            buf.trim().parse().expect("parse grandchild pid")
        };

        // Sanity: BOTH are genuinely alive before we reap.
        let sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        assert!(
            sys.process(Pid::from_u32(child_pid)).is_some(),
            "precondition: tracked child must be alive before reap"
        );
        assert!(
            sys.process(Pid::from_u32(gpid)).is_some(),
            "precondition: grandchild must be alive before reap"
        );

        // Track ONLY the direct child, with its REAL start_time AND ident so
        // 1e two-predicate re-validation accepts it — exactly as
        // `spawn_and_handshake` does.
        let (st, ident) = live_start_time(child_pid).expect("child start_time");
        track_child_pid(child_pid, st, ident);
        reap_tracked_children();

        // BOTH must now be gone — the 1d subtree walk reached the grandchild.
        let both_gone = |label: &str| {
            let mut gone = false;
            for _ in 0..100 {
                let s = System::new_with_specifics(
                    RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
                );
                let c = s.process(Pid::from_u32(child_pid)).is_some();
                let g = s.process(Pid::from_u32(gpid)).is_some();
                if !c && !g {
                    gone = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
                let _ = label;
            }
            gone
        };
        if !both_gone("post-reap") {
            // Cleanup so a failing assertion never floats orphans.
            let _ = StdCommand::new("kill")
                .arg("-9")
                .arg(child_pid.to_string())
                .arg(gpid.to_string())
                .status();
            let _ = child.wait();
            panic!(
                "RESIDUAL 1d REGRESSION: reap_tracked_children() left the \
                 grandchild (or child) alive — a double-forked pooled-MCP \
                 grandchild would orphan on the exit(0) stdio path"
            );
        }
        let _ = child.wait();

        assert!(
            tracked_child_pids().lock().unwrap().is_empty(),
            "reaper must clear the tracked-PID map after terminating subtree"
        );
        reap_tracked_children(); // idempotent, must not panic / hang
    }

    /// Residual 1a regression: the `TrackedChildGuard` (RAII leak-guard
    /// mirroring `RegistryGuard`) MUST untrack the pid on ANY early-return
    /// path between `track` and a successfully-built `ServerConnection`. The
    /// ~8 `?` early returns in `spawn_and_handshake` all run the guard's
    /// `Drop`; only the success path `disarm()`s it. This unit-tests the
    /// guard directly: an armed guard dropped (simulated early return) MUST
    /// untrack; a disarmed guard dropped (simulated success handoff) MUST
    /// leave the entry for `ServerConnection`'s own Drop. No `assert!(true)`;
    /// fails if the leak-guard were removed or its `active` logic inverted.
    #[test]
    fn test_tracked_child_guard_untracks_on_early_return() {
        // Serialize: this test mutates + asserts the shared
        // TRACKED_CHILD_PIDS global that a concurrent reaper `clear()`s
        // (see REAPER_TEST_LOCK — fail-safe inclusion).
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Use a synthetic pid extremely unlikely to collide with a tracked
        // pooled child in the test process; the guard logic is pid-agnostic.
        let pid_a: u32 = 4_000_001;
        let pid_b: u32 = 4_000_002;

        // ARMED guard dropped WITHOUT disarm == an early-return (`?`) path.
        {
            let _g = TrackedChildGuard::track(
                pid_a,
                12345,
                (Some("umb-test".into()), vec!["umb-test".into()]),
            );
            assert!(
                tracked_child_pids().lock().unwrap().contains_key(&pid_a),
                "guard::track must register the pid while armed"
            );
        } // _g dropped here — simulates a handshake `?` early return
        assert!(
            !tracked_child_pids().lock().unwrap().contains_key(&pid_a),
            "RESIDUAL 1a REGRESSION: armed TrackedChildGuard dropped on an \
             early return must UNTRACK the pid (else handshake-failure leak)"
        );

        // DISARMED guard dropped == success handoff to ServerConnection,
        // whose own Drop owns the untrack — the guard must NOT untrack.
        {
            let mut g = TrackedChildGuard::track(
                pid_b,
                67890,
                (Some("umb-test".into()), vec!["umb-test".into()]),
            );
            assert!(
                tracked_child_pids().lock().unwrap().contains_key(&pid_b),
                "precondition: pid tracked while armed"
            );
            g.disarm();
        } // g dropped here — disarmed, must be a no-op
        assert!(
            tracked_child_pids().lock().unwrap().contains_key(&pid_b),
            "disarmed guard must NOT untrack — ownership transferred to \
             ServerConnection (mirrors RegistryGuard::remove_now semantics)"
        );
        // Cleanup the deliberately-leaked-for-the-test entry.
        untrack_child_pid(pid_b);
    }

    /// Residual 1e regression: `reap_tracked_children()` MUST PID-reuse
    /// re-validate every tracked entry against the LIVE process on BOTH the
    /// `start_time` AND the `(exe_basename, cmd)` identity predicate before
    /// signalling — fully mirroring `doctor.rs::revalidate_for_kill`
    /// (`is_umb_daemon_proc(cmd, exe_basename) && start_time == expected`).
    /// start_time is only ~1s-granular, so the dangerous case this closes is
    /// a **same-pid + same-1-second-bucket recycle** to an unrelated process:
    /// start_time alone would mis-kill it; the ident predicate makes that
    /// structurally impossible. This asserts all three branches with REAL
    /// processes (no `assert!(true)`, no weakened assertions):
    ///   (a) SAME start_time but DIFFERENT exe/cmd ident ⇒ SKIPPED — the
    ///       same-second-recycle case the new predicate closes;
    ///   (b) genuine match (start_time AND ident) ⇒ still reaped;
    ///   (c) start_time MISMATCH (correct ident) ⇒ SKIPPED (original guard).
    /// Fails if either predicate were dropped (a bystander would be killed)
    /// or if a valid child were over-spared.
    ///
    /// CI/RUNNER NOTE — `#[ignore]`d (run isolated): drives the REAL
    /// `reap_tracked_children()` adopted-orphan sweep (`ppid == self`), which
    /// is unsound when the `cargo test` binary is its own session/group leader
    /// (CI runners) — its own worker threads then look like adopted orphans
    /// and the sweep SIGTERMs them, killing the test binary (signal 15). Same
    /// rationale as the already-ignored subreaper test. Run isolated:
    /// `cargo test -- --ignored --test-threads=1`.
    #[cfg(unix)]
    #[test]
    #[ignore = "drives the real adopted-orphan sweep (ppid==self); unsound when the cargo-test binary is its own session/group leader (CI runners) — its own worker threads become sweep targets. Run isolated: cargo test -- --ignored --test-threads=1"]
    fn test_reap_skips_pid_with_mismatched_start_time_but_reaps_genuine() {
        // Serialize vs all other reaper/child-spawning tests (see
        // REAPER_TEST_LOCK): cargo runs tests in one shared process.
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::os::unix::process::CommandExt;
        use std::process::Command as StdCommand;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // Every test child is its OWN process-group leader (pgid == pid),
        // mirroring production `spawn_and_handshake`'s `process_group(0)`, so
        // the reaper's primary `kill(-pgid)` path is genuinely exercised on
        // the validated `genuine` child and each SKIPPED bystander stays
        // contained in its own group regardless.
        let pg0 = |c: &mut StdCommand| {
            c.process_group(0);
        };

        // (a) SAME-SECOND-RECYCLE BYSTANDER: a real long-lived process that
        // is NOT ours. We track its pid with its CORRECT live start_time but
        // a DIFFERENT exe/cmd ident — exactly simulating a stale entry whose
        // pid the OS recycled to this unrelated process within the same 1s
        // start_time bucket (so start_time-only validation would PASS and
        // mis-kill it). The reaper MUST SKIP it on the ident predicate.
        let mut recycle_bystander = {
            let mut c = StdCommand::new("sleep");
            c.arg("3600");
            pg0(&mut c);
            c.spawn().expect("spawn same-second-recycle bystander")
        };
        let recycle_pid = recycle_bystander.id();
        let (recycle_real_st, _recycle_real_ident) =
            live_start_time(recycle_pid).expect("recycle bystander start_time");
        // Correct start_time, deliberately WRONG ident (a different exe/cmd
        // that the live `sleep 3600` cannot match).
        let wrong_ident: ChildIdent = (
            Some("node".into()),
            vec!["node".into(), "some-mcp-server.js".into()],
        );

        // (c) START_TIME-MISMATCH BYSTANDER: tracked with a deliberately
        // WRONG start_time but its CORRECT ident — the original guard. MUST
        // also be SKIPPED (proves the start_time predicate is still enforced
        // independently, not subsumed by ident).
        let mut st_bystander = {
            let mut c = StdCommand::new("sleep");
            c.arg("3600");
            pg0(&mut c);
            c.spawn().expect("spawn start_time-mismatch bystander")
        };
        let st_bystander_pid = st_bystander.id();
        let (st_real, st_ident) =
            live_start_time(st_bystander_pid).expect("st bystander start_time");
        let wrong_st = st_real.wrapping_add(999_999); // guaranteed mismatch

        // (b) GENUINE: a real child tracked with its CORRECT start_time AND
        // CORRECT ident — must still be reaped (the conjunction must not
        // over-spare a genuinely-matching child).
        let mut genuine = {
            let mut c = StdCommand::new("sleep");
            c.arg("3600");
            pg0(&mut c);
            c.spawn().expect("spawn genuine tracked child")
        };
        let genuine_pid = genuine.id();
        let (genuine_st, genuine_ident) =
            live_start_time(genuine_pid).expect("genuine start_time");

        // Clear any prior state, then install exactly these three entries.
        {
            let mut map = tracked_child_pids().lock().unwrap();
            map.clear();
            map.insert(recycle_pid, (recycle_real_st, wrong_ident));
            map.insert(st_bystander_pid, (wrong_st, st_ident));
            map.insert(genuine_pid, (genuine_st, genuine_ident));
        }

        let sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        assert!(
            sys.process(Pid::from_u32(recycle_pid)).is_some()
                && sys.process(Pid::from_u32(st_bystander_pid)).is_some()
                && sys.process(Pid::from_u32(genuine_pid)).is_some(),
            "precondition: all three processes alive before reap"
        );

        reap_tracked_children();

        // (b) The GENUINE child must be gone (re-validated → reaped).
        let mut genuine_gone = false;
        for _ in 0..100 {
            match genuine.try_wait() {
                Ok(Some(_)) | Err(_) => {
                    genuine_gone = true;
                    break;
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        }

        // Both bystanders MUST STILL be alive (each fails exactly one of the
        // two predicates → SKIPPED, never signalled).
        let alive = |child: &mut std::process::Child, pid: u32| -> bool {
            let s = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            s.process(Pid::from_u32(pid)).is_some()
                && matches!(child.try_wait(), Ok(None))
        };
        let recycle_alive = alive(&mut recycle_bystander, recycle_pid);
        let st_bystander_alive = alive(&mut st_bystander, st_bystander_pid);

        // Clean up both bystanders regardless of assertion outcome.
        let _ = recycle_bystander.kill();
        let _ = recycle_bystander.wait();
        let _ = st_bystander.kill();
        let _ = st_bystander.wait();
        let _ = genuine.wait();
        {
            let mut map = tracked_child_pids().lock().unwrap();
            map.clear();
        }

        assert!(
            genuine_gone,
            "RESIDUAL 1e REGRESSION: a genuinely-matching tracked child \
             (start_time AND ident) was NOT reaped — the two-predicate \
             re-validation must not over-spare real children"
        );
        assert!(
            recycle_alive,
            "RESIDUAL 1e REGRESSION (DESTRUCTIVE, same-second recycle): a \
             tracked pid whose live process had the SAME start_time but a \
             DIFFERENT exe/cmd ident was signalled — the ident predicate \
             must SKIP it, else a same-1s-bucket PID recycle mis-kills an \
             unrelated process"
        );
        assert!(
            st_bystander_alive,
            "RESIDUAL 1e REGRESSION (DESTRUCTIVE): a tracked pid whose live \
             process start_time MISMATCHED was signalled — the start_time \
             predicate must still independently SKIP it"
        );
    }

    /// Residual A2 regression — THE marquee test. Genuinely reproduces the
    /// real-Linux-E2E A2 failure at integration level and proves the fix.
    ///
    /// A2's pooled child MCP server **double-forks** its long-lived worker:
    /// the worker's ppid becomes 1 (init) *immediately*, while the direct
    /// child is still alive and pooled. This is the precise shape that
    /// defeated fix 1d — `reap_tracked_children()`'s parent-link BFS walks
    /// DOWN from the (live) tracked child pid, but the worker is NOT a
    /// descendant of it (ppid=1 from birth), so the BFS can never reach it.
    /// cargo-test + 4 code reviews missed exactly this because every prior
    /// test used a *direct* grandchild the BFS could still see.
    ///
    /// Here `sh -c '( sleep 3600 & ) ; echo <gpid> ; wait'`: the `( … & )`
    /// SUBSHELL backgrounds `sleep` then the subshell EXITS, so `sleep` is
    /// reparented to init (ppid=1) — a genuine double-fork orphan — yet, with
    /// NO `setsid`, it KEEPS the process group it inherited. The direct child
    /// is spawned `process_group(0)` (pgid == its pid; the tokio mirror of
    /// `spawn_and_handshake`), so the orphaned worker is still in that group.
    /// The direct child stays ALIVE (`wait`), exactly as a pooled MCP server
    /// is when the stdio pre-exit reaper runs. We track ONLY the direct child
    /// (as the pool does); `reap_tracked_children()` 1e-revalidates the LIVE
    /// leader, then `kill(-pgid)` reaches the ppid=1 worker THROUGH the group.
    ///
    /// Pre-fix this PANICS (BFS cannot reach a ppid=1 worker — it would have
    /// caught A2). Post-fix BOTH child and worker die. Hard panic + cleanup
    /// if the worker survives; no `assert!(true)`, no weakened assertion.
    ///
    /// CI/RUNNER NOTE — `#[ignore]`d (run isolated): drives the REAL
    /// `reap_tracked_children()` adopted-orphan sweep (`ppid == self`), which
    /// is unsound when the `cargo test` binary is its own session/group leader
    /// (CI runners) — its own worker threads then look like adopted orphans
    /// and the sweep SIGTERMs them, terminating the test binary with signal 15
    /// (this is the literal CI failure that was diagnosed: the harness's own
    /// `ppid==self` threads, all sharing umb's session, were swept). The
    /// process-GROUP reap this test asserts is unaffected; only the in-process
    /// adopted sweep is runner-hostile. Same rationale as the already-ignored
    /// subreaper test. Run isolated: `cargo test -- --ignored --test-threads=1`.
    #[cfg(unix)]
    #[test]
    #[ignore = "drives the real adopted-orphan sweep (ppid==self); unsound when the cargo-test binary is its own session/group leader (CI runners) — its own worker threads become sweep targets. Run isolated: cargo test -- --ignored --test-threads=1"]
    fn test_reap_kills_doubleforked_grandchild_via_process_group() {
        // Serialize vs all other reaper/child-spawning tests (see
        // REAPER_TEST_LOCK): cargo runs tests in one shared process.
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::os::unix::process::CommandExt;
        use std::process::Command as StdCommand;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // Direct child = its OWN process-group leader (pgid == pid) — exactly
        // `spawn_and_handshake`'s shape. The `( sleep 3600 & )` subshell
        // double-forks the worker (ppid→1 at once) but does NOT setsid, so
        // the worker stays in this child's process group. The child then
        // `wait`s forever ⇒ stays ALIVE (a pooled MCP server at reap time).
        // `( sleep 3600 & echo $! )`: the subshell backgrounds the worker and
        // EXITS, so the worker is double-forked → reparented away from the
        // direct child (it is NOT a job of the outer shell, so a bare `wait`
        // would return immediately and the outer shell would exit). The outer
        // shell then `exec sleep 3600`s — SAME pid (still the pgid leader),
        // long-lived, so the direct child stays ALIVE exactly like a pooled
        // MCP server at pre-exit-reap time. The worker keeps the inherited
        // process group (no setsid), but is NOT reachable from `child_pid` by
        // the parent-link BFS.
        let mut child = {
            let mut c = StdCommand::new("sh");
            c.arg("-c")
                .arg("( sleep 3600 & echo $! ) ; exec sleep 3600")
                .stdout(std::process::Stdio::piped());
            c.process_group(0); // setpgid(0,0): child becomes pgid==pid
            c.spawn()
                .expect("spawn pgid-leader child that double-forks a worker")
        };
        let child_pid = child.id();

        let gpid: u32 = {
            use std::io::Read;
            let mut out = child.stdout.take().expect("child stdout");
            let mut buf = String::new();
            for _ in 0..400 {
                let mut b = [0u8; 1];
                match out.read(&mut b) {
                    Ok(1) => {
                        if b[0] == b'\n' {
                            break;
                        }
                        buf.push(b[0] as char);
                    }
                    _ => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
            buf.trim().parse().expect("parse worker pid")
        };

        let alive = |pid: u32| -> bool {
            let s = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            s.process(Pid::from_u32(pid)).is_some()
        };
        let ppid_of = |pid: u32| -> Option<u32> {
            let s = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            s.process(Pid::from_u32(pid))
                .and_then(|p| p.parent())
                .map(|p| p.as_u32())
        };

        // Wait for the double-fork to complete: the worker must be alive AND
        // reparented away from the direct child (ppid != child_pid). This is
        // the exact A2 state — a live pooled child whose worker is already an
        // orphan the parent-link BFS cannot reach.
        let mut orphaned = false;
        for _ in 0..300 {
            if alive(gpid) && alive(child_pid) && ppid_of(gpid) != Some(child_pid) {
                orphaned = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if !orphaned {
            let _ = StdCommand::new("kill").arg("-9").arg(child_pid.to_string())
                .arg(gpid.to_string()).status();
            let _ = child.wait();
            panic!(
                "precondition: worker must be a live double-fork orphan \
                 (alive, ppid != direct child) while the direct child is \
                 still alive — could not establish the A2 race state"
            );
        }
        assert!(
            ppid_of(gpid) != Some(child_pid),
            "precondition: worker is NOT a descendant of the tracked child \
             (double-forked) — the parent-link BFS structurally cannot reach \
             it; only the process-group reap can"
        );

        // Track ONLY the direct child with its REAL start_time+ident so 1e
        // re-validation ACCEPTS the live leader — exactly as the pool does.
        let (st, ident) = live_start_time(child_pid).expect("child start_time");
        track_child_pid(child_pid, st, ident);

        // THE REAL pre-exit reaper. 1e re-validates the LIVE direct child
        // (leader, pid == pgid) → its process group is signalled; the
        // ppid=1 double-forked worker dies WITH the group despite never
        // being reachable by the parent-link BFS.
        reap_tracked_children();

        // BOTH must now be gone: the live direct child (1e-validated leader,
        // SIGTERM→grace→SIGKILL on its pgid) AND the double-forked ppid=1
        // worker (reached ONLY because it is still in that process group).
        let mut both_gone = false;
        for _ in 0..300 {
            if !alive(child_pid) && !alive(gpid) {
                both_gone = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = child.wait(); // reap the (now-dead) direct child handle
        if !both_gone {
            // Cleanup so a failing assertion never floats an orphan.
            let _ = StdCommand::new("kill")
                .arg("-9")
                .arg(child_pid.to_string())
                .arg(gpid.to_string())
                .status();
            {
                tracked_child_pids().lock().unwrap().clear();
            }
            panic!(
                "RESIDUAL A2 REGRESSION: a double-forked (ppid=1) worker of a \
                 LIVE pooled MCP child SURVIVED reap_tracked_children() — the \
                 process-group reap (kill -pgid) must terminate it; the old \
                 parent-link BFS structurally could not (worker is not a \
                 descendant of the tracked child). This is the project-killing \
                 orphan class one level deeper, exactly as real-Linux A2 caught."
            );
        }
        assert!(
            tracked_child_pids().lock().unwrap().is_empty(),
            "reaper must clear the tracked-PID map after the group reap"
        );
        reap_tracked_children(); // idempotent — must not panic / hang
    }

    /// Residual A2 + 1e regression: the process-group reap MUST NOT signal a
    /// recycled pgid. A tracked entry whose pgid (== original child pid) the
    /// OS has since recycled to an UNRELATED process group (different
    /// exe/cmd ident) must be SKIPPED by the SAME 1e two-predicate
    /// re-validation that gates the pgid kill — `kill(-pgid,…)` is destructive
    /// to an entire foreign group, so the leader-revalidation gate is
    /// load-bearing. Mirrors `test_reap_skips_pid_with_mismatched_start_time
    /// _but_reaps_genuine` but asserts specifically that NO negative-pgid
    /// signal reached the bystander group: a real `sleep 3600` in its OWN
    /// process group (pgid == its pid) is tracked under that pid with a
    /// DIFFERENT ident; the reaper must NOT kill it (would mean a recycled
    /// pgid got a group-wide SIGTERM/SIGKILL). No `assert!(true)`.
    #[cfg(unix)]
    #[test]
    fn test_pgid_reap_skips_recycled_group_on_ident_mismatch() {
        // Serialize vs all other reaper/child-spawning tests (see
        // REAPER_TEST_LOCK): cargo runs tests in one shared process.
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::os::unix::process::CommandExt;
        use std::process::Command as StdCommand;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // A real process that is its OWN process-group leader (pgid == pid) —
        // i.e. signalling `-pid` would hit a whole group, just like a
        // recycled pooled-child pgid would. It is NOT ours.
        let mut bystander = {
            let mut c = StdCommand::new("sleep");
            c.arg("3600");
            c.process_group(0);
            c.spawn().expect("spawn pgid-leader bystander group")
        };
        let bystander_pid = bystander.id();
        let (real_st, _real_ident) =
            live_start_time(bystander_pid).expect("bystander start_time");

        // Track its pid (== its pgid) with the CORRECT start_time but a
        // DIFFERENT exe/cmd ident — exactly a stale entry whose pgid the OS
        // recycled to this unrelated group within the same 1s start_time
        // bucket. start_time-only validation would PASS and `kill(-pgid)`
        // the WHOLE foreign group; the ident predicate must SKIP it.
        let wrong_ident: ChildIdent = (
            Some("node".into()),
            vec!["node".into(), "some-mcp-server.js".into()],
        );
        {
            let mut map = tracked_child_pids().lock().unwrap();
            map.clear();
            map.insert(bystander_pid, (real_st, wrong_ident));
        }

        let s = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        assert!(
            s.process(Pid::from_u32(bystander_pid)).is_some(),
            "precondition: bystander group alive before reap"
        );

        reap_tracked_children();

        // The bystander group MUST still be alive — the ident mismatch made
        // 1e SKIP it, so neither the negative-pgid SIGTERM/SIGKILL nor the
        // BFS ever targeted it.
        let still_alive = {
            let s2 = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            s2.process(Pid::from_u32(bystander_pid)).is_some()
                && matches!(bystander.try_wait(), Ok(None))
        };

        // Cleanup regardless of outcome.
        let _ = bystander.kill();
        let _ = bystander.wait();
        {
            tracked_child_pids().lock().unwrap().clear();
        }

        assert!(
            still_alive,
            "RESIDUAL A2/1e REGRESSION (DESTRUCTIVE): a tracked pgid recycled \
             to an UNRELATED process group (ident mismatch) was group-killed \
             — the 1e leader re-validation must SKIP it before any \
             kill(-pgid,…); blind-killing a recycled pgid destroys a foreign \
             process group"
        );
    }

    #[test]
    fn test_tool_registration() {
        let mut router = ToolRouter::new();

        let tool = Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({"type": "object"}),
            server: "test_server".to_string(),
        };

        router.register_tool(tool);
        assert_eq!(router.tools.len(), 1);
    }

    #[test]
    fn test_list_tools_with_filter() {
        let mut router = ToolRouter::new();

        router.register_tool(Tool {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
            server: "fs".to_string(),
        });

        router.register_tool(Tool {
            name: "write_file".to_string(),
            description: "Write a file".to_string(),
            input_schema: json!({"type": "object"}),
            server: "fs".to_string(),
        });

        let all_tools = router.list_tools(None, false, 100);
        assert_eq!(all_tools.len(), 2);

        let filtered = router.list_tools(Some("read".to_string()), false, 100);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "read_file");
    }

    #[test]
    fn test_unregister_server() {
        let mut router = ToolRouter::new();

        router.register_server(ServerConfig {
            name: "fs".to_string(),
            command: "node".to_string(),
            args: vec![],
            env: HashMap::new(),
        }).unwrap();

        router.register_tool(Tool {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
            server: "fs".to_string(),
        });

        assert_eq!(router.tool_count(), 1);
        assert_eq!(router.server_count(), 1);

        router.unregister_server("fs");

        assert_eq!(router.tool_count(), 0);
        assert_eq!(router.server_count(), 0);
    }

    /// Defense-in-depth boundary guard: the catastrophic-class `kill(-pgid,…)`
    /// MUST be refused for every degenerate pgid and permitted ONLY for a
    /// genuine child pgid (always `> 1`, never == umb's own pid because every
    /// child is spawned `process_group(0)` ⇒ pgid == its own pid). Tests the
    /// pure predicate in isolation (signalling a real foreign group from a
    /// unit test is impractical and itself dangerous); `signal_process_group`
    /// is a thin wrapper that early-returns `false` on `!pgid_safe_to_signal`.
    #[cfg(unix)]
    #[test]
    fn test_pgid_guard_refuses_invalid_and_own_pid_proceeds_for_normal() {
        let own = std::process::id();

        // pgid == 0  ⇒ kill(0,…) signals the CALLER's own group: must refuse.
        assert!(
            !pgid_safe_to_signal(0),
            "pgid=0 (kill(0) ⇒ own process group) MUST be refused"
        );
        // pgid == 1  ⇒ kill(-1,…) signals every reachable process: refuse.
        assert!(
            !pgid_safe_to_signal(1),
            "pgid=1 (kill(-1) ⇒ every reachable process) MUST be refused"
        );
        // pgid == umb's own pid ⇒ umb's own group (self-annihilation): refuse.
        assert!(
            !pgid_safe_to_signal(own),
            "pgid == own pid (self process group) MUST be refused"
        );

        // A legitimate child pgid is always > 1 and never == umb's own pid.
        // Pick a normal pgid that is provably distinct from `own`.
        let normal: u32 = if own == 2 { 3 } else { 2 };
        assert!(
            normal > 1 && normal != own,
            "test setup: chosen normal pgid must be >1 and != own pid"
        );
        assert!(
            pgid_safe_to_signal(normal),
            "a normal child pgid (>1, != own pid) MUST be permitted to signal"
        );

        // And `signal_process_group` itself must NOT issue the syscall (it
        // early-returns false) for every refused pgid — wrapper enforces guard.
        assert!(
            !signal_process_group(0, SIG_PROBE),
            "signal_process_group(0,…) must be a no-op (guard blocked)"
        );
        assert!(
            !signal_process_group(1, SIG_PROBE),
            "signal_process_group(1,…) must be a no-op (guard blocked)"
        );
        assert!(
            !signal_process_group(own, SIG_PROBE),
            "signal_process_group(own_pid,…) must be a no-op (guard blocked)"
        );
    }

    /// Attempt #8 (THE structural fix): a process double-fork+`setsid()`s a
    /// daemonized grandchild whose PPID link is severed at BIRTH — it is
    /// NEVER a descendant of anything we tracked, so no tree walk can see it.
    /// With THIS process marked a child subreaper (`prctl
    /// PR_SET_CHILD_SUBREAPER`), the kernel reparents that orphan to US when
    /// its intermediate parent exits. We then track ONLY the (already-dead)
    /// intermediate as a pooled child and run the REAL
    /// `reap_tracked_children()`; its ppid==self adopted-orphan sweep must
    /// find and SIGKILL the daemonized grandchild, AND the subsequent
    /// `waitpid` zombie-duty must leave NO `<defunct>` behind. Also asserts
    /// the 1e ident-revalidation reuse REJECTS an unrelated/dead PID.
    ///
    /// Linux-only: PR_SET_CHILD_SUBREAPER is a Linux prctl. On non-Linux the
    /// subreaper + waitpid are documented no-ops, so the test asserts that
    /// contract instead (sweep helper rejects a dead PID; zombie-reap == 0).
    ///
    /// CI/devs: this is `#[ignore]`d because it calls the REAL global
    /// `install_child_subreaper()` — `prctl(PR_SET_CHILD_SUBREAPER)` is
    /// process-global + irreversible, so under cargo's default in-process
    /// parallel test runner it makes the WHOLE test binary adopt sibling
    /// tests' helper processes, breaking them. Production is unaffected
    /// (umb's only ppid==self processes are its own children). Run it
    /// isolated: `cargo test -- --ignored --test-threads=1`.
    #[test]
    #[cfg(unix)]
    #[ignore = "mutates process-global irreversible PR_SET_CHILD_SUBREAPER; incompatible with parallel in-process cargo test. Run isolated: cargo test -- --ignored --test-threads=1"]
    fn test_subreaper_adopts_and_reaps_setsid_doublefork_grandchild_no_zombie() {
        // Serialize even though #[ignore]d: a manual `--ignored` run WITHOUT
        // --test-threads=1 must still not overlap the other reaper tests
        // (see REAPER_TEST_LOCK — fail-safe per the enumerated lock set).
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::time::Duration;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // The 1e ident-revalidation reuse: a non-existent PID (0 is never a
        // real process) must FAIL revalidation on EVERY platform — proving a
        // vanished/recycled/unrelated PID is never force-killed.
        {
            let sys = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            let bogus: ChildIdent = (Some("x".into()), vec!["x".into()]);
            assert!(
                !adopted_pid_ident_still_matches(&sys, 0, &bogus),
                "ident-revalidation MUST reject a non-existent PID (recycled/\
                 unrelated bystander protection — never force-killed)"
            );
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Documented contract: subreaper + waitpid zombie-duty are
            // Linux-only; on other platforms they are clean no-ops so
            // behaviour is unchanged and the build stays green.
            install_child_subreaper(); // no-op, must not panic
            assert_eq!(
                reap_adopted_zombies(&HashSet::new()),
                0,
                "non-Linux reap_adopted_zombies must be a 0 no-op"
            );
            assert_eq!(
                sweep_and_reap_adopted_zombies(),
                0,
                "non-Linux sweep_and_reap_adopted_zombies must be a 0 no-op"
            );
        }

        #[cfg(target_os = "linux")]
        {
            // Become a child subreaper for THIS test process so a
            // daemonized orphan reparents to US (exactly as umb does at
            // startup via `install_child_subreaper`).
            install_child_subreaper();

            // `sh` that double-forks + setsid's a daemon grandchild then
            // EXITS immediately. Sequence:
            //   outer sh → forks `setsid sh -c 'sleep 30'` (new session,
            //   detached) → the outer sh exits at once. The `sleep 30`
            //   grandchild's parent (the `setsid sh`) also exits → the
            //   `sleep` is orphaned with NO ancestor we ever tracked. As a
            //   subreaper, WE adopt it (ppid becomes our pid).
            // It prints the daemon pid so we can assert on it precisely.
            let mut child = std::process::Command::new("/bin/sh")
                .args([
                    "-c",
                    "setsid sh -c 'echo $$ >/tmp/.umb_a8_gpid; exec sleep 30' \
                     </dev/null >/dev/null 2>&1 & exit 0",
                ])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("spawn double-fork+setsid parent sh");
            let intermediate_pid = child.id();
            // The outer sh exits immediately; reap it so it is not a zombie
            // of THIS test (mirrors tokio owning/​wait()ing the real child).
            let _ = child.wait();

            // Let the kernel: run setsid, exit the inner sh, reparent the
            // daemonized `sleep` to us (the nearest subreaper).
            std::thread::sleep(Duration::from_millis(600));

            let gpid: u32 = std::fs::read_to_string("/tmp/.umb_a8_gpid")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .expect("daemon grandchild pid file");

            // Precondition: the daemonized grandchild is ALIVE and is NOT a
            // descendant of `intermediate_pid` (its ancestors all exited) —
            // it is now OUR direct child via subreaper adoption. Prove the
            // adoption actually happened (ppid == this test process).
            let sys = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            let gproc = sys
                .process(Pid::from_u32(gpid))
                .expect("daemon grandchild must be alive pre-reap");
            assert_eq!(
                gproc.parent(),
                Some(Pid::from_u32(std::process::id())),
                "PRECONDITION: subreaper adoption must have reparented the \
                 setsid+double-fork orphan to THIS process (ppid==self); \
                 got parent {:?}",
                gproc.parent()
            );

            // Track the (already-exited) intermediate as a pooled child,
            // exactly as the pool tracks a direct child. It is gone, so the
            // pgid path is a no-op — the ADOPTED-ORPHAN sweep is the ONLY
            // thing that can kill the daemonized grandchild. (Track with a
            // best-effort start_time/ident; even if it 1e-fails the tracked
            // path, the adopted sweep is independent and is what we assert.)
            if let Some((st, ident)) = live_start_time(intermediate_pid) {
                track_child_pid(intermediate_pid, st, ident);
            } else {
                // Already reaped/gone → register a placeholder so the reaper
                // has a non-empty tracked set and proceeds to the sweep.
                track_child_pid(
                    intermediate_pid,
                    0,
                    (Some("sh".into()), vec!["sh".into()]),
                );
            }

            // THE assertion: the real consolidated reaper, via its
            // ppid==self adopted-orphan sweep (NOT any tree walk), must kill
            // the daemonized grandchild.
            reap_tracked_children();

            // Give SIGKILL a beat to take effect.
            std::thread::sleep(Duration::from_millis(300));
            let sys2 = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            let still = sys2.process(Pid::from_u32(gpid));
            // Dead, OR a zombie that our waitpid duty already reaped (so it
            // is no longer present). Either way it must NOT be a live,
            // non-zombie process.
            let leaked = matches!(still.map(|p| p.status()),
                Some(s) if s != sysinfo::ProcessStatus::Zombie
                        && s != sysinfo::ProcessStatus::Dead);
            assert!(
                !leaked,
                "ATTEMPT #8 REGRESSION: the setsid+double-fork daemonized \
                 grandchild (pid {gpid}) survived reap_tracked_children() — \
                 subreaper adoption + ppid==self sweep did NOT kill it"
            );

            // Zombie duty: after our SIGKILL the grandchild is our exited
            // child; the targeted waitpid drain must leave NO <defunct>
            // zombie. Pass the EXPLICIT adopted pid (never the -1 wildcard).
            let mut just_gpid = HashSet::new();
            just_gpid.insert(gpid);
            let _ = reap_adopted_zombies(&just_gpid);
            std::thread::sleep(Duration::from_millis(100));
            let sys3 = System::new_with_specifics(
                RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
            );
            if let Some(p) = sys3.process(Pid::from_u32(gpid)) {
                assert_ne!(
                    p.status(),
                    sysinfo::ProcessStatus::Zombie,
                    "ATTEMPT #8 zombie-duty REGRESSION: adopted orphan \
                     {gpid} left as <defunct> — targeted waitpid(pid, \
                     WNOHANG) drain failed"
                );
            }

            // Cleanup best-effort.
            let _ = std::fs::remove_file("/tmp/.umb_a8_gpid");
            unsafe {
                let _ = kill(gpid as i32, SIGKILL);
            }
            let _ = reap_adopted_zombies(&just_gpid);
            if let Ok(mut m) = tracked_child_pids().lock() {
                m.clear();
            }
        }
    }

    /// C1 regression: the `reap_tracked_children()` adopted-orphan sweep
    /// MUST spare a pid in the POSITIVE probe allowlist (a UMB-spawned
    /// discovery/hot-swap probe) while STILL killing an unregistered
    /// `ppid==self` orphan. No `prctl` needed: a plain `Command::spawn()`
    /// child of the test binary is already `ppid == std::process::id()` —
    /// exactly the C1 contamination vector — so this exercises the real
    /// sweep selection without the global subreaper. It DOES take
    /// REAPER_TEST_LOCK: it calls the real `reap_tracked_children()` + spawns
    /// untracked children.
    ///
    /// CI/RUNNER NOTE — `#[ignore]`d (run isolated): because it drives the REAL
    /// adopted-orphan sweep against `ppid==self` processes, on a CI runner
    /// (where the `cargo test` binary is its own session/group leader) the
    /// sweep ALSO sees the harness's OWN worker threads (`ppid==self`, sharing
    /// umb's session) and SIGTERMs them, killing the test binary (signal 15) —
    /// the exact diagnosed CI failure. The sweep-selection logic this asserts
    /// (allowlist spare vs orphan kill) is unchanged; it is simply unsafe to
    /// run concurrently with live sibling harness threads. Same rationale as
    /// the already-ignored subreaper test. Run isolated:
    /// `cargo test -- --ignored --test-threads=1`.
    #[cfg(unix)]
    #[test]
    #[ignore = "drives the real adopted-orphan sweep (ppid==self); unsound when the cargo-test binary is its own session/group leader (CI runners) — its own worker threads become sweep targets. Run isolated: cargo test -- --ignored --test-threads=1"]
    fn test_reap_sweep_spares_allowlisted_probe_but_kills_unregistered_orphan() {
        // Serialize vs all other reaper/child-spawning tests (see
        // REAPER_TEST_LOCK): cargo runs tests in one shared process.
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::process::Command as StdCommand;
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

        // Two REAL long-lived direct children of the test binary. Both are
        // `ppid == std::process::id()` and NEITHER is tracked/pgid-led —
        // i.e. both look exactly like adopted orphans to the sweep. The
        // ONLY difference: `probe` is registered in the positive allowlist.
        let mut probe = StdCommand::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn probe-stand-in child");
        let mut orphan = StdCommand::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn unregistered-orphan child");
        let probe_pid = probe.id();
        let orphan_pid = orphan.id();

        // Register ONLY the probe pid in the positive allowlist (exactly
        // what `discover_tools_from_server` does right after its spawn).
        let _probe_guard = ProbePidGuard::new(probe_pid);
        assert!(
            probe_pid_allowlist().contains(&probe_pid),
            "precondition: probe pid registered in positive allowlist"
        );
        assert!(
            !probe_pid_allowlist().contains(&orphan_pid),
            "precondition: orphan pid is NOT allowlisted (it is a real \
             adopted-orphan analogue)"
        );

        // The reaper early-returns at the `validated.is_empty()` guard
        // (BEFORE the adopted-orphan sweep) unless at least one tracked
        // child 1e-revalidates. So spawn a THIRD real child and track it
        // with its REAL start_time+ident (exactly as the pool does) so
        // `validated` is non-empty and the adopted-orphan sweep actually
        // runs. (Its own pgid path kills it too; that's fine — we only
        // assert on `probe` vs `orphan`.)
        let mut tracked_child = StdCommand::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn tracked-validation child");
        let tracked_pid = tracked_child.id();
        // Give the kernel a beat to publish all three children.
        std::thread::sleep(std::time::Duration::from_millis(200));
        let (tst, tident) =
            live_start_time(tracked_pid).expect("tracked child start_time");
        track_child_pid(tracked_pid, tst, tident);

        // THE assertion target: the real consolidated reaper. Its
        // adopted-orphan sweep sees BOTH `ppid==self` children; it MUST
        // exclude the allowlisted probe and MUST kill the unregistered one.
        reap_tracked_children();

        std::thread::sleep(std::time::Duration::from_millis(400));
        let sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        let probe_live = matches!(
            sys.process(Pid::from_u32(probe_pid)).map(|p| p.status()),
            Some(s) if s != sysinfo::ProcessStatus::Zombie
                    && s != sysinfo::ProcessStatus::Dead
        );
        let orphan_gone = match sys.process(Pid::from_u32(orphan_pid)).map(|p| p.status()) {
            None => true,
            Some(s) => s == sysinfo::ProcessStatus::Zombie
                || s == sysinfo::ProcessStatus::Dead,
        };

        assert!(
            probe_live,
            "C1 REGRESSION: an allowlisted UMB discovery/hot-swap probe \
             (pid {probe_pid}) was KILLED by the adopted-orphan sweep — the \
             positive probe allowlist must spare it"
        );
        assert!(
            orphan_gone,
            "C1/A2 REGRESSION: an UNREGISTERED ppid==self orphan (pid \
             {orphan_pid}) SURVIVED the sweep — the allowlist must NOT \
             over-exclude; a real (non-registered) orphan must still be \
             reaped (A2 preserved)"
        );

        // Cleanup: drop the guard (unregisters), kill both children, clear
        // the throwaway tracked entry.
        drop(_probe_guard);
        let _ = probe.kill();
        let _ = probe.wait();
        let _ = orphan.kill();
        let _ = orphan.wait();
        let _ = tracked_child.kill();
        let _ = tracked_child.wait();
        if let Ok(mut m) = tracked_child_pids().lock() {
            m.clear();
        }
        if let Ok(mut s) = probe_pids().lock() {
            s.clear();
        }
    }

    // ===================================================================
    // Task #25 — TTL idle eviction tests
    // ===================================================================

    /// Build a real-process-backed `ServerConnection` for tests: spawns a
    /// real `sleep` child as its OWN process group (mirrors production
    /// `process_group(0)`) with piped stdio, so the connection's hardened
    /// `Drop` (tokio SIGKILL + `kill(-pgid)` + `untrack_child_pid`) is
    /// genuinely exercised on eviction — proving eviction reuses the
    /// EXISTING reaper with no orphan/zombie. `last_used` is set to
    /// `now - preaged` so a test can place the conn either side of the TTL.
    #[cfg(unix)]
    fn make_test_conn(
        name: &str,
        preaged: std::time::Duration,
    ) -> (Arc<tokio::sync::Mutex<ServerConnection>>, u32) {
        use std::os::unix::process::CommandExt;
        let mut cmd = TokioCommand::new("sleep");
        cmd.arg("300")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd.process_group(0); // pgid == child pid, like spawn_and_handshake
        let mut child = cmd.spawn().expect("spawn test sleep child");
        let pid = child.id().expect("test child pid");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let reader = TokioBufReader::new(stdout);
        let last = std::time::Instant::now()
            .checked_sub(preaged)
            .unwrap_or_else(std::time::Instant::now);
        let conn = ServerConnection {
            child,
            stdin: tokio::sync::Mutex::new(stdin),
            reader: tokio::sync::Mutex::new(reader),
            next_request_id: AtomicI32::new(2),
            server_name: name.to_string(),
            last_used: std::sync::Mutex::new(last),
        };
        (Arc::new(tokio::sync::Mutex::new(conn)), pid)
    }

    #[cfg(unix)]
    fn pid_alive(pid: u32) -> bool {
        use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
        let sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything()),
        );
        matches!(
            sys.process(Pid::from_u32(pid)).map(|p| p.status()),
            Some(s) if s != sysinfo::ProcessStatus::Zombie
                    && s != sysinfo::ProcessStatus::Dead
        )
    }

    /// Evicts a connection idle longer than the TTL — and the eviction
    /// runs the EXISTING hardened Drop teardown (the real child + its
    /// process group are killed: no orphan / no zombie left behind).
    #[cfg(unix)]
    #[tokio::test]
    async fn test_evict_idle_evicts_after_ttl_via_existing_reaper() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = ServerConnectionPool::new();
        // Idle for 10s; TTL 1s ⇒ must be evicted.
        let (conn, pid) =
            make_test_conn("idle-srv", std::time::Duration::from_secs(10));
        pool.connections.lock().await.insert("idle-srv".into(), conn);

        assert!(pid_alive(pid), "precondition: test child alive pre-sweep");
        let n = pool.evict_idle(std::time::Duration::from_secs(1)).await;
        assert_eq!(n, 1, "the idle connection MUST be evicted");
        assert!(
            !pool.connections.lock().await.contains_key("idle-srv"),
            "evicted entry must be gone from the pool map"
        );
        // The hardened Drop fired on remove(): child + pgid SIGKILLed.
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            !pid_alive(pid),
            "TASK#25/A2 REGRESSION: evicted connection's child (pid {pid}) \
             survived — eviction must reuse the existing hardened Drop \
             teardown (no orphan/zombie)"
        );
    }

    /// Does NOT evict a connection that is currently in-flight (its
    /// per-conn async Mutex is held) even if it is idle past the TTL —
    /// the `try_lock()` guard prevents mid-request eviction.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_evict_idle_skips_in_use_locked_connection() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = ServerConnectionPool::new();
        let (conn, pid) =
            make_test_conn("busy-srv", std::time::Duration::from_secs(10));
        pool.connections
            .lock()
            .await
            .insert("busy-srv".into(), Arc::clone(&conn));

        // Simulate an in-flight call: hold the per-conn async Mutex exactly
        // as `call_mcp_server` does for the whole request duration.
        let in_flight = conn.lock().await;

        let n = pool.evict_idle(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            n, 0,
            "an in-flight (locked) connection MUST NOT be evicted even \
             though it is idle past the TTL — mid-request eviction is the \
             critical race the try_lock guard prevents"
        );
        assert!(
            pool.connections.lock().await.contains_key("busy-srv"),
            "busy connection must remain in the pool"
        );
        assert!(pid_alive(pid), "busy connection's child must NOT be killed");

        // Release the in-flight lock AND drop the test's own Arc clone, so
        // the pool map holds the SOLE strong ref — exactly the production
        // invariant (only the pool persistently owns the connection; an
        // in-flight caller holds a transient clone only for the call's
        // duration, then drops it). Now `connections.remove()` drops the
        // last ref and the hardened `ServerConnection::Drop` runs.
        drop(in_flight);
        drop(conn);
        let n2 = pool.evict_idle(std::time::Duration::from_secs(1)).await;
        assert_eq!(
            n2, 1,
            "once the in-flight call releases the lock, the now-idle \
             connection IS evicted on the next sweep"
        );
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(!pid_alive(pid), "child killed after the (now safe) eviction");
    }

    /// Disabled (ttl == 0) ⇒ never evicts, even a very-old idle conn.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_evict_idle_disabled_when_ttl_zero() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = ServerConnectionPool::new();
        let (conn, pid) =
            make_test_conn("keep-srv", std::time::Duration::from_secs(86_400));
        pool.connections.lock().await.insert("keep-srv".into(), conn);

        let n = pool.evict_idle(std::time::Duration::from_secs(0)).await;
        assert_eq!(n, 0, "ttl==0 (disabled) MUST never evict");
        assert!(
            pool.connections.lock().await.contains_key("keep-srv"),
            "disabled eviction must leave the pool untouched"
        );
        assert!(pid_alive(pid), "disabled ⇒ child not killed");

        // spawn_idle_sweeper with ttl_secs=0 must not spawn (no-op).
        let arc_pool = Arc::new(ServerConnectionPool::new());
        arc_pool.spawn_idle_sweeper(0);
        // (no panic / no task; nothing to assert beyond "does not blow up")

        // Cleanup the still-pooled conn's child.
        pool.connections.lock().await.clear();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let _ = pid; // killed by Drop on clear()
    }

    /// TTL boundary: a connection idle EXACTLY at/over the TTL is evicted
    /// (`>=` semantics), one comfortably under it is kept.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_evict_idle_ttl_boundary() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = ServerConnectionPool::new();
        // under TTL: idle 1s, ttl 60s ⇒ KEPT
        let (fresh, fresh_pid) =
            make_test_conn("fresh", std::time::Duration::from_secs(1));
        // at/over TTL: idle 60s, ttl 60s ⇒ EVICTED (>=)
        let (stale, stale_pid) =
            make_test_conn("stale", std::time::Duration::from_secs(60));
        {
            let mut m = pool.connections.lock().await;
            m.insert("fresh".into(), fresh);
            m.insert("stale".into(), stale);
        }
        let n = pool.evict_idle(std::time::Duration::from_secs(60)).await;
        assert_eq!(n, 1, "exactly the at/over-TTL connection is evicted");
        let m = pool.connections.lock().await;
        assert!(m.contains_key("fresh"), "under-TTL connection KEPT");
        assert!(!m.contains_key("stale"), "at/over-TTL connection EVICTED");
        drop(m);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(pid_alive(fresh_pid), "kept conn child still alive");
        assert!(!pid_alive(stale_pid), "evicted conn child killed (no orphan)");
        pool.connections.lock().await.clear();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let _ = fresh_pid;
    }

    /// `touch()` (called on every successful `call_tool`) resets the idle
    /// clock so a recently-used connection is NOT evicted.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_touch_resets_idle_clock_prevents_eviction() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pool = ServerConnectionPool::new();
        let (conn, pid) =
            make_test_conn("touched", std::time::Duration::from_secs(120));
        pool.connections
            .lock()
            .await
            .insert("touched".into(), Arc::clone(&conn));
        // Simulate a successful call refreshing the clock.
        conn.lock().await.touch();
        let n = pool.evict_idle(std::time::Duration::from_secs(60)).await;
        assert_eq!(
            n, 0,
            "touch() must reset last_used so a just-used connection is NOT \
             evicted even though it was constructed pre-aged"
        );
        assert!(pid_alive(pid), "touched connection child still alive");
        pool.connections.lock().await.clear();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let _ = pid;
    }

    /// fix/idle-eviction-runtime — the TTL-scaled sweep-cadence POLICY.
    /// This pure assertion locks the H1 fix: a hardcoded long interval
    /// (the original 45s bug) or any cadence ≥ TTL means a short-TTL idle
    /// connection is observably never evicted at runtime.
    #[test]
    fn test_sweep_interval_is_ttl_scaled_not_hardcoded() {
        use std::time::Duration;
        // Short TTL ⇒ short sweep (NOT the old hardcoded 45s). With ttl=8
        // the sweep must be ≤ ttl so eviction can happen inside a ~12s
        // idle window — the exact real-VM E2E scenario that failed.
        let s8 = ServerConnectionPool::sweep_interval_for(8);
        assert!(
            s8 <= Duration::from_secs(8),
            "ttl=8 must sweep at most every 8s (got {:?}); the original \
             hardcoded 45s is what made short-TTL eviction non-functional",
            s8
        );
        assert!(s8 >= Duration::from_secs(1), "floor is 1s, got {:?}", s8);
        // ttl=8 ⇒ ttl/4 = 2s (within [1,60]).
        assert_eq!(s8, Duration::from_secs(2), "ttl/4 policy for ttl=8");
        // Tiny TTL clamps to the 1s floor (no tight-spin).
        assert_eq!(
            ServerConnectionPool::sweep_interval_for(1),
            Duration::from_secs(1),
            "floor clamp"
        );
        // Default 600s ⇒ ttl/4 = 150s clamped to the 60s ceiling.
        assert_eq!(
            ServerConnectionPool::sweep_interval_for(600),
            Duration::from_secs(60),
            "ceiling clamp keeps long-lived default overhead negligible"
        );
        // Eviction-latency invariant: sweep period MUST be < TTL for any
        // enabled TTL ≥ the floor*… so a connection cannot stay idle
        // unbounded-ly past its TTL. Spot-check a spread of TTLs.
        for ttl in [5_u64, 8, 20, 60, 120, 600, 3600] {
            let p = ServerConnectionPool::sweep_interval_for(ttl);
            assert!(
                p < Duration::from_secs(ttl),
                "sweep period {:?} must be < ttl {}s so eviction latency \
                 is bounded (≈1.25·ttl), not unbounded",
                p,
                ttl
            );
        }
    }

    /// fix/idle-eviction-runtime — THE test that would have caught the
    /// shipped bug. Unlike the other Task #25 tests (which call
    /// `evict_idle()` directly and so never exercise the live sweep
    /// cadence), this one starts the REAL detached sweeper via
    /// `spawn_idle_sweeper(ttl)` and asserts that a genuinely-idle,
    /// real-process-backed pooled connection is evicted AND its child
    /// reaped within a bounded WALL-CLOCK time — and that `ttl=0` never
    /// evicts even after waiting. On the buggy `8b14aba` the production
    /// cadence was a hardcoded 45s, so within this test's bounded ~6s
    /// window the connection is NEVER evicted and the child stays alive →
    /// this test FAILS on 8b14aba and PASSES on the TTL-scaled fix
    /// (ttl=3 ⇒ sweep every ~1s ⇒ evicted within a couple ticks).
    /// Spawns real children + exercises the hardened Drop ⇒
    /// REAPER_TEST_LOCK-gated.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_spawned_sweeper_evicts_short_ttl_idle_within_bounded_time() {
        let _serial = REAPER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // --- Part A: short TTL ⇒ a fresh (idle) conn IS evicted by the
        // live spawned sweeper within a bounded wall-time. ---
        let pool = Arc::new(ServerConnectionPool::new());
        // Not pre-aged: it becomes "idle" only by the wall clock passing,
        // exactly like a real backend left untouched after a tool call.
        let (conn, pid) = make_test_conn("idle-srv", std::time::Duration::ZERO);
        pool.connections
            .lock()
            .await
            .insert("idle-srv".into(), conn);
        assert!(pid_alive(pid), "precondition: child alive before idle");

        // ttl=3 ⇒ sweep_interval_for = clamp(1, 3/4=0→1, 60) = 1s. So the
        // conn crosses its 3s TTL and the ~1s sweeper evicts it well
        // within our 10s bound. (On 8b14aba the cadence would be the
        // hardcoded 45s ⇒ NOT evicted within this bound ⇒ test fails.)
        pool.spawn_idle_sweeper(3);

        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut evicted = false;
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if !pool.connections.lock().await.contains_key("idle-srv") {
                evicted = true;
                break;
            }
        }
        assert!(
            evicted,
            "live spawned sweeper MUST evict a >TTL-idle connection within \
             a bounded wall-time; if this fails the sweep cadence is \
             decoupled from the TTL again (the original 45s-hardcoded bug)"
        );
        // The hardened ServerConnection::Drop must have reaped the child
        // (no orphan / no zombie) — eviction reuses the EXISTING teardown.
        let reap_by =
            std::time::Instant::now() + std::time::Duration::from_secs(3);
        while pid_alive(pid) && std::time::Instant::now() < reap_by {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            !pid_alive(pid),
            "evicted connection's child MUST be reaped by the existing \
             hardened Drop (no orphan)"
        );

        // --- Part B: ttl=0 ⇒ sweeper is a no-op; a long-idle conn is
        // NEVER evicted even after we wait. ---
        let pool0 = Arc::new(ServerConnectionPool::new());
        let (keep, keep_pid) =
            make_test_conn("keep0", std::time::Duration::from_secs(3600));
        pool0
            .connections
            .lock()
            .await
            .insert("keep0".into(), keep);
        pool0.spawn_idle_sweeper(0); // disabled — must not spawn/evict
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        assert!(
            pool0.connections.lock().await.contains_key("keep0"),
            "ttl=0 MUST genuinely disable eviction (conn still pooled)"
        );
        assert!(
            pid_alive(keep_pid),
            "ttl=0 ⇒ child never killed by any sweeper"
        );

        // Cleanup: drop both pools' remaining conns ⇒ hardened Drop reaps.
        pool0.connections.lock().await.clear();
        pool.connections.lock().await.clear();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let _ = (pid, keep_pid);
    }

    /// Defect #2: a backend that writes NON-JSON-RPC chatter to stdout
    /// before its valid JSON-RPC responses must NOT stall discovery —
    /// `discover_tools_from_server` skips the junk lines and still
    /// completes (and does so well within a bounded time, proving no
    /// hang). Stub server = a `sh` script that prints a banner line, then
    /// replies to `initialize` and `tools/list` with valid JSON-RPC. It
    /// reads its own stdin so it stays alive for the handshake, then
    /// exits. REAPER_TEST_LOCK-gated (spawns a child + touches the
    /// discovery path).
    #[cfg(unix)]
    #[test]
    fn test_discovery_skips_non_json_chatter_and_does_not_stall() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // The stub: emit a NON-JSON banner first (the exact failure mode —
        // `local/universal-bridge` printing chatter to stdout), then a
        // valid initialize result, then a valid tools/list result with one
        // tool. `read` lines from stdin so we only respond after umb sends
        // each request (keeps ordering deterministic) and exit cleanly.
        let script = r#"
echo "universal-bridge v1.2.3 starting up (this is NOT json-rpc)"
echo "another banner line {not valid json"
read _init
echo '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"stub","version":"1"}}}'
read _initnotif
read _toolslist
echo '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"stub_tool","description":"d","inputSchema":{"type":"object"}}]}}'
"#;

        let config = ServerConfig {
            name: "noisy-stub".to_string(),
            command: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), script.to_string()],
            env: HashMap::new(),
        };

        let router = ToolRouter::new();
        let start = std::time::Instant::now();
        let result = router.discover_tools_from_server("noisy-stub", &config);
        let elapsed = start.elapsed();

        // (1) Must NOT hang: comfortably under the inner ~10s read timeout
        // AND the outer 20s per-server discovery ceiling. A healthy
        // handshake here is sub-second; allow generous CI slack.
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "Defect #2 REGRESSION: discovery took {elapsed:?} — non-JSON \
             chatter stalled the stdio discovery path"
        );

        // (2) Must SUCCEED: the junk lines were skipped, the real
        // JSON-RPC was found, the tool was discovered.
        let tools = result.expect(
            "Defect #2 REGRESSION: discovery FAILED on a server that emits \
             non-JSON banners before valid JSON-RPC (must skip + continue)",
        );
        assert_eq!(tools.len(), 1, "expected exactly the stub tool");
        assert_eq!(tools[0].name, "stub_tool");
    }

    // ===================================================================
    // Defect #3 — MCP Streamable HTTP transport (spec 2025-03-26)
    // ===================================================================

    /// (b) SSE-framed response: a JSON-RPC result delivered across
    /// multi-line `data:` events is correctly assembled and the message
    /// whose `id` matches is returned (others skipped). Pure parser test
    /// (`parse_sse_for_jsonrpc`) — no network.
    #[test]
    fn test_streamable_http_sse_frame_parsing_assembles_jsonrpc() {
        // Two events: an unrelated server notification, then the real
        // response split across two `data:` lines (joined with '\n').
        let body = "\
event: message\n\
data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\
\n\
: a comment line is ignored\n\
data: {\"jsonrpc\":\"2.0\",\"id\":2,\n\
data: \"result\":{\"tools\":[]}}\n\
\n";
        let got = parse_sse_for_jsonrpc(body, Some(2))
            .expect("SSE stream must yield the id=2 JSON-RPC message");
        assert_eq!(got.get("id").and_then(|i| i.as_i64()), Some(2));
        assert!(got.get("result").is_some(), "assembled result present");
        // The earlier notification (no matching id) must NOT be returned.
        assert_ne!(
            got.get("method").and_then(|m| m.as_str()),
            Some("notifications/progress")
        );
    }

    /// (a) application/json single-response mode: `body_to_jsonrpc` parses
    /// a plain JSON object directly (still valid Streamable HTTP). And
    /// (b cross-check) the same helper handles text/event-stream.
    #[test]
    fn test_streamable_http_json_vs_eventstream_branch() {
        // application/json ⇒ parse the single object directly.
        let j = StreamableHttpSession::body_to_jsonrpc(
            "application/json",
            r#"{"jsonrpc":"2.0","id":2,"result":{"ok":true}}"#,
            2,
            "srv",
        )
        .expect("json mode parses");
        assert_eq!(j["result"]["ok"], serde_json::json!(true));

        // text/event-stream ⇒ SSE-framed, same logical result.
        let s = StreamableHttpSession::body_to_jsonrpc(
            "text/event-stream; charset=utf-8",
            "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}\n\n",
            2,
            "srv",
        )
        .expect("event-stream mode parses");
        assert_eq!(s["result"]["ok"], serde_json::json!(true));
    }

    /// (c) `Mcp-Session-Id` captured from the initialize response header
    /// and re-sent on the NEXT request; plus (d) bounded behavior — a
    /// never-completing SSE stream times out (no hang). Uses a hand-rolled
    /// single-thread TCP stub (std only, no network, no new dep).
    #[test]
    fn test_streamable_http_session_id_capture_resend_and_timeout() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/mcp");
        // The stub records the Mcp-Session-Id header seen on request #2.
        let (tx, rx) = mpsc::channel::<Option<String>>();

        let handle = std::thread::spawn(move || {
            // ---- Request 1: initialize. Reply application/json + set
            // the Mcp-Session-Id response header. ----
            let (mut s1, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = s1.read(&mut buf);
            let init_body = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}"#;
            let resp1 = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: SESS-XYZ-42\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                init_body.len(),
                init_body
            );
            s1.write_all(resp1.as_bytes()).unwrap();
            drop(s1);

            // ---- Request 2: tools/list. Capture the Mcp-Session-Id the
            // client sent, then reply with an SSE stream that NEVER ends
            // (no terminating bytes) so the client's 2s timeout must fire
            // — proving bounded/no-hang. ----
            let (mut s2, _) = listener.accept().unwrap();
            let mut req2 = Vec::new();
            let mut tmp = [0u8; 2048];
            // Read just the headers (one read is enough for our tiny req).
            let n = s2.read(&mut tmp).unwrap_or(0);
            req2.extend_from_slice(&tmp[..n]);
            let req2s = String::from_utf8_lossy(&req2).to_lowercase();
            let seen = req2s
                .lines()
                .find(|l| l.starts_with("mcp-session-id:"))
                .map(|l| l["mcp-session-id:".len()..].trim().to_string());
            let _ = tx.send(seen);
            // Send headers + a partial SSE event, then hang (no close, no
            // blank line) → client read times out.
            let _ = s2.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"partial\":true}\n",
            );
            // Keep the socket open (do NOT drop) until the client times out.
            std::thread::sleep(std::time::Duration::from_secs(4));
            drop(s2);
        });

        // Client: 2s timeout (bounded — proves no hang on the stuck stream).
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let mut session = StreamableHttpSession::new();

        // Req 1: initialize → must capture the session id from the header.
        let init = session
            .post_request(&client, &url, "stub", 1, "initialize", json!({}))
            .expect("initialize should parse (application/json)");
        assert_eq!(init["result"]["protocolVersion"], serde_json::json!("2025-03-26"));
        assert_eq!(
            session.session_id.as_deref(),
            Some("SESS-XYZ-42"),
            "Mcp-Session-Id MUST be captured from the initialize response header"
        );

        // Req 2: tools/list → the stub hangs the SSE stream; the bounded
        // client timeout MUST turn this into an Err (NOT a hang).
        let start = std::time::Instant::now();
        let r2 = session.post_request(&client, &url, "stub", 2, "tools/list", json!({}));
        let elapsed = start.elapsed();
        assert!(r2.is_err(), "a never-completing SSE stream must error, not hang");
        assert!(
            elapsed < std::time::Duration::from_secs(8),
            "must be bounded by the client timeout (took {elapsed:?})"
        );

        // The stub saw request #2 carry the captured session id.
        let seen = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("stub should have recorded request 2 headers");
        assert_eq!(
            seen.as_deref(),
            Some("sess-xyz-42"),
            "the captured Mcp-Session-Id MUST be re-sent on the next request"
        );

        let _ = handle.join();
    }

    // ===================================================================
    // R1 — portal-stall: router read-guard MUST NOT span the backend await
    // ===================================================================

    /// Proves the fix is real AND that the bug it fixes is real.
    ///
    /// Setup: a real `Arc<RwLock<ToolRouter>>` with one HTTP tool whose
    /// endpoint is an unroutable TEST-NET addr, so a dispatch blocks for a
    /// bounded-but-clearly-observable time (connect attempt). We then
    /// compare two patterns under a concurrent writer + independent reader:
    ///
    ///  - BROKEN (old code): hold `router.read().await` ACROSS the slow
    ///    dispatch await. tokio RwLock is write-preferring, so a concurrent
    ///    `router.write().await` queues behind the long read and then
    ///    starves a subsequent independent `router.read().await` (a health
    ///    ping) for the WHOLE slow-call duration → asserts the ping is
    ///    delayed > the stall threshold (the bug exists).
    ///
    ///  - FIXED (new code, exactly what server.rs now does): briefly take
    ///    the read guard, `resolve_call_target`, DROP the guard, THEN
    ///    `dispatch` lock-free → asserts the concurrent write AND the
    ///    independent ping both complete PROMPTLY (well under the
    ///    slow-call duration) while the dispatch is still in flight.
    ///
    /// Deterministic + bounded: the "slow call" is a fixed 1.2s sleep
    /// stand-in (no real network flakiness); thresholds have wide margins.
    /// No process spawn ⇒ plain test (no REAPER_TEST_LOCK needed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_r1_router_read_guard_not_held_across_backend_await() {
        use std::time::{Duration, Instant};
        use tokio::sync::RwLock as TokioRwLock;

        // Build a minimal real router (we only exercise the lock scope;
        // the actual backend call is modelled by a fixed sleep so the test
        // is deterministic and network-free).
        let mut router = ToolRouter::new();
        router
            .register_http_server(HttpServerConfig {
                name: "slowsrv".into(),
                url: "http://192.0.2.1:9/mcp".into(), // TEST-NET-1, unroutable
                env: HashMap::new(),
            })
            .unwrap();
        router.register_tool(Tool {
            name: "slow_tool".into(),
            description: "d".into(),
            input_schema: json!({"type":"object"}),
            server: "slowsrv".into(),
        });
        let router = std::sync::Arc::new(TokioRwLock::new(router));

        const SLOW: Duration = Duration::from_millis(1200);
        // Generous: the lock-free path must let an independent op through
        // in a tiny fraction of SLOW; the broken path starves it ~SLOW.
        const PROMPT_MAX: Duration = Duration::from_millis(400);
        const STALL_MIN: Duration = Duration::from_millis(900);

        // ---- (1) FIXED pattern: resolve under brief guard, DROP, then
        // run the slow op lock-free. ----
        {
            let r = router.clone();
            let slow = tokio::spawn(async move {
                // Exactly server.rs's new shape: brief guard → resolve →
                // drop → lock-free await.
                let _resolved = {
                    let g = r.read().await;
                    g.resolve_call_target("slow_tool")
                }; // guard dropped HERE, before the await
                tokio::time::sleep(SLOW).await; // stands in for dispatch()
            });
            // Give the slow task time to pass the (brief) guard scope.
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Concurrent hot-swap writer + independent health-ping reader.
            let t0 = Instant::now();
            {
                let _w = router.write().await; // hot-swap acquisition
            }
            let _ = router.read().await; // independent "ping"
            let elapsed = t0.elapsed();
            assert!(
                elapsed < PROMPT_MAX,
                "R1 FIX REGRESSION: with the read guard dropped before the \
                 backend await, a concurrent write + independent ping must \
                 be PROMPT (got {elapsed:?}, slow call still in flight)"
            );
            slow.await.unwrap();
        }

        // ---- (2) BROKEN pattern (proves the bug is real on cd72a9c):
        // hold the read guard ACROSS the slow await. ----
        {
            let r = router.clone();
            let slow = tokio::spawn(async move {
                let _g = r.read().await; // guard HELD across the await...
                tokio::time::sleep(SLOW).await; // ...the original bug
                drop(_g);
            });
            tokio::time::sleep(Duration::from_millis(50)).await;

            let t0 = Instant::now();
            {
                let _w = router.write().await; // queues behind the long read
            }
            let _ = router.read().await; // starved (write-preferring)
            let elapsed = t0.elapsed();
            assert!(
                elapsed >= STALL_MIN,
                "test invalid: the BROKEN pattern must demonstrably stall \
                 (got {elapsed:?}); proves the test really distinguishes \
                 held-across-await from the fix"
            );
            slow.await.unwrap();
        }
    }

    // ===================================================================
    // Discovery-surface unified redesign — P1/A1 two-tier, P2 minifier,
    // P3/A2 ordering+ranking. Pure metadata/ranking — no process spawn.
    // ===================================================================

    fn t(name: &str, desc: &str, server: &str) -> Tool {
        Tool {
            name: name.into(),
            description: desc.into(),
            input_schema: json!({"type":"object","properties":{}}),
            server: server.into(),
        }
    }

    /// A2 (accuracy #1+#6) — keyword search: a NAME match MUST rank
    /// strictly above a DESCRIPTION-only match, and the order MUST be
    /// deterministic (stable across repeated calls). On 37a567a this
    /// would be FLAKY/WRONG: the old code did `.values().filter().take()`
    /// over a HashMap with NO ranking → `write_file` (whose description
    /// mentions "read") could outrank `read_file`, and order varied by
    /// HashMap seed. This test discriminates: it asserts read_file FIRST
    /// AND identical order across 5 runs — both fail on the old behavior.
    #[test]
    fn test_listtools_keyword_name_ranks_above_desc_and_is_deterministic() {
        let mut r = ToolRouter::new();
        // write_file's DESCRIPTION contains "read" (the query) but its
        // NAME does not; read_file's NAME contains "read".
        r.register_tool(t("write_file", "Write a file; supports read-only checks", "fsx"));
        r.register_tool(t("read_file", "Fetch file bytes", "fsx"));
        r.register_tool(t("aaa_reader", "unrelated", "fsx")); // also a name-match

        let q = Some("read".to_string());
        let first_run = r.list_tools(q.clone(), false, 10);
        let names: Vec<&str> = first_run.iter().map(|x| x.name.as_str()).collect();

        // read_file & aaa_reader are NAME matches (rank 0), write_file is
        // DESC-only (rank 1) → both name-matches precede write_file, and
        // within rank 0 the secondary sort is by name (aaa_reader < read_file).
        let wf = names.iter().position(|n| *n == "write_file").unwrap();
        let rf = names.iter().position(|n| *n == "read_file").unwrap();
        let ar = names.iter().position(|n| *n == "aaa_reader").unwrap();
        assert!(
            rf < wf && ar < wf,
            "A2 REGRESSION: name-match (read_file/aaa_reader) MUST rank \
             above description-only (write_file); got {names:?}"
        );
        assert_eq!(
            (ar, rf),
            (0, 1),
            "deterministic secondary sort by name within name-match rank"
        );

        // Determinism: 5 repeated calls on the same state ⇒ identical order.
        for _ in 0..5 {
            let again: Vec<String> =
                r.list_tools(q.clone(), false, 10).into_iter().map(|x| x.name).collect();
            let prev: Vec<String> =
                first_run.iter().map(|x| x.name.clone()).collect();
            assert_eq!(again, prev, "list_tools order MUST be deterministic");
        }
    }

    /// P3 — no-query path is a DETERMINISTIC server-grouped capability map
    /// (local first, then external servers by name, then builtins), tools
    /// alphabetical within group. On 37a567a it was a flat alphabetical
    /// slice with no grouping AND HashMap-collected (nondeterministic
    /// pre-sort) — this asserts the grouped order + stability.
    #[test]
    fn test_listtools_no_query_is_deterministic_server_grouped() {
        let mut r = ToolRouter::new();
        r.register_tool(t("zzz_local", "z", "local"));
        r.register_tool(t("aaa_local", "a", "local"));
        r.register_tool(t("ext_b", "b", "srvB"));
        r.register_tool(t("ext_a", "a", "srvA"));
        r.register_tool(t("meta1", "m", "builtin"));

        let run = |r: &ToolRouter| -> Vec<String> {
            r.list_tools(None, false, 100).into_iter().map(|x| x.name).collect()
        };
        let order = run(&r);
        // local group first (alpha within), then external (server-name
        // order srvA<srvB), then builtin last.
        assert_eq!(
            order,
            vec!["aaa_local", "zzz_local", "ext_a", "ext_b", "meta1"],
            "P3: no-query order must be local→external(by server)→builtin, \
             alpha within group; got {order:?}"
        );
        for _ in 0..5 {
            assert_eq!(run(&r), order, "no-query order MUST be deterministic");
        }
    }

    /// P1/A1 — two-tier round-trip: list_tools gives name + SHORT desc
    /// (NO inputSchema), and get_tool_info(name) returns the FULL schema
    /// sufficient to build args. Proves no agent is left blind.
    #[test]
    fn test_two_tier_list_then_get_tool_info_roundtrip() {
        let mut r = ToolRouter::new();
        r.register_tool(Tool {
            name: "send_email".into(),
            description: "Send an email message. Supports CC, BCC, and \
                          attachments via the args object; rate-limited."
                .into(),
            input_schema: json!({
                "type":"object",
                "title":"SendEmailArgs",
                "additionalProperties": false,
                "properties":{
                    "to":{"type":"string","description":"recipient"},
                    "subject":{"type":"string"}
                },
                "required":["to"]
            }),
            server: "mail".into(),
        });

        // Tier 1: list (slim). short_description must be the FIRST
        // sentence only, materially shorter than the full prose.
        let listed = r.list_tools(Some("email".into()), false, 10);
        assert_eq!(listed.len(), 1);
        let short = short_description(&listed[0].description);
        assert_eq!(short, "Send an email message");
        assert!(
            short.len() < listed[0].description.len(),
            "short desc must be materially smaller than full prose"
        );

        // Tier 2: get_tool_info → FULL schema, still sufficient to build a
        // call (properties + required preserved), display-only/default
        // noise minified away.
        let info = r.get_tool_info("send_email").expect("known tool");
        let sch = &info["inputSchema"];
        assert_eq!(info["description"], json!(
            "Send an email message. Supports CC, BCC, and \
             attachments via the args object; rate-limited."
        ), "get_tool_info returns the FULL description");
        assert!(sch["properties"]["to"].is_object(), "args still constructible");
        assert_eq!(sch["required"], json!(["to"]), "required preserved");
        assert!(sch.get("title").is_none(), "P2: display-only title stripped");
        assert!(
            sch.get("additionalProperties").is_none(),
            "P2: default-equivalent additionalProperties:false stripped"
        );
        assert!(r.get_tool_info("nonexistent").is_none(), "unknown ⇒ None");
    }

    /// P2 — minifier preserves schema VALIDITY + sufficiency, strips only
    /// default-equivalent/display noise, and is DETERMINISTIC + idempotent.
    #[test]
    fn test_minify_schema_preserves_validity_and_is_idempotent() {
        let schema = json!({
            "type":"object",
            "title":"X",
            "additionalProperties": false,
            "required": [],
            "properties":{
                "path":{"type":"string","description":"the path","title":"Path"},
                "deep":{
                    "type":"object",
                    "additionalProperties": false,
                    "properties":{"k":{"type":"number"}}
                }
            }
        });
        let m = minify_schema(&schema);
        // Stripped: title (all levels), additionalProperties:false (all
        // levels), empty required.
        assert!(m.get("title").is_none());
        assert!(m.get("additionalProperties").is_none());
        assert!(m.get("required").is_none(), "empty required removed");
        assert!(m["properties"]["path"].get("title").is_none());
        // PRESERVED (sufficiency): structure agents need to build args.
        assert_eq!(m["type"], json!("object"));
        assert_eq!(m["properties"]["path"]["type"], json!("string"));
        assert_eq!(m["properties"]["path"]["description"], json!("the path"));
        assert_eq!(m["properties"]["deep"]["properties"]["k"]["type"], json!("number"));
        // A NON-default additionalProperties value is KEPT (not over-trimmed).
        let keep = json!({"type":"object","additionalProperties":{"type":"string"}});
        assert_eq!(minify_schema(&keep), keep, "non-false additionalProperties kept");
        // Non-empty required KEPT.
        let req = json!({"type":"object","required":["a"],"properties":{}});
        assert_eq!(minify_schema(&req)["required"], json!(["a"]));
        // Deterministic + idempotent.
        assert_eq!(minify_schema(&schema), m, "deterministic");
        assert_eq!(minify_schema(&m), m, "idempotent: minify(minify(x))==minify(x)");
    }

    /// short_description determinism + safety (multibyte, long, no period).
    #[test]
    fn test_short_description_deterministic_and_safe() {
        assert_eq!(short_description("One. Two. Three."), "One");
        assert_eq!(short_description("No trailing period"), "No trailing period");
        assert_eq!(short_description("Trailing period only."), "Trailing period only");
        let long = "x".repeat(500);
        let s = short_description(&long);
        assert!(s.chars().count() <= 120, "hard-capped");
        assert_eq!(short_description(&long), s, "deterministic");
        // Multibyte must not panic / split a char.
        let mb = format!("{} sentence. more", "é".repeat(200));
        let _ = short_description(&mb); // must not panic
    }

    /// Tool-dictionary provenance round-trip: when a dict entry applies
    /// for a given (server, tool), `get_tool_info` returns
    /// `_source: "dict"` and the curated description; when no entry
    /// applies, the JSON shape is unchanged (no `_source` key — back-compat
    /// for downstream consumers of `get_tool_info` JSON).
    /// Also validates `iter_tools_for_doctor` + `apply_dict_overlay`
    /// (the surface `--doctor-tools` uses) report the same provenance.
    #[test]
    fn test_doctor_tools_and_get_tool_info_show_source_provenance() {
        use crate::server::tool_dictionary::{ShortMode, ToolDictionary};
        // Build a custom dict in a tempdir with ONE entry matching the
        // tool we'll register; the loader's compile-in fallback contains
        // 15 seed servers but none match `myserver/foo`, so this dict
        // overlay is the only path that applies.
        let tmp = tempfile::tempdir().expect("tmpdir");
        std::fs::write(
            tmp.path().join("myserver.toml"),
            r#"
[metadata]
server_name = "myserver"

[[tools]]
name = "foo"
short_description = "CURATED-SHORT"
"#,
        )
        .expect("write fixture");
        let dict = ToolDictionary::load(&[], Some(tmp.path()));
        let mut r = ToolRouter::new().with_tool_dictionary(dict, ShortMode::On);
        r.register_tool(Tool {
            name: "foo".into(),
            description: "Full live description of foo.".into(),
            input_schema: json!({"type":"object","properties":{}}),
            server: "myserver".into(),
        });
        r.register_tool(Tool {
            name: "bar".into(),
            description: "Full live description of bar.".into(),
            input_schema: json!({"type":"object","properties":{}}),
            server: "myserver".into(),
        });

        // get_tool_info on `foo` → dict applies, _source:"dict" present,
        // description = curated short form.
        let foo = r.get_tool_info("foo").expect("foo present");
        assert_eq!(foo["description"], json!("CURATED-SHORT"));
        assert_eq!(foo["_source"], json!("dict"));

        // get_tool_info on `bar` → no dict entry, no _source key
        // (back-compat: JSON shape unchanged for non-overridden tools).
        let bar = r.get_tool_info("bar").expect("bar present");
        assert_eq!(bar["description"], json!("Full live description of bar."));
        assert!(
            bar.get("_source").is_none(),
            "back-compat: no _source key when dict did not apply"
        );

        // iter_tools_for_doctor + apply_dict_overlay — the `--doctor-tools`
        // surface — agrees with `get_tool_info`'s provenance.
        let mut seen: std::collections::HashMap<String, &'static str> =
            std::collections::HashMap::new();
        for (server, name, live_desc) in r.iter_tools_for_doctor() {
            let (_desc, src) = r.apply_dict_overlay(server, name, live_desc);
            seen.insert(format!("{server}/{name}"), src);
        }
        assert_eq!(seen.get("myserver/foo"), Some(&"dict"));
        assert_eq!(seen.get("myserver/bar"), Some(&"server"));
    }

    // ===================================================================
    // R2 — hot-swap remove/update evicts the stale pooled connection
    // A3 — external↔external tool-name collision emits a warn!
    // ===================================================================

    /// R2: after a hot-swap server REMOVE (the exact path:
    /// `unregister_server` then `connection_pool().remove(name)`), the
    /// previously-pooled connection is gone from the pool AND its real
    /// child process is reaped by the EXISTING hardened
    /// `ServerConnection::Drop` — not left stale until idle-TTL. Also
    /// asserts a subsequent `get_or_create`-style lookup would NOT find
    /// the old connection (pool map no longer has the entry). Spawns a
    /// real child ⇒ REAPER_TEST_LOCK-gated.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_r2_hotswap_remove_evicts_stale_pooled_connection() {
        let _serial = REAPER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut router = ToolRouter::new();
        // Register a stdio server + a tool on it (mirrors discovery).
        router
            .register_server(ServerConfig {
                name: "victim".into(),
                command: "sleep".into(),
                args: vec!["300".into()],
                env: HashMap::new(),
            })
            .unwrap();
        router.register_tool(Tool {
            name: "victim_tool".into(),
            description: "d".into(),
            input_schema: json!({"type":"object"}),
            server: "victim".into(),
        });

        // Put a real-process-backed connection into the pool under that
        // server name (what get_or_create would have created on first call).
        let pool = router.connection_pool().clone();
        let (conn, pid) = make_test_conn("victim", std::time::Duration::from_secs(0));
        pool.connections.lock().await.insert("victim".into(), conn);
        assert!(pid_alive(pid), "precondition: pooled child alive");
        assert!(
            pool.connections.lock().await.contains_key("victim"),
            "precondition: connection pooled"
        );

        // THE hot-swap REMOVE path, exactly as hot_swap.rs now does it.
        router.unregister_server("victim");
        router.connection_pool().remove("victim").await;

        // Pool entry gone → a subsequent call can NOT route to the stale
        // conn (get_or_create would rebuild fresh).
        assert!(
            !pool.connections.lock().await.contains_key("victim"),
            "R2 REGRESSION: stale pooled connection NOT evicted on \
             hot-swap remove — agent would keep routing to the old process"
        );
        // The existing hardened Drop fired on remove() → child reaped.
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            !pid_alive(pid),
            "R2 REGRESSION: old backend child (pid {pid}) survived — \
             eviction must reuse the existing ServerConnection::Drop \
             teardown (no stale lingering process)"
        );
        // Tool also unregistered (server-not-found rather than stale route).
        assert!(
            router.get_tool_info("victim_tool").is_none(),
            "tool removed with its server"
        );
    }

    /// A3: registering an external tool whose name collides with another
    /// EXTERNAL server's tool emits a `warn!` naming BOTH servers + the
    /// tool (was a silent last-wins overwrite). Captures tracing output
    /// via a scoped custom MakeWriter (std + existing tracing-subscriber
    /// dep — no new dependency). Also asserts builtin/local precedence is
    /// UNCHANGED (no warn, still skipped) and last-wins still applies.
    #[test]
    fn test_a3_external_collision_emits_warn() {
        use std::io::Write as _;
        use std::sync::{Arc as StdArc, Mutex as StdMutex};

        #[derive(Clone)]
        struct BufWriter(StdArc<StdMutex<Vec<u8>>>);
        impl Write for BufWriter {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buf = StdArc::new(StdMutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(BufWriter(buf.clone()))
            .with_max_level(tracing::Level::WARN)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let mut router = ToolRouter::new();
            let mk = |srv: &str| Tool {
                name: "read_file".into(),
                description: "d".into(),
                input_schema: json!({"type":"object"}),
                server: srv.into(),
            };
            // First external registration — no collision yet, no warn.
            router.register_tool(mk("filesystem"));
            // Second external server, SAME tool name → collision warn.
            router.register_tool(mk("sandbox-fs"));
            // builtin/local precedence path must NOT warn (still skipped).
            router.register_tool(Tool {
                name: "read_file_local".into(),
                description: "d".into(),
                input_schema: json!({"type":"object"}),
                server: "local".into(),
            });
            router.register_tool(Tool {
                name: "read_file_local".into(),
                description: "d".into(),
                input_schema: json!({"type":"object"}),
                server: "external".into(),
            });
            // (server,name)-keyed storage: BOTH external `read_file`
            // entries COEXIST (no silent last-wins). Each is reachable
            // by `(server, name)`.
            assert!(
                router.tools.contains_key(&("filesystem".to_string(), "read_file".to_string())),
                "filesystem.read_file MUST still be reachable after the second registration"
            );
            assert!(
                router.tools.contains_key(&("sandbox-fs".to_string(), "read_file".to_string())),
                "sandbox-fs.read_file MUST coexist with filesystem.read_file"
            );
        });

        let logged = String::from_utf8_lossy(
            &buf.lock().unwrap_or_else(|e| e.into_inner()),
        )
        .to_string();
        assert!(
            logged.contains("read_file")
                && logged.contains("filesystem")
                && logged.contains("sandbox-fs")
                && (logged.contains("BOTH") || logged.contains("disambiguate")),
            "A3 REGRESSION: external↔external collision must warn! naming \
             BOTH servers + the tool; captured log was: {logged:?}"
        );
        // The local-vs-external precedence skip must NOT have warned.
        assert!(
            !logged.contains("read_file_local"),
            "builtin/local precedence path must remain a silent skip (no \
             collision warn) — existing intended behavior unchanged"
        );
    }

    // ===================================================================
    // (server, name)-keyed registry — A3 Part B: BOTH same-named external
    // tools coexist + are routable via the optional `server` hint
    // ===================================================================

    /// Helper: build a stdio `ServerConfig` for tests. The dispatch path
    /// uses the config's name to look up the live server; we don't
    /// actually call the backend in these tests (we resolve only).
    fn srv_cfg(name: &str) -> ServerConfig {
        ServerConfig {
            name: name.into(),
            command: "sleep".into(),
            args: vec!["1".into()],
            env: HashMap::new(),
        }
    }

    /// THE distinguishing trio for the server-keyed feature:
    /// (a) two external servers each register `fetch` → BOTH coexist in
    ///     storage AND in list_tools (with distinct descriptions);
    /// (b) `resolve_call_target("fetch", None)` (no hint, ambiguous) →
    ///     CLEAR ambiguous-tool error naming both candidate servers (NOT
    ///     a silent last-wins);
    /// (c) hinted resolution routes to the EXACTLY-named server (each
    ///     direction works).
    /// Would FAIL on `daec932` (single name-keyed HashMap → second
    ///   `fetch` overwrote the first; ambiguous-hint logic did not exist;
    ///   no `server`-arg routing was possible).
    /// PASSES here.
    #[test]
    fn test_server_keyed_two_external_same_name_both_routable() {
        let mut r = ToolRouter::new();
        r.register_server(srv_cfg("alpha")).unwrap();
        r.register_server(srv_cfg("beta")).unwrap();
        r.register_tool(Tool {
            name: "fetch".into(),
            description: "alpha-fetch: HTTP GET via alpha".into(),
            input_schema: json!({"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}),
            server: "alpha".into(),
        });
        r.register_tool(Tool {
            name: "fetch".into(),
            description: "beta-fetch: cached fetch via beta".into(),
            input_schema: json!({"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}),
            server: "beta".into(),
        });

        // (a) BOTH coexist in storage.
        assert!(r.tools.contains_key(&("alpha".to_string(), "fetch".to_string())));
        assert!(r.tools.contains_key(&("beta".to_string(), "fetch".to_string())));
        // list_tools shows BOTH with their distinct descriptions.
        let listed = r.list_tools(Some("fetch".into()), false, 10);
        let fetches: Vec<&Tool> =
            listed.iter().filter(|t| t.name == "fetch").collect();
        assert_eq!(fetches.len(), 2, "BOTH fetch tools must appear in list_tools");
        let mut descs: Vec<String> = fetches.iter().map(|t| t.description.clone()).collect();
        descs.sort();
        assert_eq!(
            descs,
            vec![
                "alpha-fetch: HTTP GET via alpha".to_string(),
                "beta-fetch: cached fetch via beta".to_string()
            ],
            "each entry MUST carry its OWN distinct description"
        );

        // (b) No-hint resolution on a collided name → clear ambiguous-tool
        // error naming BOTH candidates (NOT silent last-wins).
        let err = r
            .resolve_call_target("fetch")
            .err()
            .expect("collided name with no hint MUST error");
        let msg = format!("{err}");
        assert!(
            msg.contains("ambiguous") && msg.contains("alpha") && msg.contains("beta")
                && msg.contains("server"),
            "ambiguous error MUST list both candidate servers + the disambiguation hint; got: {msg}"
        );

        // (c) Hinted resolution routes to the EXACT (server, name) pair —
        // each direction works.
        let to_alpha = r
            .resolve_call_target_with_hint("fetch", Some("alpha"))
            .expect("hinted resolve to alpha must succeed");
        let to_beta = r
            .resolve_call_target_with_hint("fetch", Some("beta"))
            .expect("hinted resolve to beta must succeed");
        match (to_alpha, to_beta) {
            (
                ResolvedCall::Stdio { server: sa, .. },
                ResolvedCall::Stdio { server: sb, .. },
            ) => {
                assert_eq!(sa.name, "alpha", "alpha hint MUST resolve to alpha");
                assert_eq!(sb.name, "beta", "beta hint MUST resolve to beta");
            }
            _ => panic!("expected two Stdio Resolved targets"),
        }

        // Bad hint → clean error naming the known candidates.
        let bad = r
            .resolve_call_target_with_hint("fetch", Some("gamma"))
            .err()
            .expect("hint to unknown server must error");
        let bmsg = format!("{bad}");
        assert!(
            bmsg.contains("not found on server 'gamma'") && bmsg.contains("alpha") && bmsg.contains("beta"),
            "bad-hint error MUST list known candidate servers; got: {bmsg}"
        );
    }

    /// get_tool_info on a collided name returns the `{matches: [...]}`
    /// envelope with both entries (so the agent can read each
    /// description + server before picking). Single-match shape unchanged.
    #[test]
    fn test_server_keyed_get_tool_info_collision_shape() {
        let mut r = ToolRouter::new();
        r.register_server(srv_cfg("alpha")).unwrap();
        r.register_server(srv_cfg("beta")).unwrap();
        r.register_tool(Tool {
            name: "fetch".into(),
            description: "alpha-fetch".into(),
            input_schema: json!({"type":"object","properties":{"url":{"type":"string"}}}),
            server: "alpha".into(),
        });
        r.register_tool(Tool {
            name: "fetch".into(),
            description: "beta-fetch".into(),
            input_schema: json!({"type":"object","properties":{"url":{"type":"string"}}}),
            server: "beta".into(),
        });
        // Also a unique-name tool to verify single-match shape stays as before.
        r.register_tool(Tool {
            name: "unique_tool".into(),
            description: "u".into(),
            input_schema: json!({"type":"object"}),
            server: "alpha".into(),
        });

        let info = r.get_tool_info("fetch").expect("known collided name");
        assert_eq!(info["ambiguous"], json!(true), "collision sets `ambiguous: true`");
        assert_eq!(info["name"], json!("fetch"));
        let m = info["matches"].as_array().expect("matches array");
        assert_eq!(m.len(), 2, "BOTH candidates returned");
        let servers: Vec<&str> =
            m.iter().map(|e| e["server"].as_str().unwrap()).collect();
        assert_eq!(servers, vec!["alpha", "beta"], "deterministic order by server");
        assert!(info["hint"].as_str().unwrap().contains("server: <name>"));

        // Single-match: shape unchanged (no `matches` envelope).
        let one = r.get_tool_info("unique_tool").expect("known unique name");
        assert_eq!(one["name"], json!("unique_tool"));
        assert_eq!(one["server"], json!("alpha"));
        assert!(one.get("matches").is_none(), "single match keeps the flat shape");
        assert!(one.get("ambiguous").is_none());

        // Unknown name → None.
        assert!(r.get_tool_info("nonexistent").is_none());
    }

    /// Single-name (no collision) call still routes WITHOUT a hint
    /// (back-compat for the common case).
    #[test]
    fn test_server_keyed_unique_name_no_hint_still_routes() {
        let mut r = ToolRouter::new();
        r.register_server(srv_cfg("only_srv")).unwrap();
        r.register_tool(Tool {
            name: "uniq".into(),
            description: "u".into(),
            input_schema: json!({"type":"object"}),
            server: "only_srv".into(),
        });
        let res = r.resolve_call_target("uniq").expect("unique name resolves with no hint");
        match res {
            ResolvedCall::Stdio { server, .. } => {
                assert_eq!(server.name, "only_srv");
            }
            _ => panic!("expected Stdio Resolved target"),
        }
    }

    /// Builtin/local precedence UNCHANGED: a `local` tool with the same
    /// name as an external one shadows the external (the external is
    /// skipped on registration by the existing guard).
    #[test]
    fn test_server_keyed_builtin_local_precedence_unchanged() {
        let mut r = ToolRouter::new();
        r.register_tool(Tool {
            name: "read_file".into(),
            description: "local read".into(),
            input_schema: json!({"type":"object"}),
            server: "local".into(),
        });
        // External registration of the same name MUST be skipped (existing behavior).
        r.register_server(srv_cfg("external")).unwrap();
        r.register_tool(Tool {
            name: "read_file".into(),
            description: "external read (must be skipped)".into(),
            input_schema: json!({"type":"object"}),
            server: "external".into(),
        });
        // Only the local entry exists.
        assert!(r.tools.contains_key(&("local".to_string(), "read_file".to_string())));
        assert!(
            !r.tools.contains_key(&("external".to_string(), "read_file".to_string())),
            "external registration MUST be skipped when local already owns the name"
        );
        // Resolves to Local without ambiguity.
        let res = r.resolve_call_target("read_file").expect("local resolves");
        assert!(matches!(res, ResolvedCall::Local { .. }));
    }
}
