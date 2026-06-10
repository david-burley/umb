<!--
Thanks for contributing to UMB. Please read CONTRIBUTING.md first.
-->

## What does this change?

<!-- A short description of the change and why. -->

## Checklist

- [ ] `cargo test` passes (default features)
- [ ] `cargo test --features embed-onnx` passes (if semantic search was touched)
- [ ] No new compiler warnings beyond the documented 14
- [ ] Diff is focused; docs match the existing terse tone
- [ ] I licensed this contribution under PolyForm NC 1.0.0 (inbound = outbound)

## Daemon / process-handling surface?

- [ ] This PR touches daemon/proxy mode, stdio transport, process
      spawning/reaping, the connection pool, idle eviction, or hot-swap
      teardown.

> If checked: `cargo test` is **not** sufficient for this surface (it has
> false-greened real defects before). Maintainers will run real-Linux E2E
> validation before merge — see CONTRIBUTING.md.
