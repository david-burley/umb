#!/bin/bash
# =============================================================================
# UMB-DEV Parallel Multi-Platform Build Script
# =============================================================================
# Builds all platform binaries in PARALLEL:
#   - macOS ARM64     (native cross-compile)
#   - macOS x64       (native cross-compile)
#   - macOS Universal (lipo, waits for both macOS builds)
#   - Linux x64       (Docker)
#   - Linux ARM64     (Docker buildx + QEMU)
#
# Compatible with Bash 3.2+ (macOS default)
#
# Requirements:
#   - Rust toolchain (rustup)
#   - Docker Desktop (for Linux builds)
#   - Targets: aarch64-apple-darwin, x86_64-apple-darwin
#
# Usage:
#   ./build-all-platforms.sh              # Build all platforms (parallel)
#   ./build-all-platforms.sh --macos      # macOS only (arm64 + x64 + universal)
#   ./build-all-platforms.sh --linux      # Linux only (x64 + arm64 Docker)
#   ./build-all-platforms.sh --install    # Build macOS ARM64 + install to /usr/local/bin
#   ./build-all-platforms.sh --deploy     # Build all + deploy to web server
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$PROJECT_DIR/dist"
LOG_DIR="$PROJECT_DIR/dist/logs"

# Extract version from Cargo.toml
VERSION=$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')

# Track build results (Bash 3.2 compatible - using status files)
BUILD_START=$(date +%s)
FAILED=false

# =============================================================================
# Helpers
# =============================================================================

log()    { echo "[$(date +%H:%M:%S)] $*"; }
ok()     { echo "[$(date +%H:%M:%S)] OK: $*"; }
fail()   { echo "[$(date +%H:%M:%S)] FAIL: $*"; }
header() { echo ""; echo "============================================"; echo "  $*"; echo "============================================"; }

mark_success() { touch "$LOG_DIR/$1.success"; }
mark_fail() { touch "$LOG_DIR/$1.fail"; }
is_success() { [ -f "$LOG_DIR/$1.success" ]; }
is_fail() { [ -f "$LOG_DIR/$1.fail" ]; }

# =============================================================================
# Build Functions
# =============================================================================

build_macos_arm64() {
    log "Starting macOS ARM64 build..."
    rustup target add aarch64-apple-darwin 2>/dev/null || true
    cargo build --release --target aarch64-apple-darwin
    cp "$PROJECT_DIR/target/aarch64-apple-darwin/release/umb" "$OUTPUT_DIR/umb-macos-arm64"
    local size=$(ls -lh "$OUTPUT_DIR/umb-macos-arm64" | awk '{print $5}')
    log "macOS ARM64 done ($size)"
}

build_macos_x64() {
    log "Starting macOS x64 build..."
    rustup target add x86_64-apple-darwin 2>/dev/null || true
    cargo build --release --target x86_64-apple-darwin
    cp "$PROJECT_DIR/target/x86_64-apple-darwin/release/umb" "$OUTPUT_DIR/umb-macos-x64"
    local size=$(ls -lh "$OUTPUT_DIR/umb-macos-x64" | awk '{print $5}')
    log "macOS x64 done ($size)"
}

build_macos_universal() {
    log "Creating macOS Universal binary..."
    if [ ! -f "$OUTPUT_DIR/umb-macos-arm64" ] || [ ! -f "$OUTPUT_DIR/umb-macos-x64" ]; then
        fail "macOS Universal requires both ARM64 and x64 binaries"
        return 1
    fi
    lipo -create \
        "$OUTPUT_DIR/umb-macos-arm64" \
        "$OUTPUT_DIR/umb-macos-x64" \
        -output "$OUTPUT_DIR/umb-macos-universal"
    local size=$(ls -lh "$OUTPUT_DIR/umb-macos-universal" | awk '{print $5}')
    log "macOS Universal done ($size)"
    lipo -info "$OUTPUT_DIR/umb-macos-universal"
}

build_linux_x64() {
    log "Starting Linux x64 Docker build..."
    if ! docker info >/dev/null 2>&1; then
        fail "Docker is not running"
        return 1
    fi
    docker build --platform linux/amd64 -f "$PROJECT_DIR/Dockerfile.linux-build" -t umb-linux-x64-builder "$PROJECT_DIR"
    local container_id=$(docker create --platform linux/amd64 umb-linux-x64-builder)
    docker cp "$container_id:/build/target/release/umb" "$OUTPUT_DIR/umb-linux-x64"
    docker rm "$container_id" >/dev/null
    chmod +x "$OUTPUT_DIR/umb-linux-x64"
    local size=$(ls -lh "$OUTPUT_DIR/umb-linux-x64" | awk '{print $5}')
    log "Linux x64 done ($size)"
}

build_linux_arm64() {
    log "Starting Linux ARM64 Docker build (QEMU emulation - this is slow)..."
    if ! docker buildx version >/dev/null 2>&1; then
        fail "docker buildx required for ARM64 builds"
        return 1
    fi
    docker buildx build --platform linux/arm64 --load -f "$PROJECT_DIR/Dockerfile.linux-arm64" -t umb-linux-arm64-builder "$PROJECT_DIR"
    local container_id=$(docker create --platform linux/arm64 umb-linux-arm64-builder)
    docker cp "$container_id:/build/target/release/umb" "$OUTPUT_DIR/umb-linux-arm64"
    docker rm "$container_id" >/dev/null
    chmod +x "$OUTPUT_DIR/umb-linux-arm64"
    local size=$(ls -lh "$OUTPUT_DIR/umb-linux-arm64" | awk '{print $5}')
    log "Linux ARM64 done ($size)"
}

# Wrapper functions for parallel execution with status tracking
build_and_mark() {
    local name=$1
    local func=$2
    if $func > "$LOG_DIR/${name}.log" 2>&1; then
        mark_success "$name"
    else
        mark_fail "$name"
    fi
}

# =============================================================================
# Install to /usr/local/bin (macOS only)
# =============================================================================

install_local() {
    local binary="$OUTPUT_DIR/umb-macos-arm64"
    if [ ! -f "$binary" ]; then
        binary="$PROJECT_DIR/target/release/umb"
    fi
    if [ ! -f "$binary" ]; then
        binary="$PROJECT_DIR/target/aarch64-apple-darwin/release/umb"
    fi
    if [ ! -f "$binary" ]; then
        fail "No macOS ARM64 binary found to install"
        return 1
    fi

    log "Installing umb to /usr/local/bin..."
    sudo cp "$binary" /usr/local/bin/umb
    sudo xattr -cr /usr/local/bin/umb
    sudo chmod +x /usr/local/bin/umb

    local installed_version=$(/usr/local/bin/umb --version 2>/dev/null || echo "unknown")
    ok "Installed: $installed_version"
}

# =============================================================================
# Deploy to web server
# =============================================================================

deploy_to_server() {
    header "Deploying binaries to web server"
    # Configure your deploy target via environment variables, e.g.:
    #   UMB_DEPLOY_HOST=my-server (an ssh host/alias)
    #   UMB_DEPLOY_DIR=/var/www/downloads
    local server="${UMB_DEPLOY_HOST:-}"
    local dest="${UMB_DEPLOY_DIR:-}"

    if [ -z "$server" ] || [ -z "$dest" ]; then
        fail "--deploy requires UMB_DEPLOY_HOST and UMB_DEPLOY_DIR to be set"
        return 1
    fi

    if ! ssh -q -o ConnectTimeout=5 "$server" exit 2>/dev/null; then
        fail "Cannot connect to $server"
        return 1
    fi

    ssh "$server" "mkdir -p $dest"

    for binary in umb-macos-arm64 umb-macos-x64 umb-macos-universal umb-linux-x64 umb-linux-arm64; do
        if [ -f "$OUTPUT_DIR/$binary" ]; then
            log "Uploading $binary..."
            scp "$OUTPUT_DIR/$binary" "$server:$dest/$binary"
            ssh "$server" "chmod +x $dest/$binary"
        fi
    done

    ok "Deployed to $server:$dest/"
    ssh "$server" "ls -lh $dest/"
}

# =============================================================================
# Parse arguments
# =============================================================================

DO_MACOS=false
DO_LINUX=false
DO_INSTALL=false
DO_DEPLOY=false

if [ $# -eq 0 ]; then
    DO_MACOS=true
    DO_LINUX=true
fi

for arg in "$@"; do
    case $arg in
        --macos)       DO_MACOS=true ;;
        --linux)       DO_LINUX=true ;;
        --all)         DO_MACOS=true; DO_LINUX=true ;;
        --install)     DO_MACOS=true; DO_INSTALL=true ;;
        --deploy)      DO_MACOS=true; DO_LINUX=true; DO_DEPLOY=true ;;
        --help|-h)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all         Build all platforms (default)"
            echo "  --macos       Build macOS only (arm64 + x64 + universal)"
            echo "  --linux       Build Linux only (x64 + arm64 via Docker)"
            echo "  --install     Build macOS ARM64 and install to /usr/local/bin"
            echo "  --deploy      Build all and deploy to web server (set UMB_DEPLOY_HOST + UMB_DEPLOY_DIR)"
            echo "  --help        Show this help"
            echo ""
            echo "Platforms built:"
            echo "  macOS ARM64     - Native cross-compile (fast)"
            echo "  macOS x64       - Native cross-compile (fast)"
            echo "  macOS Universal - lipo combined (depends on above two)"
            echo "  Linux x64       - Docker build (moderate)"
            echo "  Linux ARM64     - Docker buildx + QEMU (slow)"
            echo ""
            echo "All independent builds run in PARALLEL."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --help)"
            exit 1
            ;;
    esac
done

# =============================================================================
# Main execution
# =============================================================================

header "UMB-DEV v$VERSION - Parallel Build"
log "Project: $PROJECT_DIR"
log "Output:  $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT_DIR"

# Clean previous status markers
rm -f "$LOG_DIR"/*.success "$LOG_DIR"/*.fail

# --- Launch parallel builds ---

PIDS=""

if $DO_MACOS; then
    log "Launching macOS ARM64..."
    build_and_mark "macos-arm64" build_macos_arm64 &
    PIDS="$PIDS $!"

    log "Launching macOS x64..."
    build_and_mark "macos-x64" build_macos_x64 &
    PIDS="$PIDS $!"
fi

if $DO_LINUX; then
    log "Launching Linux x64 (Docker)..."
    build_and_mark "linux-x64" build_linux_x64 &
    PIDS="$PIDS $!"

    log "Launching Linux ARM64 (Docker buildx)..."
    build_and_mark "linux-arm64" build_linux_arm64 &
    PIDS="$PIDS $!"
fi

if [ -n "$PIDS" ]; then
    echo ""
    log "Waiting for parallel builds to complete..."
    log "  PIDs: $PIDS"
    echo ""

    # Wait for all background jobs
    for pid in $PIDS; do
        if ! wait $pid; then
            FAILED=true
        fi
    done

    # Check results
    for target in macos-arm64 macos-x64 linux-x64 linux-arm64; do
        if is_success "$target"; then
            ok "$target complete"
        elif is_fail "$target"; then
            fail "$target (see $LOG_DIR/${target}.log)"
            FAILED=true
        fi
    done
fi

# --- macOS Universal (depends on both macOS builds) ---

if $DO_MACOS; then
    if is_success "macos-arm64" && is_success "macos-x64"; then
        if build_macos_universal > "$LOG_DIR/macos-universal.log" 2>&1; then
            mark_success "macos-universal"
            ok "macOS Universal complete"
        else
            mark_fail "macos-universal"
            fail "macOS Universal (see $LOG_DIR/macos-universal.log)"
            FAILED=true
        fi
    else
        log "Skipping macOS Universal (requires both ARM64 and x64)"
    fi
fi

# --- Install if requested ---

if $DO_INSTALL; then
    echo ""
    if is_success "macos-arm64"; then
        install_local
    else
        fail "Cannot install - macOS ARM64 build failed"
    fi
fi

# --- Deploy if requested ---

if $DO_DEPLOY; then
    deploy_to_server
fi

# =============================================================================
# Summary
# =============================================================================

BUILD_END=$(date +%s)
ELAPSED=$((BUILD_END - BUILD_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

header "Build Summary - v$VERSION (${MINUTES}m ${SECONDS}s)"

for target in macos-arm64 macos-x64 macos-universal linux-x64 linux-arm64; do
    binary_name="umb-${target}"

    if is_success "$target"; then
        if [ -f "$OUTPUT_DIR/$binary_name" ]; then
            size=$(ls -lh "$OUTPUT_DIR/$binary_name" | awk '{print $5}')
            echo "  [OK]   $binary_name ($size)"
        else
            echo "  [OK]   $target (no output file)"
        fi
    elif is_fail "$target"; then
        echo "  [FAIL] $target"
    else
        echo "  [SKIP] $target"
    fi
done

echo ""
echo "Output: $OUTPUT_DIR"
echo "Logs:   $LOG_DIR"

if $FAILED; then
    echo ""
    echo "Some builds failed. Check logs for details:"
    for target in macos-arm64 macos-x64 macos-universal linux-x64 linux-arm64; do
        if is_fail "$target"; then
            echo "  cat $LOG_DIR/${target}.log"
        fi
    done
    exit 1
fi
