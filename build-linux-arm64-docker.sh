#!/bin/bash
# Build Linux ARM64 binary using Docker with Ubuntu 24.04 (glibc 2.39)
# This builds natively for ARM64 using Docker's multi-platform support
# Requires: docker buildx (included in Docker Desktop)

set -e

echo "Building umb for Linux ARM64 using Docker..."

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "Error: docker buildx is required for ARM64 builds"
    echo "Install Docker Desktop or run: docker buildx install"
    exit 1
fi

# Build Docker image and compile binary (force linux/arm64 platform)
# Note: This uses QEMU emulation on x64 hosts, so it will be slow
docker buildx build --platform linux/arm64 --load -f Dockerfile.linux-arm64 -t umb-linux-arm64-builder .

# Create target directory if it doesn't exist
mkdir -p ./target

# Extract the binary from the container
container_id=$(docker create --platform linux/arm64 umb-linux-arm64-builder)
docker cp "$container_id:/build/target/release/umb" ./target/umb-linux-arm64
docker rm "$container_id"

# Verify the binary
file ./target/umb-linux-arm64
echo ""
echo "Checking library dependencies:"
docker run --rm --platform linux/arm64 -v "$(pwd)/target:/target" ubuntu:24.04 ldd /target/umb-linux-arm64 2>&1 | head -20 || true

echo ""
echo "✓ Linux ARM64 binary built successfully: ./target/umb-linux-arm64"
echo "  Size: $(du -h ./target/umb-linux-arm64 | cut -f1)"
