#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Building api_server for aarch64 using cross..."
# Mount the burn workspace so path dependencies resolve
export CROSS_CONTAINER_OPTS="--volume /Users/perro/work/burn:/Users/perro/work/burn:ro"
cross build --release --bin api_server --target aarch64-unknown-linux-gnu -p burn_inference

echo "Building Docker image..."
docker build -f Dockerfile.api -t mandelbulb/michi-api:latest .

echo "Pushing to Docker Hub..."
docker push mandelbulb/michi-api:latest

echo "Done!"
