#!/usr/bin/env bash
set -euo pipefail

# build_image.sh - Build the GPU PeopleCounter image
# Usage: ./build_image.sh

IMAGE_NAME="people-counter:gpu-final"
BACKUP_TAG="people-counter:gpu-backup-$(date +%Y%m%d)"

echo "🏗️ Starting Docker build for $IMAGE_NAME..."
DOCKER_BUILDKIT=1 docker build -t "$IMAGE_NAME" .

echo "💾 Creating backup tag: $BACKUP_TAG"
docker tag "$IMAGE_NAME" "$BACKUP_TAG"

echo "✅ Build finished successfully!"
docker images | grep people-counter
