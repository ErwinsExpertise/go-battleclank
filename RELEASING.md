# Release Process

This document describes how to create a new release of go-battleclank.

## Automated Releases with GoReleaser

The project uses [GoReleaser](https://goreleaser.com/) and GitHub Actions to **fully automate** the release process:

### What Happens Automatically

When you push commits to `main` with conventional commit messages:

1. **Auto-tagging** - GitHub Actions automatically creates a version tag based on commit messages
2. **Version calculation** - Semantic version is incremented based on commit types:
   - `feat:` → Minor version bump (v1.0.0 → v1.1.0)
   - `fix:` or `perf:` → Patch version bump (v1.0.0 → v1.0.1)
   - `feat!:` or `BREAKING CHANGE:` → Major version bump (v1.0.0 → v2.0.0)
3. **Release creation** - GoReleaser builds and publishes:
   - Binaries for multiple platforms (Linux, macOS, Windows)
   - Docker images for amd64 and arm64
   - Generated changelog from commits
   - GitHub release with all artifacts
   - Docker images to GitHub Container Registry

### Simple Release Process (Recommended)

1. **Merge your PR to main with conventional commits**
   ```bash
   git commit -m "feat: add A* pathfinding algorithm"
   git push origin main
   ```

2. **Wait for automation**
   - Auto-tag workflow creates the version tag
   - Release workflow builds and publishes everything
   - Check: https://github.com/ErwinsExpertise/go-battleclank/actions

3. **Done!** 🎉
   - Visit: https://github.com/ErwinsExpertise/go-battleclank/releases
   - Your release is live with all artifacts

## Manual Release (Alternative)

If you prefer to create tags manually:

### Prerequisites

- Push access to the main branch
- Ability to create tags in the repository

### Steps

1. **Ensure all changes are merged to main**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run tests locally**
   ```bash
   go test -v ./...
   ```

3. **Create and push a version tag manually**
   
   For a new feature (minor version):
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0 - Add A* pathfinding"
   git push origin v1.1.0
   ```

   For a bug fix (patch version):
   ```bash
   git tag -a v1.0.1 -m "Release v1.0.1 - Bug fixes"
   git push origin v1.0.1
   ```

4. **Monitor the release workflow**
   - Go to: https://github.com/ErwinsExpertise/go-battleclank/actions
   - Watch the "Release" workflow complete
   - Check for any errors

5. **Verify the release**
   - Visit: https://github.com/ErwinsExpertise/go-battleclank/releases
   - Verify all artifacts are present
   - Test downloading and running a binary

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (v2.0.0) - Incompatible API changes
- **MINOR** version (v1.1.0) - New functionality in a backward compatible manner
- **PATCH** version (v1.0.1) - Backward compatible bug fixes

### Examples

- `v1.0.0` - Initial release
- `v1.1.0` - Added A* pathfinding algorithm
- `v1.1.1` - Fixed bug in food seeking
- `v2.0.0` - Breaking change to configuration format

## What Gets Released

Each release includes:

### Binaries
- **Linux**: amd64, arm64, arm (v7)
- **macOS**: amd64, arm64 (Apple Silicon)
- **Windows**: amd64

### Docker Images
- `ghcr.io/erwinsexpertise/go-battleclank:latest`
- `ghcr.io/erwinsexpertise/go-battleclank:v1.1.0`
- Multi-architecture support (amd64, arm64)

### Documentation
- README.md
- ALGORITHMS.md
- STRATEGY_REVIEW.md
- ASTAR_IMPLEMENTATION.md
- USAGE.md
- LICENSE

### Checksums
- SHA256 checksums for all artifacts

## Release Artifacts Structure

```
dist/
├── go-battleclank_1.1.0_Darwin_arm64.tar.gz
├── go-battleclank_1.1.0_Darwin_x86_64.tar.gz
├── go-battleclank_1.1.0_Linux_arm64.tar.gz
├── go-battleclank_1.1.0_Linux_armv7.tar.gz
├── go-battleclank_1.1.0_Linux_x86_64.tar.gz
├── go-battleclank_1.1.0_Windows_x86_64.zip
└── checksums.txt
```

## Testing Locally

To test the release process without publishing:

```bash
# Install GoReleaser
go install github.com/goreleaser/goreleaser@latest

# Build snapshot (doesn't require a tag)
goreleaser build --snapshot --clean

# Check configuration
goreleaser check

# Full release dry run
goreleaser release --snapshot --clean
```

## Docker Images

### Pulling Images

```bash
# Latest version
docker pull ghcr.io/erwinsexpertise/go-battleclank:latest

# Specific version
docker pull ghcr.io/erwinsexpertise/go-battleclank:v1.1.0
```

### Running with Docker

```bash
docker run -p 8000:8000 ghcr.io/erwinsexpertise/go-battleclank:latest
```

## Rollback

If a release has issues:

1. **Delete the tag locally and remotely**
   ```bash
   git tag -d v1.1.0
   git push origin :refs/tags/v1.1.0
   ```

2. **Delete the GitHub release**
   - Go to releases page
   - Click the problematic release
   - Click "Delete"

3. **Fix the issue and create a new release**
   - Increment to next patch version (e.g., v1.1.1)

## Conventional Commits

The project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning and changelog generation.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types and Version Bumps

| Commit Type | Version Bump | Example |
|-------------|--------------|---------|
| `feat:` | **Minor** (v1.0.0 → v1.1.0) | `feat: add A* pathfinding algorithm` |
| `fix:` | **Patch** (v1.0.0 → v1.0.1) | `fix: correct food seeking logic` |
| `perf:` | **Patch** (v1.0.0 → v1.0.1) | `perf: optimize flood fill algorithm` |
| `feat!:` or `BREAKING CHANGE:` | **Major** (v1.0.0 → v2.0.0) | `feat!: redesign configuration API` |
| `docs:` | No bump | `docs: update strategy review` |
| `test:` | No bump | `test: add unit tests for A*` |
| `chore:` | No bump | `chore: update dependencies` |

### Examples

**Feature (Minor Bump)**
```bash
git commit -m "feat: add A* pathfinding algorithm

Implements A* search for more accurate food seeking.
Uses priority queue and heuristic distance calculation."
```

**Bug Fix (Patch Bump)**
```bash
git commit -m "fix: prevent chase of unreachable food

The snake was getting stuck trying to reach food blocked by walls.
Now uses A* to verify path exists before committing."
```

**Breaking Change (Major Bump)**
```bash
git commit -m "feat!: redesign strategy configuration

BREAKING CHANGE: Configuration file format has changed.
See MIGRATION.md for upgrade instructions."
```

**Performance (Patch Bump)**
```bash
git commit -m "perf: optimize flood fill with caching"
```

**Documentation (No Bump)**
```bash
git commit -m "docs: add release process documentation"
```

### Changelog

The changelog is automatically generated from these commit messages:

- `feat:` → **New Features** section
- `fix:` → **Bug Fixes** section  
- `perf:` → **Performance Improvements** section
- `docs:` → **Documentation** section

## Troubleshooting

### Release workflow fails

1. Check the GitHub Actions logs
2. Common issues:
   - Missing GITHUB_TOKEN permissions
   - GoReleaser configuration errors
   - Build failures on specific platforms

### Docker images not published

1. Verify GitHub Container Registry permissions
2. Check Docker Buildx setup in workflow
3. Ensure GITHUB_TOKEN has `packages: write` permission

### Binaries don't work

1. Test locally with `goreleaser build --snapshot`
2. Check ldflags configuration
3. Verify CGO_ENABLED=0 for static binaries

## References

- [GoReleaser Documentation](https://goreleaser.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
