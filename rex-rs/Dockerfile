# ==============================================================================
# Multi-stage Dockerfile for Rex On-Premise Encrypted Model Runtime
# ==============================================================================

# Stage 1: Build binary
FROM rust:1-bookworm AS builder

WORKDIR /usr/src/rex-rs

# Cache dependencies
COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/pack.rs && \
    echo "pub mod dummy {}" > src/lib.rs && \
    cargo build --release || true

# Copy full source and compile release binaries
COPY src ./src
COPY tests ./tests
RUN cargo test --release && \
    cargo build --release

# Stage 2: Minimal hardened runtime image
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user for secure execution
RUN groupadd -g 10001 rex && \
    useradd -u 10001 -g rex -s /bin/false -M rex

WORKDIR /app

# Copy compiled binaries from builder
COPY --from=builder /usr/src/rex-rs/target/release/rex-runtime /usr/local/bin/rex-runtime
COPY --from=builder /usr/src/rex-rs/target/release/rex-pack /usr/local/bin/rex-pack

# Directory to mount encrypted model packages
RUN mkdir -p /models && chown -R rex:rex /app /models

USER rex:rex

ENV REX_HOST=0.0.0.0
ENV REX_PORT=8080
ENV REX_MODEL_PATH=/models/model.rexenc

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/rex-runtime"]
