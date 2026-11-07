# Build:  docker build -t kytypipes-app .
FROM rust:1-bookworm AS build
WORKDIR /src
# Dependencies first, so source changes do not rebuild them.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && touch src/lib.rs \
    && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs src/lib.rs && cargo build --release --locked \
    && mkdir -p /out/data && strip target/release/pipeline-share-server

# Minimal image: glibc + libgcc only, no shell, non-root user (uid 65532).
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/pipeline-share-server /usr/local/bin/pipeline-share-server
COPY --from=build --chown=65532:65532 /out/data /data
VOLUME /data
EXPOSE 8787
USER nonroot
ENTRYPOINT ["/usr/local/bin/pipeline-share-server"]
CMD ["--listen", "0.0.0.0:8787", "--data-dir", "/data"]
