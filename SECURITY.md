# Security Policy

Rex is designed from the ground up for local, on-device execution. Because Rex runs inside the user's own process or on self-hosted infrastructure, security is centered around supply-chain integrity, runtime resource containment, safe deserialization, and network isolation.

---

## Supported Versions

Security patches and vulnerability remediation are prioritized for the current active release line and the previous release line.

| Version     |     Supported      | Status                         |
| :---------- | :----------------: | :----------------------------- |
| **0.4.x**   | :white_check_mark: | Current active release line    |
| **0.3.x**   | :white_check_mark: | Critical security patches only |
| **< 0.3.0** |        :x:         | End of life                    |

---

## Reporting a Vulnerability

We appreciate the efforts of security researchers and operators who help keep Rex secure. If you believe you have found a security vulnerability in Rex, `rex-serve`, `rex-ts`, or official Docker containers, **please do not disclose it in a public GitHub issue, pull request, or discussion.**

### Reporting Channels

1. **GitHub Private Vulnerability Reporting (Preferred)**:
   Submit an encrypted disclosure via [GitHub Security Advisories](https://github.com/RequinLabs/rex/security/advisories/new). This enables private, end-to-end coordinated patching with the maintainers before publication.

2. **Direct Maintainer Contact**:
   If private vulnerability reporting is unavailable or you prefer direct contact, contact the RequinLabs security team at:
   `security@requinlabs.com` with the subject line:
   `[SECURITY] Rex Vulnerability Report - <Brief Summary>`

### What to Include

To accelerate triage and verification, please provide:

- **Vulnerability classification**: (e.g., authentication bypass, denial of service, remote memory exhaustion, unhandled exception crash, path traversal).
- **Affected component**: (`rex` Python core, `rex.serve`, `rex-ts`, official Docker image, MCP integration).
- **Affected versions**: Specific commit SHA or release tag (e.g., `v0.4.1`).
- **Reproduction steps**: A minimal script, curl command, or payload demonstrating the vulnerability.
- **Impact assessment**: Practical severity, threat model, prerequisites, and realistic exploit scenarios.
- **Remediation**: (Optional) Suggested patch or mitigation strategy if already identified.

### Coordinated Disclosure Timeline

- **Acknowledgment**: Within 48 hours of initial report.
- **Triage & Reproduction**: Within 5 business days, including confirmation of severity and affected versions.
- **Remediation & Advisory**: We will collaborate with the reporter to develop, verify, and release a fix before public disclosure. Reporters will be credited in the security advisory and release notes unless they request anonymity.

---

## Security Architecture & Operational Hardening

When deploying Rex in production environments (especially when self-hosting `rex-serve` or running on shared hosts), adhere to the following security guidelines:

### 1. Model Weights & Supply Chain Integrity

- **Safe Weight Deserialization**: Rex checkpoints rely on modern, safe serialization formats (`safetensors` and ONNX). The engine avoids unsafe arbitrary object pickling.
- **Revision Pinning**: Pass `revision=PINNED_REVISIONS[...]` when loading weights to guarantee that the runtime only downloads and executes reviewed commit revisions from the Hugging Face Hub, protecting against moving branch attacks.
- **Cryptographic Digest Verification**: Enforce SHA-256 fingerprint verification of every artifact prior to parsing and weight loading.

### 2. HTTP Server Hardening (`rex-serve`)

- **Bearer Token Authentication**:
  - `rex.serve` reads the `REX_API_KEY` (or `LAYA_API_KEY`) environment variable. When set, all requests require `Authorization: Bearer <token>`.
  - Authentication comparisons use constant-time `hmac.compare_digest` across UTF-8 encoded bytes with surrogate-escape handling to prevent both timing attacks and latin-1 decoding exceptions.
  - **Warning**: If API keys are unset, authentication is disabled and the server answers requests without credentials. Always set authentication in non-isolated environments.
- **Network Interface Binding**:
  - By default, `rex-serve` binds to loopback or configured interfaces (`REX_HOST=0.0.0.0` or `127.0.0.1`).
  - For local or loopback-only deployments, explicitly set host to `127.0.0.1` to prevent exposure on external network interfaces.
  - For public or multi-tenant deployments, place `rex-serve` behind a hardened reverse proxy (e.g., NGINX, Envoy, Caddy) that enforces TLS termination, strict rate limiting, connection limits, and request timeouts.
- **Denial of Service & Memory Amplification Protections**:
  - **Option Amplification Guard**: The server rejects questions with excessive options with HTTP 413 before running inference, preventing memory and forward-pass amplification attacks.
  - **Admission Semaphores**: Bounded concurrent requests buffer large request bodies to prevent memory exhaustion from slow-body attacks.

### 3. Container & Host Isolation

- **Non-Root Execution**: Official container images execute under an unprivileged user (`UID 10001`, `GID 10001`). Do not override this with `USER root` in production.
- **File-Backed Secrets**: In containerized deployments, secrets can be mounted from secure volumes, automatically clearing path variables from the environment before launching the process.
- **Native JIT Isolation**: Container sets `TORCH_DISABLE_NATIVE_JIT=1` by default to prevent runtime invocation of compilation backends without a local C toolchain.

### 4. Non-Autoregressive Adversarial Considerations

- **Structured Output Boundary**: Unlike generative autoregressive LLMs, Rex is a non-autoregressive encoder-head classifier that produces discrete, bounded predictions (choices, scores, and binary decisions). Because it does not generate freeform token streams, Rex itself cannot be manipulated into emitting arbitrary adversarial text, executing injected code, or leaking internal instructions through text completion.
- **Defense-in-Depth Telemetry Sanitization**: Rex features built-in PII scrubbing for outgoing tools and telemetry, removing emails, payment card PANs, and authorization tokens before evaluation.

---

## Automated Security Audits in CI/CD

Rex employs automated quality and security gates on every commit and pull request:

- **Secret Leak Detection**: [`gitleaks`](.github/workflows/security.yml) scans git commit history and the checked-out working tree to prevent credentials, API tokens, and secrets from entering the repository.
- **Dependency Vulnerability Scanning**: [`pip-audit`](.github/workflows/security.yml) scans shipped core and `[serve]` dependencies against known vulnerability databases (OSV / PyPI advisories).
- **Static Analysis**: CodeQL runs automated semantic and security static analysis on every push.
- **npm Build Provenance**: The TypeScript SDK (`rex-ts`) is published with cryptographic build provenance and OIDC Trusted Publishing.
