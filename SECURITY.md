# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 13.x    | yes       |
| < 13.0  | no        |

Only the latest 13.x release receives security fixes. The 13.0.0 release
changed how tools register against the MCP Python SDK; installs pinned to
older SDK majors are not maintained.

## Reporting a vulnerability

Report privately through GitHub, not through a public issue:

**[Report a vulnerability](https://github.com/vbcherepanov/total-agent-memory/security/advisories/new)**

Expect an acknowledgement within 72 hours and an assessment within 7 days.
If a fix is warranted, it ships in the next patch release across all six
distribution channels (PyPI, npm, ghcr, Homebrew, the Claude Code plugin,
and the install script), and the advisory is published once the release is
out.

## Threat model

This server is local-first, and that shapes what counts as a vulnerability.

**In scope**

- Reading or writing memory records outside the caller's own store.
- Path traversal or arbitrary file read through `ingest_codebase`,
  `memory_export`, or any tool that accepts a filesystem path.
- SQL injection into the SQLite store or the FTS5 index.
- Prompt injection through stored memory content that causes a tool to
  execute an action the caller did not request.
- Leaking secrets into memory records, logs, or the enrichment queue —
  the privacy filter in the save hot path is a security control.
- Remote code execution in any transport, including the stateless
  `2026-07-28` HTTP revision.
- Vulnerabilities in the published Docker image or install scripts.

**Out of scope**

- Anything that requires an attacker to already have write access to the
  machine. The database at `~/.claude-memory/memory.db` is a local file
  protected by filesystem permissions and nothing else.
- Denial of service from a caller flooding your own local server.
- Third-party LLM or embedding providers you opt into. API keys live in
  your environment; the default `MEMORY_MODE=fast` hot path makes zero
  network calls.

## Data handling

No telemetry, no analytics, no phone-home. Memory stays in SQLite on the
machine that wrote it. Optional providers (OpenAI, Anthropic, Ollama) are
contacted only when you configure them, and never from the default hot
path.
