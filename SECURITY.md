# Security Policy

cc-lite is a local research tool.
It has no server, account, or network service, but it does load files and can launch an external engine you name.

## Supported Versions

Security fixes target the latest release.

## Reporting A Vulnerability

Report suspected vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/hskl18/cc-lite/security/advisories/new).
Do not open a public issue that contains credentials, private game records, or a malicious file.

## Security Model

- Checkpoints are loaded with `torch.load(..., weights_only=True)`, which refuses pickled code objects.
  Treat a checkpoint from an unknown source as untrusted input anyway, and check its SHA-256 against the evidence manifest before use.
- Engine commands such as `--engine ./pikafish` are started without a shell, but they still run with your user's permissions.
  Only point cc-lite at engine binaries you trust.
- Game-record ingestion validates every move and requires a provenance file, but it does not check that a dataset's license is genuine.
- Engine binaries, downloaded datasets, and generated checkpoints stay out of Git through `.gitignore`.
  The one committed checkpoint is the 6.8 MB debug artifact under `evidence/debug-v1/checkpoint/`.
