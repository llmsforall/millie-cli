# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them privately
through GitHub's "Report a vulnerability" button on the
[Security tab](https://github.com/llmsforall/millie-cli/security) of this
repository. We will acknowledge reports as quickly as we can and keep you
informed while we work on a fix.

## How Millie limits what it can do

Millie runs commands and edits files on your behalf, so its safety boundaries
matter. Commands run inside a sandbox, file writes are limited to the
workspace unless you approve otherwise, and network access from the sandbox is
off by default. See [docs/sandbox.md](./docs/sandbox.md) for the details and
the configuration that controls them.
