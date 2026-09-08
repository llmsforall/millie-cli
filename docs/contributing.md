## Contributing

Bug reports, reproductions and pull requests are welcome.

- For a bug, open an issue with the Millie version (`millie --version`), your
  platform, the model in use, and steps to reproduce.
- For a feature or a change in behaviour, open an issue first so the approach
  can be agreed before code is written.
- Keep pull requests focused: one fix or feature per PR, with tests where the
  change is testable, and a short description of what changed and why.

### Development workflow

- Create a topic branch from `main`.
- Build and test from `codex-rs/` (the Rust workspace); see
  [install.md](./install.md) for the toolchain and commands.
- Run `just fmt` and `just fix -p <crate>` for the crates you touched, and the
  relevant tests, before opening the PR.
- Make sure your branch is up to date with `main` and CI is green.
