use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-ObjC");
    }

    // The version string shown by `--version`. Release builds stamp an exact
    // version via MILLIE_VERSION; everything else self-identifies as a dev
    // build of the workspace version plus the commit it was built from, so
    // two pre-release binaries are never indistinguishable.
    println!("cargo:rerun-if-env-changed=MILLIE_VERSION");
    let version = match std::env::var("MILLIE_VERSION") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            let base = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
            match git(&["rev-parse", "--short=9", "HEAD"]) {
                Some(hash) => {
                    let dirty = git(&["status", "--porcelain"])
                        .map(|_| ".dirty")
                        .unwrap_or("");
                    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
                        // HEAD only changes on branch switches; commits move the
                        // branch ref and the index, so watch those too or the
                        // embedded hash goes stale across commits.
                        println!("cargo:rerun-if-changed={dir}/HEAD");
                        println!("cargo:rerun-if-changed={dir}/index");
                        if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
                            println!("cargo:rerun-if-changed={dir}/{r}");
                        }
                    }
                    format!("{base}-dev+g{hash}{dirty}")
                }
                None => format!("{base}-dev"),
            }
        }
    };
    println!("cargo:rustc-env=MILLIE_VERSION_STRING={version}");
}
