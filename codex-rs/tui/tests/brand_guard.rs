//! Brand guard: millie must never present itself as Codex/OpenAI/GPT. This
//! regressed after every upstream sync when fixes edited individual strings;
//! this test makes any reintroduction fail CI so the merger rebrands at
//! merge time instead of shipping it.
//!
//! Scope: double-quoted string literals in non-test sources of the tui and
//! cli crates.
//!
//! Rules:
//! * "Codex" is FORBIDDEN everywhere — it is OpenAI's product name and has
//!   no place in this software's text. (Identifier-shaped usages like the
//!   CODEX_* env vars and the codex-resources binary-layout contract are
//!   internal API surface, not product text, and are excluded.)
//! * "OpenAI"/"ChatGPT"/GPT model names are forbidden except in modules
//!   whose subject matter IS that third-party service (auth, app links,
//!   doctor's auth diagnostics): naming someone else's service is fine;
//!   presenting OURS as theirs is not.

use std::path::Path;

const FORBIDDEN_EVERYWHERE: &[&str] = &["Codex", " codex ", "codex login", "codex home"];
const FORBIDDEN_AS_PRODUCT: &[&str] = &["OpenAI", "ChatGPT", "GPT-5", "gpt-5", "openai.com"];

// Modules allowed to mention the third-party SERVICE by name (never "Codex").
const SERVICE_MODULES: &[&str] = &[
    "local_chatgpt_auth.rs",
    "app_link_view.rs",
    "doctor.rs",
    "doctor/",
    "onboarding/auth",
    "chatwidget/plugins.rs",
    "login",
];

fn literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            let mut s = String::new();
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' && j + 1 < bytes.len() {
                    j += 2;
                    s.push('\\');
                    continue;
                }
                s.push(bytes[j] as char);
                j += 1;
            }
            out.push(s);
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

// Attribution to the upstream project is allowed (and for the license,
// required): a literal that explicitly credits the parent software is not
// product branding.
const ATTRIBUTION_MARKERS: &[&str] = &[
    "fork of",
    "based on",
    "derived from",
    "Apache",
    "License",
    "NOTICE",
    "upstream",
];

fn is_attribution(lit: &str) -> bool {
    ATTRIBUTION_MARKERS.iter().any(|m| lit.contains(m))
}

// Exact literals that are functional, not branding: security allowlists of
// third-party hosts and the error text that truthfully describes them.
const ALLOWED_LITERALS: &[&str] = &[
    "openai.com",
    "openai.org",
    "remote exec-server API-key authentication is restricted to HTTPS openai.com and openai.org hosts and subdomains or loopback hosts",
];

// Runtime env vars were renamed MILLIE_* (2026-08) so millie and a real
// codex install on the same machine cannot read each other's state or
// sandbox markers. Only compile-time vars keep the old prefix.
const ALLOWED_CODEX_ENV_VARS: &[&str] = &[
    "CODEX_BUILD_COMMIT",
    "CODEX_BWRAP_SHA256",
    "CODEX_REPO_ROOT_MARKER",
];

fn names_forbidden_env_var(lit: &str) -> Option<&str> {
    let start = lit.find("CODEX_")?;
    // `{CODEX_...}` is a format-string capture of a Rust const, not an env var
    if start > 0 && lit.as_bytes()[start - 1] == b'{' {
        return None;
    }
    let name: String = lit[start..]
        .chars()
        .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
        .collect();
    if ALLOWED_CODEX_ENV_VARS.contains(&name.as_str()) {
        None
    } else {
        Some("CODEX_ env var (rename to MILLIE_)")
    }
}

fn strip_identifiers(lit: &str) -> String {
    lit.replace("CODEX_", "")
        .replace("codex-resources", "")
        .replace("codex_", "")
        .replace("codex-", "")
        .replace(".codex", "")
        .replace("OpenAI-Beta", "")
        .replace("@openai.com", "")
}

fn scan(dir: &Path, violations: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let name = path.to_string_lossy().to_string();
        if path.is_dir() {
            scan(&path, violations);
            continue;
        }
        if !name.ends_with(".rs") || name.contains("test") {
            continue;
        }
        let in_service_module = SERVICE_MODULES.iter().any(|a| name.contains(a));
        let source = std::fs::read_to_string(&path).expect("read source");
        for (ln, line) in source.lines().enumerate() {
            // production strings only: stop at the first in-file test module
            if line.contains("#[cfg(test)]") {
                break;
            }
            for lit in literals(line) {
                if is_attribution(&lit) || ALLOWED_LITERALS.contains(&lit.as_str()) {
                    continue;
                }
                if let Some(reason) = names_forbidden_env_var(&lit) {
                    violations.push(format!("{name}:{}: {lit:?} contains {reason}", ln + 1));
                }
                let cleaned = strip_identifiers(&lit);
                for word in FORBIDDEN_EVERYWHERE {
                    if cleaned.contains(word) {
                        violations.push(format!("{name}:{}: {lit:?} contains {word}", ln + 1));
                    }
                }
                if !in_service_module {
                    for word in FORBIDDEN_AS_PRODUCT {
                        if cleaned.contains(word) {
                            violations.push(format!("{name}:{}: {lit:?} contains {word}", ln + 1));
                        }
                    }
                }
            }
        }
    }
}

fn scan_env_vars_only(dir: &Path, violations: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let name = path.to_string_lossy().to_string();
        if path.is_dir() {
            if !name.ends_with("/target") && !name.ends_with("/.git") {
                scan_env_vars_only(&path, violations);
            }
            continue;
        }
        if !name.ends_with(".rs") || name.ends_with("tests/brand_guard.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read source");
        for (ln, line) in source.lines().enumerate() {
            for lit in literals(line) {
                if let Some(reason) = names_forbidden_env_var(&lit) {
                    violations.push(format!("{name}:{}: {lit:?} contains {reason}", ln + 1));
                }
            }
        }
    }
}

// Env vars are a functional boundary, not just branding: millie sets sandbox
// markers and server parameters into child process environments, and reads
// its own back. Sharing the CODEX_ prefix with a real codex install on the
// same machine lets the two products obey each other's state. This scans the
// entire workspace, tests included, because tests setting the old names would
// silently pass against readers that no longer look at them.
#[test]
fn env_vars_use_the_millie_prefix_workspace_wide() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().expect("workspace root");
    let mut violations = Vec::new();
    scan_env_vars_only(workspace_root, &mut violations);
    assert!(
        violations.is_empty(),
        "CODEX_-prefixed env vars can clash with a codex install on the same machine (rename to MILLIE_, or allowlist a compile-time-only var):\n{}",
        violations.join("\n")
    );
}

#[test]
fn user_facing_strings_never_brand_the_product_as_openai_codex() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    scan(&manifest.join("src"), &mut violations);
    scan(&manifest.join("../cli/src"), &mut violations);
    scan(&manifest.join("../exec/src"), &mut violations);
    assert!(
        violations.is_empty(),
        "forbidden branding in user-facing strings (rebrand to Millie; only genuinely third-party-service modules may name OpenAI/ChatGPT):\n{}",
        violations.join("\n")
    );
}
