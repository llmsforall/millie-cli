//! Conservative correction of a `str_replace` whose `old_str` was not found.
//!
//! Pure functions, no I/O. Given the file text and the model's `old_str`, find
//! the span the model most plausibly meant, and decide -- with an explicit
//! probability model, not a heuristic threshold -- whether it is safe to
//! treat that span as the intended text.
//!
//! Stages:
//! 1. Linear normalization tiers (line endings / trailing whitespace, collapsed
//!    space runs, indentation). Each tier counts exact occurrences of the
//!    normalized `old_str` in the normalized file: one -> correction, two or
//!    more -> decline (a looser tier cannot disambiguate), zero -> next tier.
//! 2. Fuzzy: character k-gram seed-and-vote over the file to find a few
//!    candidate spans, each verified with a fitting (semi-global) edit-distance
//!    alignment that pins both ends of the matched span exactly.
//! 3. Decision: Bayesian posterior over hypotheses {candidate c is the intended
//!    span} plus H0 {the intended text is not in this file}. Under H_c the
//!    candidate's distance follows a Beta-Binomial (per-character slip rate
//!    integrated out under a weak prior); all other distances, and all
//!    distances under H0, follow the background distribution of unrelated text
//!    estimated from random windows of the same file. Accept only if the best
//!    candidate's posterior clears the threshold and its distance is under an
//!    absolute sanity cap.

use std::collections::HashMap;

/// Tunables. Defaults are the agreed conservative settings.
#[derive(Debug, Clone)]
pub struct CorrectionConfig {
    /// Accept only if P(best candidate | data) >= this.
    pub posterior_threshold: f64,
    /// Prior mass on "the intended text is not in this file".
    pub h0_prior: f64,
    /// Beta prior on the per-character slip rate: mean and strength (a + b).
    pub error_prior_mean: f64,
    pub error_prior_strength: f64,
    /// Never correct if the best distance exceeds this fraction of `old_str`.
    pub max_distance_fraction: f64,
    /// Cost caps: above these the corrector declines.
    pub max_file_bytes: usize,
    pub max_old_str_chars: usize,
    pub max_candidates: usize,
    /// k-gram length for seeding; k-grams occurring more often than
    /// `max_kgram_frequency` in the file are ignored as uninformative.
    pub kgram: usize,
    pub max_kgram_frequency: usize,
    /// Minimum fraction of `old_str`'s k-grams that must vote for a start
    /// position for it to become a candidate.
    pub vote_fraction: f64,
    /// Random windows sampled to estimate the background distance.
    pub background_samples: usize,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            posterior_threshold: 0.9999,
            h0_prior: 0.5,
            error_prior_mean: 0.02,
            error_prior_strength: 50.0,
            max_distance_fraction: 0.25,
            max_file_bytes: 2 * 1024 * 1024,
            max_old_str_chars: 4000,
            max_candidates: 8,
            kgram: 12,
            max_kgram_frequency: 64,
            vote_fraction: 0.25,
            background_samples: 32,
        }
    }
}

/// A proposed correction: the span in the ORIGINAL file text that `old_str`
/// most plausibly referred to.
#[derive(Debug, Clone, PartialEq)]
pub struct Correction {
    /// Byte range in the original file text.
    pub start: usize,
    pub end: usize,
    /// The file text in that range.
    pub matched: String,
    /// Which stage produced it.
    pub tier: &'static str,
    /// Edit distance between `old_str` and the matched text (0 for exact
    /// normalized matches).
    pub distance: usize,
    /// Posterior that this span is the intended one (1.0 for exact normalized
    /// matches, which have no competitor by construction).
    pub posterior: f64,
    /// Distance of the runner-up candidate, when there was one.
    pub runner_up_distance: Option<usize>,
    /// `new_str` with the same indentation delta applied that was observed
    /// between `old_str` and the matched text (indentation tier only).
    pub adjusted_new_str: Option<String>,
}

/// Why the corrector declined.
#[derive(Debug, Clone, PartialEq)]
pub struct Declined {
    pub reason: String,
    pub best_distance: Option<usize>,
    pub runner_up_distance: Option<usize>,
    pub posterior: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Corrected(Correction),
    Declined(Declined),
}

fn declined(reason: impl Into<String>) -> Outcome {
    Outcome::Declined(Declined {
        reason: reason.into(),
        best_distance: None,
        runner_up_distance: None,
        posterior: None,
    })
}

/// Main entry: propose a correction for `old_str` in `file`, or decline.
pub fn suggest(file: &str, old_str: &str, new_str: &str, cfg: &CorrectionConfig) -> Outcome {
    if old_str.is_empty() {
        return declined("old_str is empty");
    }
    if file.len() > cfg.max_file_bytes {
        return declined(format!("file larger than {} bytes", cfg.max_file_bytes));
    }
    let old_chars = old_str.chars().count();
    if old_chars > cfg.max_old_str_chars {
        return declined(format!("old_str longer than {} chars", cfg.max_old_str_chars));
    }
    if file.contains(old_str) {
        return declined("old_str is present verbatim; nothing to correct");
    }

    // ---- Stage 1: normalization tiers -------------------------------------
    for (tier, norm) in [
        ("line-endings-and-trailing-whitespace", Normalization::Endings),
        ("collapsed-whitespace", Normalization::Collapse),
        ("indentation", Normalization::Indent),
    ] {
        let nf = normalize_with_map(file, norm);
        let no = normalize_plain(old_str, norm);
        if no.is_empty() {
            continue;
        }
        let hits = find_all(&nf.text, &no);
        match hits.len() {
            0 => continue,
            1 => {
                let (s_norm, e_norm) = (hits[0], hits[0] + no.len());
                let mut start = nf.orig_start(s_norm);
                let end = nf.orig_end(e_norm);
                if norm == Normalization::Indent {
                    // The indentation tier strips leading whitespace, so the
                    // mapped start lands on the first non-blank character.
                    // Extend the span back over that line's indentation so the
                    // matched text is the lines exactly as they appear in the
                    // file, and new_str (re-indented) replaces whole lines.
                    let bytes = file.as_bytes();
                    while start > 0 && matches!(bytes[start - 1], b' ' | b'\t') {
                        start -= 1;
                    }
                }
                let matched = file[start..end].to_string();
                let adjusted_new_str = if norm == Normalization::Indent {
                    indentation_adjusted_new_str(old_str, &matched, new_str)
                } else {
                    None
                };
                return Outcome::Corrected(Correction {
                    start,
                    end,
                    matched,
                    tier,
                    distance: 0,
                    posterior: 1.0,
                    runner_up_distance: None,
                    adjusted_new_str,
                });
            }
            n => return declined(format!("{n} matches at tier {tier}; ambiguous")),
        }
    }

    // ---- Stage 2: fuzzy candidates ---------------------------------------
    let nf = normalize_with_map(file, Normalization::Endings);
    let pattern: Vec<char> = normalize_plain(old_str, Normalization::Endings).chars().collect();
    let text: Vec<char> = nf.text.chars().collect();
    let m = pattern.len();
    if m < 4 || text.len() < m {
        return declined("old_str too short or longer than the file for fuzzy matching");
    }
    let k = cfg.kgram.min((m / 3).max(4));
    let band = band_for(m, cfg);
    let candidates = seed_and_vote(&text, &pattern, k, band, cfg);
    if candidates.is_empty() {
        return declined("no candidate span shares enough of old_str");
    }

    // Verify each candidate with a fitting alignment over its window.
    let mut verified: Vec<(usize, usize, usize)> = Vec::new(); // (distance, start_char, end_char)
    for (c_lo, c_hi) in candidates.iter().take(cfg.max_candidates) {
        let w_lo = c_lo.saturating_sub(band);
        let w_hi = (c_hi + m + band).min(text.len());
        if w_hi <= w_lo {
            continue;
        }
        let (dist, s, e) = fitting_alignment(&pattern, &text[w_lo..w_hi]);
        verified.push((dist, w_lo + s, w_lo + e));
    }
    if verified.is_empty() {
        return declined("no candidate window could be aligned");
    }
    verified.sort_by_key(|v| v.0);
    // Collapse candidates that aligned to the same span.
    verified.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
    let (best_d, best_s, best_e) = verified[0];
    let runner_up = verified.get(1).map(|v| v.0);

    // ---- Stage 3: decision ------------------------------------------------
    let max_d = ((m as f64) * cfg.max_distance_fraction).floor() as usize;
    if best_d > max_d.max(2) {
        return Outcome::Declined(Declined {
            reason: format!("best candidate needs {best_d} edits, above the cap of {}", max_d.max(2)),
            best_distance: Some(best_d),
            runner_up_distance: runner_up,
            posterior: None,
        });
    }
    let background = background_rate(&text, &pattern, &verified, cfg);
    let posterior = posterior_best(&verified, m, background, cfg);
    if posterior < cfg.posterior_threshold {
        return Outcome::Declined(Declined {
            reason: format!(
                "posterior {posterior:.6} below {}: another span or a missing source is too plausible",
                cfg.posterior_threshold
            ),
            best_distance: Some(best_d),
            runner_up_distance: runner_up,
            posterior: Some(posterior),
        });
    }
    let start = nf.orig_start(char_to_byte(&nf.text, best_s));
    let end = nf.orig_end(char_to_byte(&nf.text, best_e));
    Outcome::Corrected(Correction {
        start,
        end,
        matched: file[start..end].to_string(),
        tier: "fuzzy",
        distance: best_d,
        posterior,
        runner_up_distance: runner_up,
        adjusted_new_str: None,
    })
}

// ---------------------------------------------------------------------------
// Normalization with an offset map back to the original text.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Normalization {
    /// CRLF -> LF, trailing whitespace per line removed, leading/trailing
    /// blank lines removed (pattern only).
    Endings,
    /// Endings + runs of spaces/tabs collapsed to one space.
    Collapse,
    /// Collapse + leading indentation of every line removed.
    Indent,
}

struct Normalized {
    text: String,
    /// For each byte offset in `text` (plus one past the end), the byte offset
    /// in the original it came from.
    map: Vec<usize>,
}

impl Normalized {
    fn orig_start(&self, norm_byte: usize) -> usize {
        self.map[norm_byte.min(self.map.len() - 1)]
    }
    fn orig_end(&self, norm_byte: usize) -> usize {
        self.map[norm_byte.min(self.map.len() - 1)]
    }
}

fn normalize_with_map(input: &str, norm: Normalization) -> Normalized {
    // Work line by line; within a line track original byte offsets.
    let mut text = String::new();
    let mut map: Vec<usize> = Vec::new();
    let mut line_start = 0usize;
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i <= bytes.len() {
        // find end of line (exclusive of newline)
        let mut j = i;
        while j < bytes.len() && bytes[j] != b'\n' {
            j += 1;
        }
        let line = &input[i..j];
        let line_content = line.strip_suffix('\r').unwrap_or(line);
        // trailing whitespace trimmed
        let trimmed_end = line_content.trim_end().len();
        let mut content = &line_content[..trimmed_end];
        let mut content_start = i;
        let lead = content.len() - content.trim_start().len();
        match norm {
            Normalization::Indent => {
                // indentation is dropped entirely
                content = &content[lead..];
                content_start += lead;
            }
            Normalization::Collapse => {
                // indentation is kept verbatim (so indentation drift is only
                // matched by the Indent tier, which re-indents new_str);
                // only interior whitespace runs collapse
                for (b, ch) in content[..lead].char_indices() {
                    text.push(ch);
                    map.push(content_start + b);
                }
                content = &content[lead..];
                content_start += lead;
            }
            Normalization::Endings => {}
        }
        // emit, collapsing whitespace runs for Collapse/Indent; every
        // normalized byte maps to the original byte it came from
        let mut prev_space = false;
        let mut off = content_start;
        for ch in content.chars() {
            let is_space = ch == ' ' || ch == '\t';
            if is_space && norm != Normalization::Endings {
                if !prev_space {
                    text.push(' ');
                    map.push(off);
                }
                prev_space = true;
            } else {
                text.push(ch);
                for b in 0..ch.len_utf8() {
                    map.push(off + b);
                }
                prev_space = false;
            }
            off += ch.len_utf8();
        }
        if j < bytes.len() {
            text.push('\n');
            map.push(j);
        }
        line_start = j + 1;
        i = j + 1;
        if j >= bytes.len() {
            break;
        }
    }
    let _ = line_start;
    map.push(input.len());
    Normalized { text, map }
}

fn normalize_plain(input: &str, norm: Normalization) -> String {
    let mut out = normalize_with_map(input, norm).text;
    // For the pattern, blank lines at both ends are noise.
    let trimmed: Vec<&str> = {
        let lines: Vec<&str> = out.split('\n').collect();
        let mut lo = 0;
        let mut hi = lines.len();
        while lo < hi && lines[lo].trim().is_empty() {
            lo += 1;
        }
        while hi > lo && lines[hi - 1].trim().is_empty() {
            hi -= 1;
        }
        lines[lo..hi].to_vec()
    };
    out = trimmed.join("\n");
    out
}

fn find_all(haystack: &str, needle: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    if needle.is_empty() {
        return hits;
    }
    let mut from = 0;
    while let Some(pos) = haystack[from..].find(needle) {
        hits.push(from + pos);
        from += pos + 1;
        // char-boundary safety for the next search start
        while from < haystack.len() && !haystack.is_char_boundary(from) {
            from += 1;
        }
    }
    hits
}

fn char_to_byte(text: &str, char_idx: usize) -> usize {
    text.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(text.len())
}

/// If every line of the matched text is the corresponding line of `old_str`
/// with the same leading-whitespace delta, apply that delta to `new_str`.
fn indentation_adjusted_new_str(old_str: &str, matched: &str, new_str: &str) -> Option<String> {
    let o: Vec<&str> = old_str.lines().collect();
    let m: Vec<&str> = matched.lines().collect();
    if o.is_empty() || o.len() != m.len() {
        return None;
    }
    let lead = |s: &str| s.len() - s.trim_start().len();
    let mut delta: Option<(bool, String)> = None; // (add?, whitespace)
    for (ol, ml) in o.iter().zip(m.iter()) {
        if ol.trim().is_empty() && ml.trim().is_empty() {
            continue;
        }
        let (lo, lm) = (lead(ol), lead(ml));
        let this = if lm >= lo {
            (true, ml[..lm - lo].to_string())
        } else {
            (false, ol[..lo - lm].to_string())
        };
        match &delta {
            None => delta = Some(this),
            Some(d) if *d == this => {}
            Some(_) => return None,
        }
    }
    let (add, ws) = delta?;
    if ws.is_empty() {
        return None;
    }
    let adjusted: Vec<String> = new_str
        .split('\n')
        .map(|line| {
            if line.trim().is_empty() {
                line.to_string()
            } else if add {
                format!("{ws}{line}")
            } else {
                line.strip_prefix(ws.as_str()).unwrap_or(line).to_string()
            }
        })
        .collect();
    Some(adjusted.join("\n"))
}

// ---------------------------------------------------------------------------
// Fuzzy stage.

fn band_for(m: usize, cfg: &CorrectionConfig) -> usize {
    (((m as f64) * cfg.max_distance_fraction).ceil() as usize).max(2)
}

/// k-gram seed-and-vote. Returns candidate (start_lo, start_hi) clusters of
/// implied pattern start positions (in chars), most-voted first.
fn seed_and_vote(text: &[char], pattern: &[char], k: usize, band: usize, cfg: &CorrectionConfig) -> Vec<(usize, usize)> {
    if pattern.len() < k || text.len() < k {
        return Vec::new();
    }
    // k-gram -> offsets within pattern
    let mut pat_grams: HashMap<&[char], Vec<usize>> = HashMap::new();
    for off in 0..=pattern.len() - k {
        pat_grams.entry(&pattern[off..off + k]).or_default().push(off);
    }
    // frequency of those grams in the file (only grams of interest)
    let mut freq: HashMap<&[char], usize> = HashMap::new();
    for i in 0..=text.len() - k {
        let g = &text[i..i + k];
        if pat_grams.contains_key(g) {
            *freq.entry(g).or_default() += 1;
        }
    }
    // votes per implied start
    let mut votes: HashMap<i64, usize> = HashMap::new();
    let mut informative = 0usize;
    for (g, offs) in &pat_grams {
        let f = freq.get(g).copied().unwrap_or(0);
        if f == 0 || f > cfg.max_kgram_frequency {
            continue;
        }
        informative += offs.len();
    }
    if informative == 0 {
        return Vec::new();
    }
    for i in 0..=text.len() - k {
        let g = &text[i..i + k];
        let Some(offs) = pat_grams.get(g) else { continue };
        let f = freq.get(g).copied().unwrap_or(0);
        if f > cfg.max_kgram_frequency {
            continue;
        }
        for off in offs {
            let start = i as i64 - *off as i64;
            *votes.entry(start).or_default() += 1;
        }
    }
    // cluster implied starts within `band` of each other
    let mut starts: Vec<(i64, usize)> = votes.into_iter().collect();
    starts.sort_by_key(|(s, _)| *s);
    let mut clusters: Vec<(i64, i64, usize)> = Vec::new(); // (lo, hi, votes)
    for (s, v) in starts {
        match clusters.last_mut() {
            Some((lo, hi, total)) if s - *hi <= band as i64 => {
                let _ = lo;
                *hi = s;
                *total += v;
            }
            _ => clusters.push((s, s, v)),
        }
    }
    let needed = ((informative as f64) * cfg.vote_fraction).ceil() as usize;
    let mut out: Vec<(usize, usize, usize)> = clusters
        .into_iter()
        .filter(|(_, _, v)| *v >= needed.max(1))
        .map(|(lo, hi, v)| (lo.max(0) as usize, hi.max(0) as usize, v))
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out.into_iter().map(|(lo, hi, _)| (lo, hi)).collect()
}

/// Fitting (semi-global) alignment of `pattern` inside `window`: the whole
/// pattern must be matched, the match may start and end anywhere in the
/// window. Returns (edit distance, start, end) in window char offsets.
fn fitting_alignment(pattern: &[char], window: &[char]) -> (usize, usize, usize) {
    let m = pattern.len();
    let w = window.len();
    // dp[j] = cost of aligning pattern[..i] ending at window[..j]; origin[j] = start col
    let mut prev: Vec<usize> = vec![0; w + 1];
    let mut prev_origin: Vec<usize> = (0..=w).collect();
    let mut cur: Vec<usize> = vec![0; w + 1];
    let mut cur_origin: Vec<usize> = vec![0; w + 1];
    for i in 1..=m {
        cur[0] = i;
        cur_origin[0] = 0;
        for j in 1..=w {
            let sub = prev[j - 1] + usize::from(pattern[i - 1] != window[j - 1]);
            let del = prev[j] + 1; // pattern char unmatched
            let ins = cur[j - 1] + 1; // extra window char
            let (best, origin) = if sub <= del && sub <= ins {
                (sub, prev_origin[j - 1])
            } else if del <= ins {
                (del, prev_origin[j])
            } else {
                (ins, cur_origin[j - 1])
            };
            cur[j] = best;
            cur_origin[j] = origin;
        }
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut prev_origin, &mut cur_origin);
    }
    let mut best_j = 0;
    let mut best = usize::MAX;
    for j in 0..=w {
        if prev[j] < best {
            best = prev[j];
            best_j = j;
        }
    }
    (best, prev_origin[best_j], best_j)
}

/// Estimate the per-character background mismatch rate q: the distance of
/// `pattern` to unrelated text of the same length in this file. Uses random
/// windows away from the candidates (deterministic sampling) plus the
/// non-best candidates themselves.
fn background_rate(text: &[char], pattern: &[char], verified: &[(usize, usize, usize)], cfg: &CorrectionConfig) -> f64 {
    let m = pattern.len();
    let mut dists: Vec<usize> = Vec::new();
    if text.len() > m + 2 {
        // Keep cost bounded: fewer samples for long patterns.
        let samples = if m > 2500 {
            4
        } else if m > 1000 {
            8
        } else {
            cfg.background_samples
        };
        let mut seed: u64 = (text.len() as u64) * 1_000_003 ^ (m as u64) * 97;
        let span = text.len() - m;
        for _ in 0..samples {
            // xorshift64
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let pos = (seed % (span as u64 + 1)) as usize;
            // skip windows overlapping the best candidate
            if let Some((_, bs, be)) = verified.first()
                && pos < *be
                && pos + m > *bs
            {
                continue;
            }
            let (d, _, _) = fitting_alignment(pattern, &text[pos..pos + m]);
            dists.push(d);
        }
    }
    for v in verified.iter().skip(1) {
        dists.push(v.0);
    }
    if dists.is_empty() {
        // Unrelated text of the same length typically differs in most
        // characters; a fixed conservative value when nothing can be sampled.
        return 0.6;
    }
    let mean = dists.iter().sum::<usize>() as f64 / dists.len() as f64 / (m as f64);
    mean.clamp(0.05, 0.95)
}

/// Posterior probability that the best (first) candidate is the intended span.
fn posterior_best(verified: &[(usize, usize, usize)], m: usize, q: f64, cfg: &CorrectionConfig) -> f64 {
    let n = verified.len();
    let a = cfg.error_prior_mean * cfg.error_prior_strength;
    let b = cfg.error_prior_strength - a;
    let ln_bg = |d: usize| ln_binom_pmf(d, m, q);
    let ln_intended = |d: usize| ln_beta_binom_pmf(d, m, a, b);
    let ln_prior_c = ((1.0 - cfg.h0_prior) / n as f64).ln();
    let ln_prior_0 = cfg.h0_prior.ln();
    let all_bg: f64 = verified.iter().map(|v| ln_bg(v.0)).sum();
    let mut logs: Vec<f64> = Vec::with_capacity(n + 1);
    for (idx, v) in verified.iter().enumerate() {
        let _ = idx;
        logs.push(ln_prior_c + all_bg - ln_bg(v.0) + ln_intended(v.0));
    }
    logs.push(ln_prior_0 + all_bg);
    let max = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let denom: f64 = logs.iter().map(|l| (l - max).exp()).sum();
    ((logs[0] - max).exp() / denom).clamp(0.0, 1.0)
}

fn ln_choose(n: usize, k: usize) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

fn ln_binom_pmf(d: usize, m: usize, q: f64) -> f64 {
    ln_choose(m, d) + (d as f64) * q.ln() + ((m - d) as f64) * (1.0 - q).ln()
}

fn ln_beta(x: f64, y: f64) -> f64 {
    ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)
}

fn ln_beta_binom_pmf(d: usize, m: usize, a: f64, b: f64) -> f64 {
    ln_choose(m, d) + ln_beta(d as f64 + a, (m - d) as f64 + b) - ln_beta(a, b)
}

/// Lanczos approximation of ln(Gamma(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + G + 0.5;
    for (i, c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// A short line-based unified-style diff of `old_str` against the matched
/// text, capped at `max_lines` lines, for the warning shown to the model.
pub fn short_diff(old_str: &str, matched: &str, max_lines: usize) -> String {
    let a: Vec<&str> = old_str.lines().collect();
    let b: Vec<&str> = matched.lines().collect();
    // LCS table (inputs are small)
    let n = a.len();
    let m = b.len();
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let mut out: Vec<String> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            i += 1;
            j += 1;
        } else if i < n && (j >= m || lcs[i + 1][j] >= lcs[i][j + 1]) {
            out.push(format!("-{}", a[i]));
            i += 1;
        } else if j < m {
            out.push(format!("+{}", b[j]));
            j += 1;
        }
    }
    if out.len() > max_lines {
        let shown = out.len();
        out.truncate(max_lines);
        out.push(format!("... ({} more changed lines)", shown - max_lines));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CorrectionConfig {
        CorrectionConfig::default()
    }

    const FILE: &str = "import os\n\ndef add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n\nclass Calc:\n    def mul(self, a, b):\n        return a * b\n";

    #[test]
    fn verbatim_present_is_not_corrected() {
        assert!(matches!(suggest(FILE, "def add(a, b):", "x", &cfg()), Outcome::Declined(_)));
    }

    #[test]
    fn trailing_whitespace_and_crlf_are_forgiven() {
        let out = suggest(FILE, "def add(a, b):   \r\n    return a + b  ", "def add(a, b):\n    return a + b + 0", &cfg());
        let Outcome::Corrected(c) = out else { panic!("expected correction, got {out:?}") };
        assert_eq!(c.tier, "line-endings-and-trailing-whitespace");
        assert_eq!(c.matched, "def add(a, b):\n    return a + b");
        assert_eq!(c.distance, 0);
    }

    #[test]
    fn indentation_drift_is_corrected_and_new_str_re_indented() {
        let out = suggest(FILE, "def mul(self, a, b):\n    return a * b", "def mul(self, a, b):\n    return a * b * 1", &cfg());
        let Outcome::Corrected(c) = out else { panic!("expected correction, got {out:?}") };
        assert_eq!(c.tier, "indentation");
        assert_eq!(c.matched, "    def mul(self, a, b):\n        return a * b");
        assert_eq!(c.adjusted_new_str.as_deref(), Some("    def mul(self, a, b):\n        return a * b * 1"));
    }

    #[test]
    fn ambiguity_at_a_tier_declines() {
        // "return a" (collapsed) matches twice -> decline, never guess.
        let out = suggest(FILE, "return  a", "x", &cfg());
        assert!(matches!(out, Outcome::Declined(_)), "{out:?}");
    }

    #[test]
    fn single_typo_in_unique_region_is_corrected_fuzzily() {
        let out = suggest(FILE, "class Calc:\n    def mull(self, a, b):\n        return a * b", "X", &cfg());
        let Outcome::Corrected(c) = out else { panic!("expected correction, got {out:?}") };
        assert_eq!(c.tier, "fuzzy");
        assert_eq!(c.matched, "class Calc:\n    def mul(self, a, b):\n        return a * b");
        assert_eq!(c.distance, 1);
        assert!(c.posterior >= cfg().posterior_threshold);
    }

    #[test]
    fn near_duplicate_regions_decline() {
        // Two near-identical functions; a pattern that is one edit from both.
        let file = "def f1(x):\n    return x + 1\n\ndef f2(x):\n    return x + 2\n";
        let out = suggest(file, "def f9(x):\n    return x + 3", "X", &cfg());
        assert!(matches!(out, Outcome::Declined(_)), "{out:?}");
    }

    #[test]
    fn hallucinated_text_declines() {
        let out = suggest(FILE, "def divide(self, numerator, denominator):\n        return numerator / denominator", "X", &cfg());
        assert!(matches!(out, Outcome::Declined(_)), "{out:?}");
    }

    #[test]
    fn fitting_alignment_pins_both_ends() {
        let pattern: Vec<char> = "hello world".chars().collect();
        let window: Vec<char> = "xx hello wrld yy".chars().collect();
        let (d, s, e) = fitting_alignment(&pattern, &window);
        assert_eq!(d, 1);
        assert_eq!(window[s..e].iter().collect::<String>(), "hello wrld");
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        assert!((ln_gamma(5.0) - (24.0f64).ln()).abs() < 1e-9);
        assert!((ln_gamma(1.0)).abs() < 1e-9);
    }

    #[test]
    fn short_diff_marks_changed_lines_only() {
        let d = short_diff("a\nb\nc", "a\nB\nc", 12);
        assert_eq!(d, "-b\n+B");
    }
}
