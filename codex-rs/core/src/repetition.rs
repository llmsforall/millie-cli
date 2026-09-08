//! Client-side repetition stop: the same single-gram rule as the llama.cpp
//! fork's `repeat_stop` (tools/server/server-repetition.h), applied to the
//! stream of generated units when the backend cannot do it itself (vLLM).
//! Units are the streamed deltas, hashed; on vLLM each delta is one token.

use serde_json::Value;

#[derive(Debug, Clone)]
pub(crate) struct RepeatStopParams {
    pub windows: Vec<usize>,
    pub min_tokens: usize,
    pub min_repeats: usize,
    pub coverage: f32,
    pub stride: usize,
}

impl Default for RepeatStopParams {
    fn default() -> Self {
        Self { windows: vec![300, 600, 900], min_tokens: 300, min_repeats: 3, coverage: 0.8, stride: 4 }
    }
}

impl RepeatStopParams {
    pub(crate) fn from_json(v: &Value) -> Self {
        let d = Self::default();
        let usize_of = |k: &str, def: usize| v.get(k).and_then(Value::as_u64).map(|x| x as usize).unwrap_or(def);
        let windows = v
            .get("windows")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect::<Vec<_>>())
            .filter(|w| !w.is_empty())
            .unwrap_or(d.windows.clone());
        Self {
            windows,
            min_tokens: usize_of("min_tokens", d.min_tokens),
            min_repeats: usize_of("min_repeats", d.min_repeats).max(1),
            coverage: v.get("coverage").and_then(Value::as_f64).map(|x| x as f32).unwrap_or(d.coverage),
            stride: usize_of("stride", d.stride).max(1),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RepetitionStats {
    pub flagged: bool,
    pub window: usize,
    pub gram_len: usize,
    pub count: usize,
    pub coverage: f32,
    /// index into the unit sequence of one occurrence
    pub gram_pos: usize,
}

fn gram_lengths(max_len: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (3..=max_len.min(16)).collect();
    let mut l = 16.0f64;
    loop {
        l *= 1.12;
        let li = l as usize;
        if li > max_len {
            break;
        }
        if out.last().is_none_or(|&last| li > last) {
            out.push(li);
        }
    }
    out
}

/// Polynomial rolling hash of every L-gram of `tail`.
fn gram_hashes(tail: &[u64], l: usize) -> Vec<u64> {
    const B: u64 = 1_000_003;
    let n = tail.len();
    if l == 0 || l > n {
        return Vec::new();
    }
    let mut pow = 1u64;
    for _ in 0..l {
        pow = pow.wrapping_mul(B);
    }
    let mut h = 0u64;
    for &t in &tail[..l] {
        h = h.wrapping_mul(B).wrapping_add(t ^ 0x9E37_79B9_7F4A_7C15);
    }
    let mut out = Vec::with_capacity(n - l + 1);
    out.push(h);
    for i in l..n {
        h = h
            .wrapping_mul(B)
            .wrapping_add(tail[i] ^ 0x9E37_79B9_7F4A_7C15)
            .wrapping_sub(pow.wrapping_mul(tail[i - l] ^ 0x9E37_79B9_7F4A_7C15));
        out.push(h);
    }
    out
}

fn detect_window(toks: &[u64], w: usize, p: &RepeatStopParams) -> RepetitionStats {
    let mut best = RepetitionStats { window: w, ..Default::default() };
    if toks.len() < w || w == 0 {
        return best;
    }
    let start = toks.len() - w;
    let tail = &toks[start..];
    let max_len = w / p.min_repeats.max(1);
    for l in gram_lengths(max_len) {
        let hashes = gram_hashes(tail, l);
        let mut pos: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::with_capacity(w);
        for (i, h) in hashes.iter().enumerate() {
            pos.entry(*h).or_default().push(i);
        }
        let Some(dom) = pos.values().max_by_key(|v| v.len()) else { continue };
        if dom.len() < p.min_repeats {
            continue;
        }
        let first = dom[0];
        let mut count = 0usize;
        let mut last_end: isize = -1;
        for &i in dom {
            if (i as isize) < last_end || tail[i..i + l] != tail[first..first + l] {
                continue;
            }
            count += 1;
            last_end = (i + l) as isize;
        }
        let cov = (count * l) as f32 / w as f32;
        if count >= p.min_repeats && cov > best.coverage {
            best.gram_len = l;
            best.count = count;
            best.coverage = cov;
            best.gram_pos = start + first;
        }
    }
    best.flagged = best.count >= p.min_repeats && best.coverage >= p.coverage;
    best
}

/// Evaluate all configured windows; returns the flagged one if any, else the
/// window with the highest coverage.
pub(crate) fn detect(toks: &[u64], p: &RepeatStopParams) -> RepetitionStats {
    let mut best = RepetitionStats::default();
    if toks.len() < p.min_tokens {
        return best;
    }
    for &w in &p.windows {
        let w_eff = w.min(toks.len());
        if w_eff < p.min_tokens {
            continue;
        }
        let st = detect_window(toks, w_eff, p);
        if st.flagged {
            return st;
        }
        if st.coverage > best.coverage {
            best = st;
        }
    }
    best
}

pub(crate) fn hash_unit(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(from: u64, n: usize) -> Vec<u64> {
        (0..n as u64).map(|i| from + i).collect()
    }
    fn repeat_block(block: &[u64], times: usize) -> Vec<u64> {
        let mut v = Vec::new();
        for _ in 0..times {
            v.extend_from_slice(block);
        }
        v
    }

    #[test]
    fn verbatim_loop_is_flagged() {
        let p = RepeatStopParams::default();
        let st = detect(&repeat_block(&seq(1000, 40), 10), &p);
        assert!(st.flagged);
        assert!(st.count >= 3 && st.coverage >= 0.8);
    }

    #[test]
    fn alternating_names_list_is_not_flagged() {
        let p = RepeatStopParams::default();
        let mut toks = Vec::new();
        let (a, b) = (seq(5000, 8), seq(6000, 8));
        for i in 1..=100u64 {
            for c in i.to_string().bytes() {
                toks.push(100 + (c - b'0') as u64);
            }
            toks.push(7);
            toks.extend_from_slice(if i % 2 == 1 { &a } else { &b });
            toks.push(8);
        }
        let st = detect(&toks, &p);
        assert!(!st.flagged, "{st:?}");
    }

    #[test]
    fn random_order_names_are_not_flagged() {
        let p = RepeatStopParams::default();
        let mut toks = Vec::new();
        let mut rng: u32 = 12345;
        for _ in 0..120 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            toks.extend(seq(9000 + ((rng >> 16) % 20) as u64 * 10, 5));
            toks.push(8);
        }
        assert!(!detect(&toks, &p).flagged);
    }

    #[test]
    fn below_min_tokens_nothing_is_judged() {
        let p = RepeatStopParams::default();
        assert!(!detect(&repeat_block(&seq(1, 20), 10), &p).flagged);
    }

    #[test]
    fn long_cycle_caught_by_larger_window() {
        let p = RepeatStopParams::default();
        let st = detect(&repeat_block(&seq(1, 250), 4), &p);
        assert!(st.flagged);
        assert!(st.window >= 600);
    }

    #[test]
    fn params_from_json() {
        let p = RepeatStopParams::from_json(&serde_json::json!({"windows": [100], "min_tokens": 50, "coverage": 0.5}));
        assert_eq!(p.windows, vec![100]);
        assert_eq!(p.min_tokens, 50);
        assert_eq!(p.min_repeats, 3);
        assert!((p.coverage - 0.5).abs() < 1e-6);
    }
}
