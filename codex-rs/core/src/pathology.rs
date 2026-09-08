//! Loop guard for the local llamacpp provider.
//!
//! Two pathologies are handled: a completion that loops (the server stops it
//! with finish reason `repetition`, or it runs to the token cap while
//! repeating) and a tool call repeated identically (the repeat guard). After
//! either, the text that repeated is sent as reference spans for an
//! escalating n-gram penalty on the next completion, so the model cannot
//! re-emit it verbatim. Each consecutive failing completion raises the
//! penalty scale by a step and lowers the first penalized match length by
//! one (down to 1, where every reference token is penalized without
//! context); the first clean completion clears the state.
//!
//! The state is session-scoped and survives user turns: it is cleared only
//! by a clean completion.

use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;

/// What triggered the penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Trigger {
    /// The server stopped the completion for looping, or the cap was hit while looping.
    Loop,
    /// An identical tool call was repeated.
    RepeatedToolCall,
}

impl Trigger {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Trigger::Loop => "loop",
            Trigger::RepeatedToolCall => "repeated_tool_call",
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PathologyState {
    /// Penalty active for the next completion.
    pub(crate) active: bool,
    /// Consecutive completions that failed (looped or repeated a call).
    pub(crate) consecutive_failures: u32,
    /// Current penalty scale.
    pub(crate) scale: f32,
    /// Current first penalized match length; drops by one per failure, down to 1.
    pub(crate) start_n: u32,
    /// Reference spans (text) for the penalty.
    pub(crate) spans: Vec<String>,
    /// Set by `on_failure` since the last `begin_call`.
    failed_since_begin: bool,
    /// The last completion finished cleanly (as far as the finish reason shows).
    clean_finish: bool,
}

/// First line of defense, fully sealed off from the penalty machinery: a
/// budget of clean retries per pathology episode. A claimed resample
/// discards the failed completion and re-issues the identical request with
/// a fresh seed -- no penalties, no injected notes, and nothing recorded in
/// `PathologyState`. Only when the budget is exhausted does a failure fall
/// through to the escalating-penalty flow, which then behaves exactly as it
/// always has. The budget refills on a clean completion.
#[derive(Debug, Default)]
pub(crate) struct ResampleState {
    /// Resamples used since the last clean completion.
    used: u32,
    /// A repeated tool call claimed a resample at dispatch; the turn driver
    /// takes this to discard the completion.
    requested: bool,
}

impl ResampleState {
    /// Claim one resample from the budget (loop / cap triggers).
    pub(crate) fn try_claim(&mut self, budget: u32) -> bool {
        if self.used < budget {
            self.used += 1;
            true
        } else {
            false
        }
    }

    /// Claim a resample for a repeated tool call at dispatch time. Returns
    /// true when this call should be skipped and the completion discarded.
    /// Idempotent within one completion: once requested, later calls skip
    /// without consuming more budget.
    pub(crate) fn request_repeated_call(&mut self, budget: u32) -> bool {
        if self.requested {
            return true;
        }
        if self.used < budget {
            self.used += 1;
            self.requested = true;
            true
        } else {
            false
        }
    }

    /// Whether a resample was requested at dispatch; clears the request.
    pub(crate) fn take_requested(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }

    /// Whether a resample is already requested for the current completion.
    pub(crate) fn is_requested(&self) -> bool {
        self.requested
    }

    /// Attempts used so far in this episode (for logging).
    pub(crate) fn used(&self) -> u32 {
        self.used
    }


    /// A clean completion refills the budget.
    pub(crate) fn reset(&mut self) {
        *self = ResampleState::default();
    }
}

/// Longest span kept; longer spans are trimmed from the front (the tail is
/// what the model would continue from).
const MAX_SPAN_CHARS: usize = 4000;
const MAX_SPANS: usize = 16;

impl PathologyState {
    /// Called before every completion. A clean finish with no failure since
    /// resets the state (rule: the first clean call resets).
    pub(crate) fn begin_call(&mut self) {
        if self.clean_finish && !self.failed_since_begin {
            *self = PathologyState::default();
        }
        self.failed_since_begin = false;
        self.clean_finish = false;
    }

    /// Record a clean finish of a completion.
    pub(crate) fn note_clean_finish(&mut self) {
        self.clean_finish = true;
    }

    /// Record a failure. `spans` is the text that repeated.
    pub(crate) fn on_failure(
        &mut self,
        trigger: Trigger,
        spans: Vec<String>,
        params: &codex_llamacpp::PathologyParams,
    ) {
        let _ = trigger;
        self.clean_finish = false;
        self.failed_since_begin = true;
        self.consecutive_failures += 1;
        self.scale = params.ngram_penalty_scale
            + params.ngram_penalty_step * (self.consecutive_failures - 1) as f32;
        self.start_n = params
            .ngram_penalty_start_n
            .saturating_sub(self.consecutive_failures - 1)
            .max(1);
        self.active = params.ngram_penalty_scale > 0.0;
        for span in spans {
            let span = trim_span(&span);
            if span.is_empty() || self.spans.iter().any(|s| s == &span) {
                continue;
            }
            self.spans.push(span);
        }
        while self.spans.len() > MAX_SPANS {
            self.spans.remove(0);
        }
    }

    /// Extra request fields for the next completion.
    pub(crate) fn request_extra(&self, params: &codex_llamacpp::PathologyParams) -> Option<Value> {
        let mut extra = serde_json::Map::new();
        if params.repeat_stop {
            extra.insert("repeat_stop".to_string(), json!({}));
        }
        if self.active && !self.spans.is_empty() && self.scale > 0.0 {
            extra.insert(
                "ngram_penalty".to_string(),
                json!({
                    "spans": self.spans,
                    "start_n": self.start_n,
                    "scale": self.scale,
                    "max": params.ngram_penalty_max,
                }),
            );
        }
        if extra.is_empty() {
            None
        } else {
            Some(Value::Object(extra))
        }
    }

    pub(crate) fn summary(&self) -> Value {
        json!({
            "active": self.active,
            "consecutive_failures": self.consecutive_failures,
            "scale": self.scale,
            "start_n": self.start_n,
            "spans": self.spans.len(),
        })
    }
}

fn trim_span(span: &str) -> String {
    let count = span.chars().count();
    if count <= MAX_SPAN_CHARS {
        return span.to_string();
    }
    span.chars().skip(count - MAX_SPAN_CHARS).collect()
}

/// Repetition statistics reported by the server for one completion, keyed by
/// (thread id, model-call ordinal). The transport stores them; the turn loop
/// takes them when the completion finishes.
static REPETITION_STATS: Mutex<Option<HashMap<(String, u64), Value>>> = Mutex::new(None);

pub(crate) fn store_repetition_stats(thread_id: &str, call_ordinal: u64, stats: Value) {
    if let Ok(mut guard) = REPETITION_STATS.lock() {
        guard
            .get_or_insert_with(HashMap::new)
            .insert((thread_id.to_string(), call_ordinal), stats);
    }
}

pub(crate) fn take_repetition_stats(thread_id: &str, call_ordinal: u64) -> Option<Value> {
    let mut guard = REPETITION_STATS.lock().ok()?;
    guard
        .as_mut()
        .and_then(|m| m.remove(&(thread_id.to_string(), call_ordinal)))
}

/// Classify a finished completion: did it loop? Returns the repeated text
/// when it did.
pub(crate) fn loop_span(finish_reason: Option<&str>, stats: Option<&Value>) -> Option<String> {
    let flagged = stats
        .and_then(|s| s.get("flagged"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let is_loop = finish_reason == Some(REPETITION_FINISH_REASON) || (finish_reason == Some("length") && flagged);
    if !is_loop {
        return None;
    }
    Some(
        stats
            .and_then(|s| s.get("gram_text"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    )
}

/// Finish reason the server reports when the repetition stop fired.
pub(crate) const REPETITION_FINISH_REASON: &str = "repetition";

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> codex_llamacpp::PathologyParams {
        codex_llamacpp::PathologyParams::default()
    }

    #[test]
    fn resample_budget_claims_then_exhausts_and_resets() {
        let mut r = ResampleState::default();
        assert!(r.try_claim(2));
        assert!(r.try_claim(2));
        assert!(!r.try_claim(2));
        r.reset();
        assert!(r.try_claim(2));
    }

    #[test]
    fn repeated_call_request_is_idempotent_within_a_completion() {
        let mut r = ResampleState::default();
        assert!(r.request_repeated_call(1));
        assert!(r.request_repeated_call(1)); // same completion: no extra budget
        assert_eq!(r.used(), 1);
        assert!(r.take_requested());
        assert!(!r.take_requested());
        assert!(!r.request_repeated_call(1)); // budget gone
    }

    #[test]
    fn zero_budget_never_claims() {
        let mut r = ResampleState::default();
        assert!(!r.try_claim(0));
        assert!(!r.request_repeated_call(0));
    }

    #[test]
    fn scale_escalates_and_resets_on_clean_call() {
        let p = params();
        let mut st = PathologyState::default();
        st.begin_call();
        st.on_failure(Trigger::Loop, vec!["abc".into()], &p);
        assert_eq!(st.scale, 1.0);
        assert_eq!(st.start_n, 3);
        st.begin_call();
        st.on_failure(Trigger::RepeatedToolCall, vec!["def".into()], &p);
        assert_eq!(st.scale, 1.5);
        assert_eq!(st.start_n, 2);
        assert_eq!(st.spans, vec!["abc".to_string(), "def".to_string()]);
        // a clean completion: penalty still applies to that call, reset at the next begin_call
        st.begin_call();
        st.note_clean_finish();
        assert!(st.request_extra(&p).unwrap().get("ngram_penalty").is_some());
        st.begin_call();
        assert!(!st.active);
        assert_eq!(st.consecutive_failures, 0);
        assert!(st.spans.is_empty());
        assert!(st.request_extra(&p).unwrap().get("ngram_penalty").is_none());
    }

    #[test]
    fn repeat_warning_after_clean_finish_does_not_reset() {
        let p = params();
        let mut st = PathologyState::default();
        st.begin_call();
        st.on_failure(Trigger::Loop, vec!["abc".into()], &p);
        st.begin_call();
        st.note_clean_finish();
        // the tool call of that completion turned out to be a repeat
        st.on_failure(Trigger::RepeatedToolCall, vec!["call".into()], &p);
        st.begin_call();
        assert!(st.active);
        assert_eq!(st.consecutive_failures, 2);
        assert_eq!(st.scale, 1.5);
        assert_eq!(st.start_n, 2);
    }

    #[test]
    fn start_n_drops_to_one_and_stays() {
        let p = params();
        let mut st = PathologyState::default();
        for _ in 0..4 {
            st.begin_call();
            st.on_failure(Trigger::Loop, vec!["x".into()], &p);
        }
        assert_eq!(st.consecutive_failures, 4);
        assert_eq!(st.scale, 2.5);
        assert_eq!(st.start_n, 1);
        assert_eq!(st.request_extra(&p).unwrap()["ngram_penalty"]["start_n"], json!(1));
    }

    #[test]
    fn request_extra_shape() {
        let p = params();
        let mut st = PathologyState::default();
        let extra = st.request_extra(&p).unwrap();
        assert_eq!(extra, json!({"repeat_stop": {}}));
        st.on_failure(Trigger::Loop, vec!["x y z".into()], &p);
        let extra = st.request_extra(&p).unwrap();
        assert_eq!(extra["ngram_penalty"]["spans"], json!(["x y z"]));
        assert_eq!(extra["ngram_penalty"]["start_n"], json!(3));
        assert_eq!(extra["ngram_penalty"]["max"], json!(10.0));
        let off = codex_llamacpp::PathologyParams { repeat_stop: false, ngram_penalty_scale: 0.0, ..p };
        let mut st2 = PathologyState::default();
        st2.on_failure(Trigger::Loop, vec!["x".into()], &off);
        assert!(st2.request_extra(&off).is_none());
    }

    #[test]
    fn loop_classification() {
        assert_eq!(loop_span(Some("repetition"), Some(&json!({"gram_text": "ab"}))), Some("ab".into()));
        assert_eq!(loop_span(Some("length"), Some(&json!({"flagged": true, "gram_text": "ab"}))), Some("ab".into()));
        assert_eq!(loop_span(Some("length"), Some(&json!({"flagged": false, "gram_text": "ab"}))), None);
        assert_eq!(loop_span(Some("stop"), None), None);
    }

    #[test]
    fn stats_registry_roundtrip() {
        store_repetition_stats("t", 7, json!({"flagged": true}));
        assert_eq!(take_repetition_stats("t", 7), Some(json!({"flagged": true})));
        assert_eq!(take_repetition_stats("t", 7), None);
    }
}
