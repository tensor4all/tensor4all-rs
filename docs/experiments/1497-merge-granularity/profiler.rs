//! Diagnostic-only serial scopes; copied into a disposable pinned checkout.
use super::InputKey;
use std::collections::BTreeMap;
use std::time::Instant;

fn enabled() -> bool {
    std::env::var_os("T4A_MERGE_GRANULARITY").is_some()
}

#[derive(Default)]
struct Parent {
    ns: u128,
    items: usize,
    additions: usize,
}

pub(super) struct Probe {
    label: String,
    start: Option<Instant>,
    parents: BTreeMap<InputKey, Parent>,
    input_items: usize,
    additions_before: usize,
}

impl Probe {
    pub(super) fn new(label: &str, input_items: usize, additions_before: usize) -> Self {
        Self { label: label.into(), start: enabled().then(Instant::now), parents: BTreeMap::new(), input_items, additions_before }
    }

    pub(super) fn timer(&self) -> Option<Instant> {
        self.start.map(|_| Instant::now())
    }

    pub(super) fn record(&mut self, key: InputKey, start: Option<Instant>, additions: usize) {
        if let Some(start) = start {
            let elapsed = start.elapsed().as_nanos();
            let parent = self.parents.entry(key).or_default();
            parent.ns += elapsed;
            parent.items += 1;
            parent.additions += additions;
        }
    }

    pub(super) fn finish(self, additions_after: usize) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed().as_nanos();
            let emission = Instant::now();
            let parents = self.parents.values().enumerate().map(|(ordinal, p)| {
                format!("{{\"ordinal\":{ordinal},\"ns\":{},\"items\":{},\"additions\":{}}}", p.ns, p.items, p.additions)
            }).collect::<Vec<_>>().join(",");
            eprintln!("T4A_PARENT_PROBE {{\"label\":\"{}\",\"ns\":{elapsed},\"input_items\":{},\"additions\":{},\"parents\":[{parents}]}}", self.label, self.input_items, additions_after - self.additions_before);
            eprintln!("T4A_EMIT {{\"ns\":{}}}", emission.elapsed().as_nanos());
        }
    }
}

pub(super) struct Candidate {
    label: String,
    start: Option<Instant>,
    complete: bool,
}

impl Candidate {
    pub(super) fn new(label: impl Into<String>) -> Self {
        Self { label: label.into(), start: enabled().then(Instant::now), complete: false }
    }
    pub(super) fn complete(&mut self) { self.complete = true; }
}

impl Drop for Candidate {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed().as_nanos();
            let emission = Instant::now();
            eprintln!("T4A_CANDIDATE {{\"label\":\"{}\",\"ns\":{elapsed},\"complete\":{}}}", self.label, self.complete);
            eprintln!("T4A_EMIT {{\"ns\":{}}}", emission.elapsed().as_nanos());
        }
    }
}

pub(super) struct Region {
    level: usize,
    ordinal: usize,
    start: Option<Instant>,
}

impl Region {
    pub(super) fn new(level: usize, ordinal: usize) -> Self {
        Self { level, ordinal, start: enabled().then(Instant::now) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed().as_nanos();
            let emission = Instant::now();
            let threads = std::fs::read_to_string("/proc/self/status").ok().and_then(|status| {
                status.lines().find_map(|line| line.strip_prefix("Threads:").and_then(|value| value.trim().parse::<usize>().ok()))
            }).map_or_else(|| "null".to_owned(), |count| count.to_string());
            eprintln!("T4A_REGION {{\"level\":{},\"ordinal\":{},\"ns\":{elapsed},\"os_threads\":{threads}}}", self.level, self.ordinal);
            eprintln!("T4A_EMIT {{\"ns\":{}}}", emission.elapsed().as_nanos());
        }
    }
}
