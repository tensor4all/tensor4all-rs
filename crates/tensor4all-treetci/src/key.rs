/// Canonical ordered subtree key for one side of an edge bipartition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtreeKey(Box<[usize]>);

impl SubtreeKey {
    /// Create a canonical subtree key from site ids.
    pub fn new(mut sites: Vec<usize>) -> Self {
        sites.sort_unstable();
        sites.dedup();
        Self(sites.into_boxed_slice())
    }

    /// Borrow the ordered site ids.
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}
