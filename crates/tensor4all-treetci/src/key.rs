/// Canonical ordered subtree key for one side of an edge bipartition.
///
/// A `SubtreeKey` identifies a set of tree sites by their sorted, deduplicated
/// site indices. It is used as a `HashMap` key for storing pivot sets in
/// [`TreeTCI2`](crate::TreeTCI2).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::SubtreeKey;
///
/// // Sites are sorted and deduplicated on construction
/// let key = SubtreeKey::new(vec![3, 1, 2, 1]);
/// assert_eq!(key.as_slice(), &[1, 2, 3]);
///
/// // Two keys with the same sites are equal
/// let key2 = SubtreeKey::new(vec![2, 3, 1]);
/// assert_eq!(key, key2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtreeKey(Box<[usize]>);

impl SubtreeKey {
    /// Create a canonical subtree key from site ids.
    ///
    /// The input sites are sorted and deduplicated.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::SubtreeKey;
    ///
    /// let key = SubtreeKey::new(vec![5, 0, 3]);
    /// assert_eq!(key.as_slice(), &[0, 3, 5]);
    /// ```
    pub fn new(mut sites: Vec<usize>) -> Self {
        sites.sort_unstable();
        sites.dedup();
        Self(sites.into_boxed_slice())
    }

    /// Borrow the ordered site ids.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::SubtreeKey;
    ///
    /// let key = SubtreeKey::new(vec![2, 0]);
    /// assert_eq!(key.as_slice(), &[0, 2]);
    /// assert_eq!(key.as_slice().len(), 2);
    /// ```
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}
