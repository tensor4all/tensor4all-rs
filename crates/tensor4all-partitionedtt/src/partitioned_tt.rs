//! PartitionedTT: A collection of non-overlapping SubDomainTTs
//!
//! A `PartitionedTT` represents a tensor train that is decomposed into
//! multiple independent sub-components, each associated with a projector
//! onto a subdomain of the full index set.

use std::collections::HashMap;

use crate::error::{PartitionedTTError, Result};
use crate::projector::Projector;
use crate::subdomain_tt::SubDomainTT;
use tensor4all_core::DynIndex;
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// A partitioned tensor train: a collection of non-overlapping SubDomainTTs.
///
/// Each SubDomainTT covers a disjoint region of the index space defined by
/// its projector. The projectors must be mutually disjoint (non-overlapping).
#[derive(Debug, Clone, Default)]
pub struct PartitionedTT {
    /// Map from projector to subdomain
    data: HashMap<Projector, SubDomainTT>,
}

impl PartitionedTT {
    /// Create an empty partitioned tensor train.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a PartitionedTT from a vector of SubDomainTTs.
    ///
    /// Returns an error if the projectors are not mutually disjoint.
    pub fn from_subdomains(subdomains: Vec<SubDomainTT>) -> Result<Self> {
        // Check that projectors are disjoint
        let projectors: Vec<_> = subdomains.iter().map(|s| s.projector().clone()).collect();
        if !Projector::are_disjoint(&projectors) {
            return Err(PartitionedTTError::OverlappingProjectors);
        }

        let mut data = HashMap::new();
        for subdomain in subdomains {
            data.insert(subdomain.projector().clone(), subdomain);
        }

        Ok(Self { data })
    }

    /// Create a PartitionedTT from a single SubDomainTT.
    pub fn from_subdomain(subdomain: SubDomainTT) -> Self {
        let mut data = HashMap::new();
        data.insert(subdomain.projector().clone(), subdomain);
        Self { data }
    }

    /// Number of subdomains (patches).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get projectors as an iterator.
    pub fn projectors(&self) -> impl Iterator<Item = &Projector> {
        self.data.keys()
    }

    /// Get subdomain by projector.
    pub fn get(&self, projector: &Projector) -> Option<&SubDomainTT> {
        self.data.get(projector)
    }

    /// Get mutable subdomain by projector.
    pub fn get_mut(&mut self, projector: &Projector) -> Option<&mut SubDomainTT> {
        self.data.get_mut(projector)
    }

    /// Check if a projector exists.
    pub fn contains(&self, projector: &Projector) -> bool {
        self.data.contains_key(projector)
    }

    /// Iterate over (projector, subdomain) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Projector, &SubDomainTT)> {
        self.data.iter()
    }

    /// Iterate over subdomains.
    pub fn values(&self) -> impl Iterator<Item = &SubDomainTT> {
        self.data.values()
    }

    /// Iterate over mutable subdomains.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut SubDomainTT> {
        self.data.values_mut()
    }

    /// Insert a subdomain, replacing any existing one with the same projector.
    pub fn insert(&mut self, subdomain: SubDomainTT) {
        self.data.insert(subdomain.projector().clone(), subdomain);
    }

    /// Append another PartitionedTT (must have non-overlapping projectors).
    pub fn append(&mut self, other: Self) -> Result<()> {
        // Check for overlap
        for proj in other.data.keys() {
            for existing_proj in self.data.keys() {
                if proj.is_compatible_with(existing_proj) {
                    return Err(PartitionedTTError::OverlappingProjectors);
                }
            }
        }

        // Merge
        for (proj, subdomain) in other.data {
            self.data.insert(proj, subdomain);
        }

        Ok(())
    }

    /// Append subdomains.
    pub fn append_subdomains(&mut self, subdomains: Vec<SubDomainTT>) -> Result<()> {
        let other = Self::from_subdomains(subdomains)?;
        self.append(other)
    }

    /// Compute the total Frobenius norm (sqrt of sum of squared norms).
    pub fn norm(&self) -> f64 {
        let sum_sq: f64 = self.data.values().map(|s| s.norm_squared()).sum();
        sum_sq.sqrt()
    }

    /// Contract with another PartitionedTT.
    ///
    /// Performs pairwise contraction of compatible SubDomainTTs and combines results.
    pub fn contract(&self, other: &Self, options: &ContractOptions) -> Result<Self> {
        let mut result = Self::new();

        // Build contraction tasks
        let tasks = self.contraction_tasks(other)?;

        // Execute contractions
        for (proj, m1, m2) in tasks {
            if let Some(contracted) = m1.contract(&m2, options)? {
                // Check if we already have a subdomain with the same projector
                if let Some(existing) = result.get_mut(&proj) {
                    // Sum the subdomains using TT addition
                    let mut summed_tt = existing.data().add(contracted.data()).map_err(|e| {
                        PartitionedTTError::TensorTrainError(format!(
                            "TT addition in contract failed: {}",
                            e
                        ))
                    })?;
                    // Truncate after addition using the same truncation params as contraction
                    let mut truncate_opts = TruncateOptions::svd();
                    if let Some(rtol) = options.rtol() {
                        truncate_opts = truncate_opts.with_rtol(rtol);
                    }
                    if let Some(max_rank) = options.max_rank() {
                        truncate_opts = truncate_opts.with_max_rank(max_rank);
                    }
                    summed_tt.truncate(&truncate_opts).map_err(|e| {
                        PartitionedTTError::TensorTrainError(format!(
                            "TT truncation after addition failed: {}",
                            e
                        ))
                    })?;
                    *existing = SubDomainTT::new(summed_tt, proj.clone());
                } else {
                    result.insert(contracted);
                }
            }
        }

        Ok(result)
    }

    /// Add another PartitionedTT patch-by-patch.
    ///
    /// Both PartitionedTTs must have compatible patch structures:
    /// - The union of all projectors from both must be pairwise disjoint
    /// - Missing patches in either side are allowed (treated as zero)
    ///
    /// For each projector present in both, the corresponding TTs are added
    /// and then truncated according to the provided options.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The projectors are not pairwise disjoint (overlapping patches)
    /// - TT addition or truncation fails
    pub fn add(&self, other: &Self, options: &TruncateOptions) -> Result<Self> {
        // Collect unique projectors from both (union)
        let mut unique_projectors: std::collections::HashSet<Projector> =
            self.projectors().cloned().collect();
        unique_projectors.extend(other.projectors().cloned());
        let all_projectors: Vec<Projector> = unique_projectors.into_iter().collect();

        // Check that all unique projectors are pairwise disjoint
        if !Projector::are_disjoint(&all_projectors) {
            return Err(PartitionedTTError::IncompatibleProjectors(
                "Projectors must be pairwise disjoint for patch-wise addition".to_string(),
            ));
        }

        let mut result = Self::new();

        // Process projectors from self
        for (proj, subdomain) in self.iter() {
            if let Some(other_subdomain) = other.get(proj) {
                // Both have this projector: add and truncate
                let mut summed_tt = subdomain.data().add(other_subdomain.data()).map_err(|e| {
                    PartitionedTTError::TensorTrainError(format!(
                        "TT addition in add failed: {}",
                        e
                    ))
                })?;
                summed_tt.truncate(options).map_err(|e| {
                    PartitionedTTError::TensorTrainError(format!(
                        "TT truncation after addition failed: {}",
                        e
                    ))
                })?;
                result.insert(SubDomainTT::new(summed_tt, proj.clone()));
            } else {
                // Only self has this projector: clone it
                result.insert(subdomain.clone());
            }
        }

        // Process projectors only in other (not in self)
        for (proj, subdomain) in other.iter() {
            if !self.contains(proj) {
                result.insert(subdomain.clone());
            }
        }

        Ok(result)
    }

    /// Build contraction tasks for two PartitionedTTs.
    fn contraction_tasks(
        &self,
        other: &Self,
    ) -> Result<Vec<(Projector, SubDomainTT, SubDomainTT)>> {
        let mut tasks = Vec::new();

        for m1 in self.data.values() {
            for m2 in other.data.values() {
                // Check if projectors are compatible
                if m1.projector().is_compatible_with(m2.projector()) {
                    // Compute the projector after contraction
                    let indices1: std::collections::HashSet<_> =
                        m1.all_indices().into_iter().collect();
                    let indices2: std::collections::HashSet<_> =
                        m2.all_indices().into_iter().collect();

                    // External indices
                    let common: std::collections::HashSet<_> =
                        indices1.intersection(&indices2).cloned().collect();
                    let all: std::collections::HashSet<_> =
                        indices1.union(&indices2).cloned().collect();
                    let external: std::collections::HashSet<DynIndex> =
                        all.difference(&common).cloned().collect();

                    // Build projector for external indices
                    let mut proj_data = Vec::new();
                    for idx in &external {
                        if let Some(val) = m1.projector().get(idx) {
                            proj_data.push((idx.clone(), val));
                        } else if let Some(val) = m2.projector().get(idx) {
                            proj_data.push((idx.clone(), val));
                        }
                    }
                    let proj_after = Projector::from_pairs(proj_data);

                    // Project inputs to the combined projector
                    let merged_proj = m1.projector().intersection(m2.projector());
                    if let Some(merged) = merged_proj {
                        let m1_proj = m1.project(&merged).unwrap_or_else(|| m1.clone());
                        let m2_proj = m2.project(&merged).unwrap_or_else(|| m2.clone());
                        tasks.push((proj_after, m1_proj, m2_proj));
                    }
                }
            }
        }

        Ok(tasks)
    }

    /// Convert to a single TensorTrain by summing all subdomains.
    ///
    /// Uses direct-sum (block) addition to combine all SubDomainTT tensors.
    /// The result has bond dimension equal to the sum of individual bond dimensions.
    ///
    /// Subdomains are processed in a deterministic order (sorted by projector)
    /// to ensure reproducible results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The PartitionedTT is empty
    /// - The subdomains have incompatible structures (different lengths)
    pub fn to_tensor_train(&self) -> Result<TensorTrain> {
        if self.is_empty() {
            return Err(PartitionedTTError::Empty);
        }

        // Sort subdomains by projector for deterministic ordering
        let mut sorted: Vec<_> = self.data.iter().collect();
        sorted.sort_by(|(p1, _), (p2, _)| Self::projector_cmp(p1, p2));

        let mut iter = sorted.into_iter().map(|(_, subdomain)| subdomain);
        let first = iter.next().unwrap();
        let mut result = first.data().clone();

        for subdomain in iter {
            result = result.add(subdomain.data()).map_err(|e| {
                PartitionedTTError::TensorTrainError(format!("TT addition failed: {}", e))
            })?;
        }

        Ok(result)
    }

    /// Compare two projectors for deterministic ordering.
    ///
    /// Orders by: number of projections, then by sorted (index_id, value) pairs.
    fn projector_cmp(a: &Projector, b: &Projector) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare by length
        match a.len().cmp(&b.len()) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Then compare by sorted (id, value) pairs
        let mut a_pairs: Vec<_> = a.iter().map(|(idx, &val)| (idx.id, val)).collect();
        let mut b_pairs: Vec<_> = b.iter().map(|(idx, &val)| (idx.id, val)).collect();
        a_pairs.sort();
        b_pairs.sort();

        a_pairs.cmp(&b_pairs)
    }
}

impl std::ops::Index<&Projector> for PartitionedTT {
    type Output = SubDomainTT;

    fn index(&self, projector: &Projector) -> &Self::Output {
        self.data
            .get(projector)
            .expect("Projector not found in PartitionedTT")
    }
}

impl IntoIterator for PartitionedTT {
    type Item = (Projector, SubDomainTT);
    type IntoIter = std::collections::hash_map::IntoIter<Projector, SubDomainTT>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a PartitionedTT {
    type Item = (&'a Projector, &'a SubDomainTT);
    type IntoIter = std::collections::hash_map::Iter<'a, Projector, SubDomainTT>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::index::Index;
    use tensor4all_core::StorageScalar;
    use tensor4all_core::TensorDynLen;

    fn make_index(size: usize) -> DynIndex {
        Index::new_dyn(size)
    }

    fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
        let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
        let size: usize = dims.iter().product();
        let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
        let storage = f64::dense_storage_with_shape(data, &dims);
        TensorDynLen::new(indices, storage)
    }

    /// Create shared indices for testing
    fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
        let s0 = make_index(2); // site 0
        let l01 = make_index(3); // link 0-1
        let s1 = make_index(2); // site 1
        (vec![s0, s1], l01)
    }

    /// Create a TT using the provided indices
    fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
        let t0 = make_tensor(vec![site_inds[0].clone(), link_ind.clone()]);
        let t1 = make_tensor(vec![link_ind.clone(), site_inds[1].clone()]);
        TensorTrain::new(vec![t0, t1]).unwrap()
    }

    fn make_simple_tt() -> (TensorTrain, Vec<DynIndex>) {
        let (site_inds, link_ind) = make_shared_indices();
        let tt = make_tt_with_indices(&site_inds, &link_ind);
        (tt, site_inds)
    }

    #[test]
    fn test_partitioned_tt_creation() {
        // Create shared indices so both TTs have the same site indices
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

        let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

        assert_eq!(partitioned.len(), 2);
        assert!(!partitioned.is_empty());
    }

    #[test]
    fn test_partitioned_tt_empty() {
        let partitioned = PartitionedTT::new();

        assert_eq!(partitioned.len(), 0);
        assert!(partitioned.is_empty());
    }

    #[test]
    fn test_partitioned_tt_overlapping_projectors() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        // Same projector as subdomain1 - this should fail
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let result = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partitioned_tt_norm() {
        let (tt, _) = make_simple_tt();
        let subdomain = SubDomainTT::from_tt(tt);
        let partitioned = PartitionedTT::from_subdomain(subdomain);

        let norm = partitioned.norm();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_partitioned_tt_append() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let mut partitioned1 = PartitionedTT::from_subdomain(subdomain1);

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));
        let partitioned2 = PartitionedTT::from_subdomain(subdomain2);

        partitioned1.append(partitioned2).unwrap();

        assert_eq!(partitioned1.len(), 2);
    }

    #[test]
    fn test_partitioned_tt_append_overlapping() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let mut partitioned1 = PartitionedTT::from_subdomain(subdomain1);

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        // Same projector - should fail
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let partitioned2 = PartitionedTT::from_subdomain(subdomain2);

        let result = partitioned1.append(partitioned2);
        assert!(result.is_err());
    }

    #[test]
    fn test_partitioned_tt_iteration() {
        let (site_inds, link_ind) = make_shared_indices();

        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

        let partitioned = PartitionedTT::from_subdomains(vec![subdomain1, subdomain2]).unwrap();

        let count = partitioned.iter().count();
        assert_eq!(count, 2);

        let projector_count = partitioned.projectors().count();
        assert_eq!(projector_count, 2);
    }

    #[test]
    fn test_partitioned_tt_index() {
        let (tt, site_inds) = make_simple_tt();
        let projector = Projector::from_pairs([(site_inds[0].clone(), 0)]);
        let subdomain = SubDomainTT::new(tt, projector.clone());
        let partitioned = PartitionedTT::from_subdomain(subdomain);

        // Access by projector
        let retrieved = &partitioned[&projector];
        assert_eq!(retrieved.projector(), &projector);
    }

    /// Create contraction test indices:
    /// TT1: s0 -- l01 -- s1
    /// TT2: s1 -- l12 -- s2
    fn make_contraction_indices() -> (DynIndex, DynIndex, DynIndex, DynIndex, DynIndex) {
        let s0 = make_index(2);
        let l01 = make_index(3);
        let s1 = make_index(2);
        let l12 = make_index(3);
        let s2 = make_index(2);
        (s0, l01, s1, l12, s2)
    }

    fn make_tt1(s0: &DynIndex, l01: &DynIndex, s1: &DynIndex) -> TensorTrain {
        let t0 = make_tensor(vec![s0.clone(), l01.clone()]);
        let t1 = make_tensor(vec![l01.clone(), s1.clone()]);
        TensorTrain::new(vec![t0, t1]).unwrap()
    }

    fn make_tt2(s1: &DynIndex, l12: &DynIndex, s2: &DynIndex) -> TensorTrain {
        let t0 = make_tensor(vec![s1.clone(), l12.clone()]);
        let t1 = make_tensor(vec![l12.clone(), s2.clone()]);
        TensorTrain::new(vec![t0, t1]).unwrap()
    }

    #[test]
    fn test_partitioned_tt_contract_numerical() {
        let (s0, l01, s1, l12, s2) = make_contraction_indices();

        // Create PartitionedTT1 with 2 subdomains: s0=0 and s0=1
        let tt1_a = make_tt1(&s0, &l01, &s1);
        let tt1_b = make_tt1(&s0, &l01, &s1);
        let subdomain1_a = SubDomainTT::new(tt1_a, Projector::from_pairs([(s0.clone(), 0)]));
        let subdomain1_b = SubDomainTT::new(tt1_b, Projector::from_pairs([(s0.clone(), 1)]));
        let partitioned1 =
            PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

        // Create PartitionedTT2 with 2 subdomains: s2=0 and s2=1
        let tt2_a = make_tt2(&s1, &l12, &s2);
        let tt2_b = make_tt2(&s1, &l12, &s2);
        let subdomain2_a = SubDomainTT::new(tt2_a, Projector::from_pairs([(s2.clone(), 0)]));
        let subdomain2_b = SubDomainTT::new(tt2_b, Projector::from_pairs([(s2.clone(), 1)]));
        let partitioned2 =
            PartitionedTT::from_subdomains(vec![subdomain2_a, subdomain2_b]).unwrap();

        // Contract
        let options = ContractOptions::default();
        let result = partitioned1.contract(&partitioned2, &options).unwrap();

        // Result should have 4 subdomains: (s0=0,s2=0), (s0=0,s2=1), (s0=1,s2=0), (s0=1,s2=1)
        assert_eq!(result.len(), 4);

        // Verify each subdomain numerically
        for (proj, subdomain) in result.iter() {
            let contracted_full = subdomain.data().to_dense().unwrap();
            let contracted_data = contracted_full.as_slice_f64().unwrap();

            // Get the projected values
            let s0_val = proj.get(&s0).unwrap();
            let s2_val = proj.get(&s2).unwrap();

            // Compute expected by projecting full TTs
            let tt1 = make_tt1(&s0, &l01, &s1);
            let tt2 = make_tt2(&s1, &l12, &s2);
            let t1_full = tt1.to_dense().unwrap();
            let t2_full = tt2.to_dense().unwrap();

            // Project t1 to s0=s0_val
            let t1_data = t1_full.as_slice_f64().unwrap();
            let mut t1_proj_data = vec![0.0f64; t1_data.len()];
            // s0 is first index, so s0_val*s1.dim to s0_val*s1.dim + s1.dim - 1
            for s1_idx in 0..s1.dim {
                t1_proj_data[s0_val * s1.dim + s1_idx] = t1_data[s0_val * s1.dim + s1_idx];
            }
            let t1_proj = TensorDynLen::from_dense_f64(vec![s0.clone(), s1.clone()], t1_proj_data);

            // Project t2 to s2=s2_val
            let t2_data = t2_full.as_slice_f64().unwrap();
            let mut t2_proj_data = vec![0.0f64; t2_data.len()];
            // s2 is second index
            for s1_idx in 0..s1.dim {
                t2_proj_data[s1_idx * s2.dim + s2_val] = t2_data[s1_idx * s2.dim + s2_val];
            }
            let t2_proj = TensorDynLen::from_dense_f64(vec![s1.clone(), s2.clone()], t2_proj_data);

            let expected = t1_proj.contract(&t2_proj);
            let expected_data = expected.as_slice_f64().unwrap();

            assert_eq!(
                contracted_data.len(),
                expected_data.len(),
                "Size mismatch for projector {:?}",
                proj
            );
            for (i, (&actual, &exp)) in contracted_data.iter().zip(expected_data.iter()).enumerate()
            {
                assert!(
                    (actual - exp).abs() < 1e-10,
                    "Mismatch at index {} for projector {:?}: actual={}, expected={}",
                    i,
                    proj,
                    actual,
                    exp
                );
            }
        }
    }

    #[test]
    fn test_subdomain_tt_norm_with_projector() {
        let (site_inds, link_ind) = make_shared_indices();

        // Create TT and get its full tensor
        let tt = make_tt_with_indices(&site_inds, &link_ind);
        let full_tensor = tt.to_dense().unwrap();
        let full_data = full_tensor.as_slice_f64().unwrap();

        // Create SubDomainTT with projector s0=0
        let projector = Projector::from_pairs([(site_inds[0].clone(), 0)]);
        let subdomain = SubDomainTT::new(tt.clone(), projector);

        // Current norm() returns the norm of the underlying TT data without projection
        // This test documents current behavior
        let norm_raw = subdomain.norm();
        let tt_norm = tt.norm();
        assert!((norm_raw - tt_norm).abs() < 1e-10);

        // Compute expected norm if projection were applied:
        // For s0=0, we keep only indices 0, 1 (first row of 2x2 matrix)
        let mut projected_sum_sq = 0.0;
        for &x in full_data.iter().take(site_inds[1].dim) {
            projected_sum_sq += x.powi(2);
        }
        let _expected_projected_norm = projected_sum_sq.sqrt();

        // Note: Current implementation returns raw TT norm, not projected norm
        // If we want projected norm, we'd need to modify the implementation
        // For now, this test just verifies current behavior is consistent
    }

    #[test]
    fn test_partitioned_tt_add_same_structure() {
        let (site_inds, link_ind) = make_shared_indices();

        // Create two PartitionedTTs with the same patch structure
        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let partitioned1 = PartitionedTT::from_subdomain(subdomain1);

        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let partitioned2 = PartitionedTT::from_subdomain(subdomain2);

        // Add them
        let options = TruncateOptions::svd();
        let result = partitioned1.add(&partitioned2, &options).unwrap();

        // Result should have 1 subdomain (same projector)
        assert_eq!(result.len(), 1);

        // The sum should be 2x the original (same TT added to itself)
        let proj = Projector::from_pairs([(site_inds[0].clone(), 0)]);
        let summed = result.get(&proj).unwrap();
        let summed_dense = summed.data().to_dense().unwrap();
        let summed_data = summed_dense.as_slice_f64().unwrap();

        let original = make_tt_with_indices(&site_inds, &link_ind);
        let original_dense = original.to_dense().unwrap();
        let original_data = original_dense.as_slice_f64().unwrap();

        for (i, (&s, &o)) in summed_data.iter().zip(original_data.iter()).enumerate() {
            assert!(
                (s - 2.0 * o).abs() < 1e-10,
                "Mismatch at index {}: summed={}, expected={}",
                i,
                s,
                2.0 * o
            );
        }
    }

    #[test]
    fn test_partitioned_tt_add_missing_patch() {
        let (site_inds, link_ind) = make_shared_indices();

        // partitioned1 has patches for s0=0 and s0=1
        let tt1_a = make_tt_with_indices(&site_inds, &link_ind);
        let tt1_b = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1_a =
            SubDomainTT::new(tt1_a, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let subdomain1_b =
            SubDomainTT::new(tt1_b, Projector::from_pairs([(site_inds[0].clone(), 1)]));
        let partitioned1 =
            PartitionedTT::from_subdomains(vec![subdomain1_a, subdomain1_b]).unwrap();

        // partitioned2 has only patch for s0=0
        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let partitioned2 = PartitionedTT::from_subdomain(subdomain2);

        // Add them
        let options = TruncateOptions::svd();
        let result = partitioned1.add(&partitioned2, &options).unwrap();

        // Result should have 2 subdomains
        assert_eq!(result.len(), 2);

        // s0=0 patch should be summed (2x)
        let proj0 = Projector::from_pairs([(site_inds[0].clone(), 0)]);
        let summed0 = result.get(&proj0).unwrap();
        let original = make_tt_with_indices(&site_inds, &link_ind);
        let original_norm = original.norm();
        // Norm of 2*TT is 2*norm(TT)
        assert!((summed0.norm() - 2.0 * original_norm).abs() < 1e-10);

        // s0=1 patch should be unchanged (only in partitioned1)
        let proj1 = Projector::from_pairs([(site_inds[0].clone(), 1)]);
        let unchanged = result.get(&proj1).unwrap();
        assert!((unchanged.norm() - original_norm).abs() < 1e-10);
    }

    #[test]
    fn test_partitioned_tt_add_overlapping_fails() {
        let (site_inds, link_ind) = make_shared_indices();

        // partitioned1 has patch for s0=0
        let tt1 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));
        let partitioned1 = PartitionedTT::from_subdomain(subdomain1);

        // partitioned2 has patch for s1=0 (overlaps with s0=0 since they're compatible)
        let tt2 = make_tt_with_indices(&site_inds, &link_ind);
        let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[1].clone(), 0)]));
        let partitioned2 = PartitionedTT::from_subdomain(subdomain2);

        // Add should fail because projectors are compatible (not disjoint)
        let options = TruncateOptions::svd();
        let result = partitioned1.add(&partitioned2, &options);
        assert!(result.is_err());
    }
}
