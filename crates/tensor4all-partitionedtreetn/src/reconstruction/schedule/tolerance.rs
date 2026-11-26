//! Tolerance-only metadata planning and serial candidate selection.

use super::*;

type Level<V> = BTreeMap<OutputKey, BTreeMap<InputKey, Item<V>>>;

#[derive(Debug, Clone, Copy)]
pub(super) struct Plan {
    pub active: bool,
    pub merging_levels_left: usize,
}

/// Predict key collisions without tensors, truncated at the executed depth.
pub(super) fn plan<V>(
    preimage: &ReconstructionTarget<V>,
    geometry: &ScheduleGeometry,
    depth: usize,
) -> Result<Vec<Plan>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut cells: BTreeMap<usize, BTreeSet<InputKey>> = BTreeMap::new();
    for (projector, groups) in preimage.patch_projectors().zip(&geometry.source_groups) {
        let key = geometry
            .input_prefix(projector)
            .ok_or_else(|| invalid("validated input prefix could not be represented"))?;
        for &group in groups {
            cells.entry(group).or_default().insert(key.clone());
        }
    }
    forecast(cells, depth)
}

fn forecast(mut cells: BTreeMap<usize, BTreeSet<InputKey>>, depth: usize) -> Result<Vec<Plan>> {
    let mut plans = Vec::with_capacity(depth);
    for _ in 0..depth {
        let mut active = false;
        for keys in cells.values_mut() {
            let mut parents = BTreeSet::new();
            for key in &*keys {
                if !parents.insert(ScheduleGeometry::input_parent(key)) {
                    active = true;
                }
            }
            *keys = parents;
        }
        plans.push(Plan {
            active,
            merging_levels_left: 0,
        });
    }
    let mut left = 0usize;
    for step in plans.iter_mut().rev() {
        if step.active {
            left = checked_add(left, 1)?;
        }
        step.merging_levels_left = left;
    }
    Ok(plans)
}

#[derive(Debug, Clone, Copy)]
struct MergeShape {
    parents: usize,
    pairs: usize,
}

fn merge_shape<V>(items: &BTreeMap<InputKey, Item<V>>) -> Result<MergeShape>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut parents = BTreeSet::new();
    let mut pairs = 0usize;
    for key in items.keys() {
        if !parents.insert(ScheduleGeometry::input_parent(key)) {
            pairs = checked_add(pairs, 1)?;
        }
    }
    Ok(MergeShape {
        parents: parents.len(),
        pairs,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Unsplit,
    Single(usize),
    Joint,
}

impl Kind {
    fn priority(self) -> (usize, usize) {
        match self {
            Self::Joint => (0, 0),
            Self::Single(axis) => (1, axis),
            Self::Unsplit => (2, 0),
        }
    }
}

/// A candidate owns both its representation and its retained-error ledger.
struct Candidate<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    regions: Vec<ChildRegion<V>>,
    parameters: usize,
    kind: Kind,
    retained: RetainedWork,
}

/// Actual nonempty items and terms, including earlier selected regions.
#[derive(Clone, Copy, Default)]
struct RetainedWork {
    items: usize,
    terms: usize,
}

impl RetainedWork {
    fn include<V>(
        &mut self,
        items: &BTreeMap<InputKey, Item<V>>,
        options: &MergeRefineOptions<V>,
    ) -> Result<()>
    where
        V: Clone + Hash + Eq + Send + Sync + Debug,
    {
        self.items = checked_add(self.items, items.len())?;
        self.terms = items.values().try_fold(self.terms, |total, item| {
            checked_add(total, item.terms.len())
        })?;
        for (value, maximum, limit) in [
            (self.items, options.max_work_items, "max_work_items"),
            (self.terms, options.max_terms, "max_terms"),
        ] {
            if value > maximum {
                return Err(PartitionedTreeTNError::ResourceLimit {
                    operation: "tolerance merge-refine retained candidate",
                    limit,
                    value,
                });
            }
        }
        Ok(())
    }
}

fn prefers(
    probe_parameters: usize,
    probe: Kind,
    incumbent_parameters: usize,
    incumbent: Kind,
) -> bool {
    (probe_parameters, probe.priority()) < (incumbent_parameters, incumbent.priority())
}

/// Bound key metadata before tensor work; retained budgets are checked afterward.
fn child_limit<V>(context: &ProbeContext<'_, V>, shape: MergeShape) -> Result<(usize, &'static str)>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    if shape.parents == 0 {
        return Err(invalid(
            "a tolerance candidate needs at least one input parent",
        ));
    }
    // Key metadata has its own bounded planning cost. Parent counts are upper
    // bounds before restriction, not actual retained items or terms.
    Ok((context.options.max_work_items, "max_work_items"))
}

/// Immutable inputs and allocation shared by alternative probes of one region.
struct RegionTrial<'a, 'ctx, V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    context: &'a ProbeContext<'ctx, V>,
    region: &'a OutputKey,
    items: &'a BTreeMap<InputKey, Item<V>>,
    wanted: Option<&'a [InputKey]>,
    budget: f64,
    shape: MergeShape,
    max_children: (usize, &'static str),
    retained: RetainedWork,
}

impl<V> RegionTrial<'_, '_, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    fn candidate(
        &self,
        axes: &[usize],
        kind: Kind,
        counters: &mut MergeCounters,
    ) -> Result<Candidate<V>> {
        let context = self.context;
        let region = self.region;
        let items = self.items;
        let wanted = self.wanted;
        let region_budget = self.budget;
        let shape = self.shape;
        let max_children = self.max_children;

        let keys = context
            .geometry
            .child_keys(region, axes, wanted, max_children)?;
        // Empty children found during tensor work never change this fixed allocation.
        let share = if shape.pairs == 0 || keys.is_empty() {
            0.0
        } else {
            finite(region_budget / (keys.len() as f64).sqrt() / shape.pairs as f64)?
        };
        let child_context = ProbeContext { share, ..*context };
        let mut regions = Vec::with_capacity(keys.len());
        let mut parameters = 0usize;
        let mut retained = self.retained;
        for key in keys {
            let projector = context.geometry.output_projector(&key)?;
            let mut incurred = 0.0;
            let mut merged =
                restrict_and_merge(&child_context, items, &projector, counters, &mut incurred)?;
            merged.retain(|_, item| !item.terms.is_empty());
            retained.include(&merged, context.options)?;
            parameters = checked_add(parameters, region_parameters(&merged)?)?;
            regions.push((key, merged, incurred));
        }
        Ok(Candidate {
            regions,
            parameters,
            kind,
            retained,
        })
    }

    fn choose(&self, level_index: usize, counters: &mut MergeCounters) -> Result<Candidate<V>> {
        let context = self.context;
        let region = self.region;

        let mut incumbent = self.candidate(&[], Kind::Unsplit, counters)?;
        let unsplit_parameters = incumbent.parameters;
        let axes = context.geometry.refining_axes(region, level_index);
        let mut passing = Vec::new();
        for &axis in &axes {
            let probe = self.candidate(&[axis], Kind::Single(axis), counters)?;
            if split_pays(unsplit_parameters, probe.parameters) {
                passing.push(axis);
                if prefers(
                    probe.parameters,
                    probe.kind,
                    incumbent.parameters,
                    incumbent.kind,
                ) {
                    incumbent = probe;
                }
            }
            // The losing probe (or replaced incumbent) is released here. Only the
            // incumbent and a single probe own candidate tensors at any instant.
        }
        if passing.len() > 1 {
            let probe = self.candidate(&passing, Kind::Joint, counters)?;
            if split_pays(unsplit_parameters, probe.parameters)
                && prefers(
                    probe.parameters,
                    probe.kind,
                    incumbent.parameters,
                    incumbent.kind,
                )
            {
                incumbent = probe;
            }
        }
        let refined_axes = match incumbent.kind {
            Kind::Unsplit => 0,
            Kind::Single(_) => 1,
            Kind::Joint => passing.len(),
        };
        if refined_axes > 0 {
            counters.refined_regions = checked_add(counters.refined_regions, 1)?;
        }
        counters.stopped_regions =
            checked_add(counters.stopped_regions, axes.len() - refined_axes)?;
        Ok(incumbent)
    }
}

pub(super) fn advance<V>(
    context: &ProbeContext<'_, V>,
    level: Level<V>,
    level_index: usize,
    wanted: Option<&[InputKey]>,
    allowance: f64,
    counters: &mut MergeCounters,
) -> Result<(Level<V>, BTreeMap<OutputKey, f64>)>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let shapes = level
        .values()
        .map(|items| merge_shape(items))
        .collect::<Result<Vec<_>>>()?;
    let total_pairs = shapes
        .iter()
        .try_fold(0usize, |total, shape| checked_add(total, shape.pairs))?;
    let mut next = BTreeMap::new();
    let mut components = BTreeMap::new();
    let mut retained = RetainedWork::default();
    // INVARIANT: shapes and level entries use the same deterministic BTree order.
    for ((region, items), shape) in level.into_iter().zip(shapes) {
        let budget = if total_pairs == 0 {
            0.0
        } else {
            finite(allowance * (shape.pairs as f64 / total_pairs as f64).sqrt())?
        };
        let trial = RegionTrial {
            context,
            region: &region,
            items: &items,
            wanted,
            budget,
            shape,
            max_children: child_limit(context, shape)?,
            retained,
        };
        let chosen = trial.choose(level_index, counters)?;
        retained = chosen.retained;
        for (key, items, incurred) in chosen.regions {
            if incurred > 0.0 {
                components.insert(key.clone(), incurred);
            }
            if items.values().all(|item| item.terms.is_empty()) {
                continue;
            }
            next.insert(key, items);
        }
    }
    Ok((next, components))
}

#[cfg(test)]
mod tests;
