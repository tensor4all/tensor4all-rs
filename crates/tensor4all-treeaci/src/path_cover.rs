//! Deterministic continuous minimum-retracing walks for tree sweeps.

use std::collections::HashSet;
use std::sync::Arc;

use thiserror::Error;

use crate::TreeAciTraversalStrategy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OrientedEdgeStep {
    pub(crate) edge: usize,
    pub(crate) from: usize,
    pub(crate) to: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OrientedPath {
    pub(crate) steps: Vec<OrientedEdgeStep>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PathPhase {
    pub(crate) paths: Vec<OrientedPath>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SweepPlan {
    pub(crate) start: usize,
    // Immutable run-scoped plans can be retained while execution mutates the
    // state without copying every path and edge step on each directional pass.
    pub(crate) forward: Arc<[PathPhase]>,
    pub(crate) reverse: Arc<[PathPhase]>,
    /// Sorted `(neighbor, edge)` lists per node, for the re-commit excursions
    /// of [`Self::reverse_with_recommits`].
    adjacency: Arc<[Vec<(usize, usize)>]>,
}

#[derive(Debug, Error)]
pub(crate) enum PathPlanError {
    #[error("a tree scheduler requires at least one node")]
    Empty,
    #[error("edge {edge} has an endpoint outside 0..{node_count}")]
    InvalidEndpoint { edge: usize, node_count: usize },
    #[error("edge {edge} is a self-loop")]
    SelfLoop { edge: usize },
    #[error("the supplied topology is not a connected acyclic tree")]
    NotTree,
    #[error("tree-walk construction violated its connectivity invariant")]
    InvalidCover,
    #[error("invalid sweep plan: {0}")]
    InvalidPlan(&'static str),
}

impl SweepPlan {
    pub(crate) fn for_strategy(
        strategy: TreeAciTraversalStrategy,
        node_count: usize,
        edges: &[(usize, usize)],
        requested_start: Option<usize>,
    ) -> Result<Self, PathPlanError> {
        match strategy {
            TreeAciTraversalStrategy::MinimumRetracingWalk => {
                Self::from_minimum_retracing_walk(node_count, edges, requested_start)
            }
        }
    }

    fn from_minimum_retracing_walk(
        node_count: usize,
        edges: &[(usize, usize)],
        requested_start: Option<usize>,
    ) -> Result<Self, PathPlanError> {
        validate_tree(node_count, edges)?;
        if edges.is_empty() {
            return Ok(Self {
                start: 0,
                forward: Arc::from([]),
                reverse: Arc::from([]),
                adjacency: Arc::from([]),
            });
        }
        let adjacency = adjacency(node_count, edges);
        let start = match requested_start {
            Some(start) if start < node_count => start,
            Some(_) => {
                return Err(PathPlanError::InvalidEndpoint {
                    edge: 0,
                    node_count,
                })
            }
            None => farthest_node(0, &adjacency).0,
        };
        let end = farthest_node(start, &adjacency).0;
        let spine = path_nodes(start, end, &adjacency)?;
        let mut spine_next = vec![None; node_count];
        for pair in spine.windows(2) {
            spine_next[pair[0]] = Some(pair[1]);
        }
        let steps = continuous_walk(start, &adjacency, &spine_next)?;
        let forward = vec![PathPhase {
            paths: vec![OrientedPath { steps }],
        }];
        let reverse = vec![PathPhase {
            paths: vec![OrientedPath {
                steps: reverse_spine(&spine, &adjacency)?,
            }],
        }];
        let plan = Self {
            start,
            forward: forward.into(),
            reverse: reverse.into(),
            adjacency: adjacency.into(),
        };
        plan.validate(node_count, edges)?;
        Ok(plan)
    }

    fn validate(&self, node_count: usize, edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
        validate_tree(node_count, edges)?;
        validate_pass(&self.forward, edges)?;
        validate_pass(&self.reverse, edges)?;
        validate_round(&self.forward, &self.reverse, edges)?;
        Ok(())
    }

    /// The return pass, with a re-commit excursion for every off-path bond
    /// whose candidate set must be re-committed before a return commit reads
    /// it.
    ///
    /// A return step `a -> b` reads, as rows, the candidate set of every
    /// incoming bond of `a`: the step that reached `a` (just re-committed) and
    /// each off-path bond `c -> a`. Before that step, every off-path `c -> a`
    /// for which `needs_recommit` holds is re-committed by the retracing
    /// excursion `a -> c`, then the same excursion below `c` for each such
    /// child bond of `c`, then `c -> a`. Children come before their parent, as
    /// in the forward walk, so each upward commit reads child bonds that were
    /// already re-committed. Each excursion starts and ends at `a`, the current
    /// canonical centre, so the walk stays continuous. The last node of the
    /// path receives no excursion: no commit of the pass reads its off-path
    /// bonds as rows.
    ///
    /// When `needs_recommit` holds for no off-path bond, the stored return
    /// pass is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns the first error of `needs_recommit`.
    pub(crate) fn reverse_with_recommits<E>(
        &self,
        mut needs_recommit: impl FnMut(OrientedEdgeStep) -> Result<bool, E>,
    ) -> Result<Arc<[PathPhase]>, E> {
        let mut extended = Vec::with_capacity(self.reverse.len());
        let mut changed = false;
        for phase in self.reverse.iter() {
            let mut paths = Vec::with_capacity(phase.paths.len());
            for path in &phase.paths {
                let steps =
                    with_recommit_excursions(&self.adjacency, &path.steps, &mut needs_recommit)?;
                changed |= steps.len() != path.steps.len();
                paths.push(OrientedPath { steps });
            }
            extended.push(PathPhase { paths });
        }
        Ok(if changed {
            extended.into()
        } else {
            Arc::clone(&self.reverse)
        })
    }
}

fn adjacency(node_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for (edge, &(left, right)) in edges.iter().enumerate() {
        adjacency[left].push((right, edge));
        adjacency[right].push((left, edge));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    adjacency
}

fn farthest_node(start: usize, adjacency: &[Vec<(usize, usize)>]) -> (usize, usize) {
    let mut best = (start, 0usize);
    let mut stack = vec![(start, usize::MAX, 0usize)];
    while let Some((node, parent, distance)) = stack.pop() {
        if distance > best.1 || (distance == best.1 && node < best.0) {
            best = (node, distance);
        }
        for &(neighbor, _) in adjacency[node].iter().rev() {
            if neighbor != parent {
                stack.push((neighbor, node, distance + 1));
            }
        }
    }
    best
}

fn path_nodes(
    start: usize,
    end: usize,
    adjacency: &[Vec<(usize, usize)>],
) -> Result<Vec<usize>, PathPlanError> {
    let mut parent = vec![None; adjacency.len()];
    let mut stack = vec![start];
    parent[start] = Some(start);
    while let Some(node) = stack.pop() {
        if node == end {
            break;
        }
        for &(neighbor, _) in adjacency[node].iter().rev() {
            if parent[neighbor].is_none() {
                parent[neighbor] = Some(node);
                stack.push(neighbor);
            }
        }
    }
    if parent[end].is_none() {
        return Err(PathPlanError::NotTree);
    }
    let mut path = vec![end];
    while *path.last().ok_or(PathPlanError::InvalidCover)? != start {
        let node = *path.last().ok_or(PathPlanError::InvalidCover)?;
        path.push(parent[node].ok_or(PathPlanError::InvalidCover)?);
    }
    path.reverse();
    Ok(path)
}

#[derive(Clone, Copy)]
enum WalkAction {
    Explore { node: usize, parent: Option<usize> },
    Move(OrientedEdgeStep),
}

fn continuous_walk(
    start: usize,
    adjacency: &[Vec<(usize, usize)>],
    spine_next: &[Option<usize>],
) -> Result<Vec<OrientedEdgeStep>, PathPlanError> {
    let mut result = Vec::new();
    let mut stack = vec![WalkAction::Explore {
        node: start,
        parent: None,
    }];
    while let Some(action) = stack.pop() {
        match action {
            WalkAction::Move(step) => result.push(step),
            WalkAction::Explore { node, parent } => {
                let designated = spine_next[node];
                let mut actions = Vec::new();
                for &(neighbor, edge) in &adjacency[node] {
                    if Some(neighbor) == parent || Some(neighbor) == designated {
                        continue;
                    }
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: node,
                        to: neighbor,
                    }));
                    actions.push(WalkAction::Explore {
                        node: neighbor,
                        parent: Some(node),
                    });
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: neighbor,
                        to: node,
                    }));
                }
                if let Some(next) = designated {
                    let edge = edge_between(node, next, adjacency)?;
                    actions.push(WalkAction::Move(OrientedEdgeStep {
                        edge,
                        from: node,
                        to: next,
                    }));
                    actions.push(WalkAction::Explore {
                        node: next,
                        parent: Some(node),
                    });
                }
                stack.extend(actions.into_iter().rev());
            }
        }
    }
    Ok(result)
}

fn edge_between(
    from: usize,
    to: usize,
    adjacency: &[Vec<(usize, usize)>],
) -> Result<usize, PathPlanError> {
    adjacency
        .get(from)
        .and_then(|neighbors| {
            neighbors
                .iter()
                .find_map(|&(neighbor, edge)| (neighbor == to).then_some(edge))
        })
        .ok_or(PathPlanError::InvalidCover)
}

fn reverse_spine(
    spine: &[usize],
    adjacency: &[Vec<(usize, usize)>],
) -> Result<Vec<OrientedEdgeStep>, PathPlanError> {
    spine
        .windows(2)
        .rev()
        .map(|pair| {
            let edge = edge_between(pair[0], pair[1], adjacency)?;
            Ok(OrientedEdgeStep {
                edge,
                from: pair[1],
                to: pair[0],
            })
        })
        .collect()
}

/// `steps` with a re-commit excursion inserted before each step for every
/// off-path bond into the step's source that `needs_recommit` selects.
fn with_recommit_excursions<E>(
    adjacency: &[Vec<(usize, usize)>],
    steps: &[OrientedEdgeStep],
    needs_recommit: &mut impl FnMut(OrientedEdgeStep) -> Result<bool, E>,
) -> Result<Vec<OrientedEdgeStep>, E> {
    let on_path: HashSet<usize> = steps.iter().flat_map(|step| [step.from, step.to]).collect();
    let mut result = Vec::with_capacity(steps.len());
    for step in steps {
        for &(neighbor, edge) in &adjacency[step.from] {
            if on_path.contains(&neighbor) {
                continue;
            }
            let upward = OrientedEdgeStep {
                edge,
                from: neighbor,
                to: step.from,
            };
            if needs_recommit(upward)? {
                push_recommit_excursion(adjacency, upward, needs_recommit, &mut result)?;
            }
        }
        result.push(*step);
    }
    Ok(result)
}

/// Appends `upward.to -> upward.from`, the excursions below `upward.from`
/// that `needs_recommit` selects (in neighbour order), and `upward` itself.
fn push_recommit_excursion<E>(
    adjacency: &[Vec<(usize, usize)>],
    upward: OrientedEdgeStep,
    needs_recommit: &mut impl FnMut(OrientedEdgeStep) -> Result<bool, E>,
    result: &mut Vec<OrientedEdgeStep>,
) -> Result<(), E> {
    enum Visit {
        Enter(OrientedEdgeStep),
        Leave(OrientedEdgeStep),
    }
    // An explicit stack keeps deep branches off the call stack.
    let mut stack = vec![Visit::Enter(upward)];
    while let Some(visit) = stack.pop() {
        match visit {
            Visit::Enter(upward) => {
                result.push(OrientedEdgeStep {
                    edge: upward.edge,
                    from: upward.to,
                    to: upward.from,
                });
                stack.push(Visit::Leave(upward));
                let mut children = Vec::new();
                for &(neighbor, edge) in &adjacency[upward.from] {
                    if neighbor == upward.to {
                        continue;
                    }
                    let child = OrientedEdgeStep {
                        edge,
                        from: neighbor,
                        to: upward.from,
                    };
                    if needs_recommit(child)? {
                        children.push(child);
                    }
                }
                stack.extend(children.into_iter().rev().map(Visit::Enter));
            }
            Visit::Leave(upward) => result.push(upward),
        }
    }
    Ok(())
}

fn validate_pass(phases: &[PathPhase], edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
    for phase in phases {
        let mut phase_vertices = HashSet::new();
        for path in &phase.paths {
            if path.steps.is_empty() {
                return Err(PathPlanError::InvalidPlan("a path is empty"));
            }
            let mut path_vertices = HashSet::with_capacity(path.steps.len() + 1);
            for (position, step) in path.steps.iter().enumerate() {
                let Some(&(left, right)) = edges.get(step.edge) else {
                    return Err(PathPlanError::InvalidPlan(
                        "a step references an unknown edge",
                    ));
                };
                if !((step.from == left && step.to == right)
                    || (step.from == right && step.to == left))
                {
                    return Err(PathPlanError::InvalidPlan(
                        "a step orientation does not match its edge",
                    ));
                }
                if position > 0 && path.steps[position - 1].to != step.from {
                    return Err(PathPlanError::InvalidPlan(
                        "consecutive path steps are disconnected",
                    ));
                }
                path_vertices.insert(step.from);
                path_vertices.insert(step.to);
            }
            if !phase_vertices.is_disjoint(&path_vertices) {
                return Err(PathPlanError::InvalidPlan(
                    "paths in one phase share a vertex",
                ));
            }
            phase_vertices.extend(path_vertices);
        }
    }
    Ok(())
}

fn validate_round(
    forward: &[PathPhase],
    reverse: &[PathPhase],
    edges: &[(usize, usize)],
) -> Result<(), PathPlanError> {
    let forward_steps = pass_steps(forward);
    let reverse_steps = pass_steps(reverse);
    let first_forward = forward_steps
        .first()
        .ok_or(PathPlanError::InvalidPlan("forward pass is empty"))?;
    let last_forward = forward_steps
        .last()
        .ok_or(PathPlanError::InvalidPlan("forward pass is empty"))?;
    let first_reverse = reverse_steps
        .first()
        .ok_or(PathPlanError::InvalidPlan("reverse pass is empty"))?;
    let last_reverse = reverse_steps
        .last()
        .ok_or(PathPlanError::InvalidPlan("reverse pass is empty"))?;
    if last_forward.to != first_reverse.from {
        return Err(PathPlanError::InvalidPlan(
            "return pass does not start at the forward endpoint",
        ));
    }
    if last_reverse.to != first_forward.from {
        return Err(PathPlanError::InvalidPlan(
            "forward and return passes do not form a closed round",
        ));
    }

    let expected_steps = edges
        .len()
        .checked_mul(2)
        .ok_or(PathPlanError::InvalidPlan("round edge count overflowed"))?;
    let mut directed_steps = HashSet::with_capacity(expected_steps);
    for step in forward_steps.iter().chain(&reverse_steps) {
        if !directed_steps.insert((step.edge, step.from, step.to)) {
            return Err(PathPlanError::InvalidPlan(
                "a directed edge occurs more than once in a round",
            ));
        }
    }
    if directed_steps.len() != expected_steps {
        return Err(PathPlanError::InvalidPlan(
            "a round does not cover every directed edge exactly once",
        ));
    }
    Ok(())
}

fn pass_steps(phases: &[PathPhase]) -> Vec<OrientedEdgeStep> {
    phases
        .iter()
        .flat_map(|phase| &phase.paths)
        .flat_map(|path| &path.steps)
        .copied()
        .collect()
}

fn validate_tree(node_count: usize, edges: &[(usize, usize)]) -> Result<(), PathPlanError> {
    if node_count == 0 {
        return Err(PathPlanError::Empty);
    }
    if edges.len() != node_count - 1 {
        return Err(PathPlanError::NotTree);
    }
    let mut union_find = UnionFind::new(node_count);
    for (edge, &(left, right)) in edges.iter().enumerate() {
        if left >= node_count || right >= node_count {
            return Err(PathPlanError::InvalidEndpoint { edge, node_count });
        }
        if left == right {
            return Err(PathPlanError::SelfLoop { edge });
        }
        if !union_find.union(left, right) {
            return Err(PathPlanError::NotTree);
        }
    }
    let root = union_find.find(0);
    if (1..node_count).any(|node| union_find.find(node) != root) {
        return Err(PathPlanError::NotTree);
    }
    Ok(())
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests;
