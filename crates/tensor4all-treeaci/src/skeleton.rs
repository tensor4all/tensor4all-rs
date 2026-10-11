use tensor4all_tensorbackend::{solve_matrix_owned, Matrix};

use crate::{
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{PivotPairs, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciScalar,
};

#[derive(Clone, Debug)]
pub(crate) struct SkeletonTensors<T> {
    pub(crate) node: Vec<Vec<T>>,
    pub(crate) node_shape: Vec<Vec<usize>>,
    pub(crate) gauge: Vec<Matrix<T>>,
}

pub(crate) fn skeleton_tensors<T, V, O>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    pivots: &PivotPairs,
    oracle: &mut O,
) -> Result<SkeletonTensors<T>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    O: FnMut(&[usize]) -> Result<T>,
{
    let edge_count = problem.directed_edges.len() / 2;
    if pivots.per_edge.len() != edge_count {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton pivot count differs from prepared edge count",
        });
    }

    let mut node = Vec::with_capacity(problem.node_order.len());
    let mut node_shape = Vec::with_capacity(problem.node_order.len());
    for node_position in 0..problem.node_order.len() {
        let incidents = incident_edges(problem, node_position)?;
        let mut shape = vec![problem.physical[node_position].local_dim];
        for &(edge_number, incoming) in &incidents {
            shape.push(oriented_pivot_ids(pivots, edge_number, incoming)?.len());
        }
        let element_count = checked_product(&shape)?;
        let mut values = vec![T::zero(); element_count];
        for (flat, value) in values.iter_mut().enumerate() {
            let coordinates = decode_mixed_radix(flat, &shape)?;
            let mut incoming_samples = Vec::with_capacity(incidents.len());
            for (axis, &(edge_number, incoming)) in incidents.iter().enumerate() {
                let ids = oriented_pivot_ids(pivots, edge_number, incoming)?;
                incoming_samples.push((incoming, ids[coordinates[axis + 1]]));
            }
            let point = materialize_node_point(
                problem,
                arena,
                node_position,
                coordinates[0],
                &incoming_samples,
            )?;
            *value = oracle(&point)?;
        }
        node.push(values);
        node_shape.push(shape);
    }

    let mut gauge = Vec::with_capacity(edge_count);
    for edge_number in 0..edge_count {
        let pairs = pivots
            .per_edge
            .get(edge_number)
            .ok_or(TreeAciError::InternalInvariant {
                message: "skeleton edge has no pivot pair list",
            })?;
        let rank = pairs.len();
        if rank == 0 {
            return Err(TreeAciError::InternalInvariant {
                message: "skeleton cannot invert an empty pivot block",
            });
        }
        let forward = 2 * edge_number;
        let mut cross = Matrix::zeros(rank, rank);
        for (row, &(left, _)) in pairs.iter().enumerate() {
            for (column, &(_, right)) in pairs.iter().enumerate() {
                let point = arena.materialize_global_point(problem, forward, left, right)?;
                cross[[row, column]] = oracle(&point)?;
            }
        }
        let mut identity = Matrix::zeros(rank, rank);
        for diagonal in 0..rank {
            identity[[diagonal, diagonal]] = T::one();
        }
        let inverse =
            solve_matrix_owned(cross, identity).map_err(|error| TreeAciError::Numerical {
                message: format!("skeleton pivot block solve failed: {error}"),
            })?;
        gauge.push(inverse);
    }

    Ok(SkeletonTensors {
        node,
        node_shape,
        gauge,
    })
}

pub(crate) fn skeleton_evaluate<T, V>(
    tensors: &SkeletonTensors<T>,
    problem: &PreparedTreeProblem<V>,
    sigma: &[usize],
) -> Result<T>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    if sigma.len() != problem.node_order.len() || tensors.node.len() != sigma.len() {
        return Err(TreeAciError::PointLengthMismatch {
            expected: problem.node_order.len(),
            actual: sigma.len(),
        });
    }
    for (node, (&coordinate, physical)) in sigma.iter().zip(&problem.physical).enumerate() {
        if coordinate >= physical.local_dim {
            return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
                node,
                coordinate,
                local_dim: physical.local_dim,
            });
        }
    }
    if tensors.gauge.len() * 2 != problem.directed_edges.len() {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton gauge count differs from prepared edge count",
        });
    }

    // Independent reference contraction of the fixed-point skeleton. All
    // physical coordinates have already been selected; the resulting network
    // has only a scalar output. This avoids explicitly summing rank^(2E)
    // products whose cancellation amplifies round-off in the inverse gauges.
    use tensor4all_core::{DynIndex, IdxTensor, IdxTensorError};
    let tensor_error = |error: IdxTensorError| TreeAciError::Numerical {
        message: error.to_string(),
    };
    use tensor4all_treetn::TreeTN;
    let forward_bonds = tensors
        .gauge
        .iter()
        .map(|g| DynIndex::new_dyn(g.ncols()))
        .collect::<Vec<_>>();
    let reverse_bonds = tensors
        .gauge
        .iter()
        .map(|g| DynIndex::new_dyn(g.nrows()))
        .collect::<Vec<_>>();
    let mut cores = Vec::with_capacity(sigma.len() + tensors.gauge.len());
    for (node_position, &coordinate) in sigma.iter().enumerate() {
        let shape =
            tensors
                .node_shape
                .get(node_position)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "skeleton node shape is missing",
                })?;
        let incidents = incident_edges(problem, node_position)?;
        if shape.len() != incidents.len() + 1
            || shape[0] != problem.physical[node_position].local_dim
        {
            return Err(TreeAciError::InternalInvariant {
                message: "skeleton node shape has the wrong incident-edge count",
            });
        }
        let values = tensors
            .node
            .get(node_position)
            .ok_or(TreeAciError::InternalInvariant {
                message: "skeleton node tensor is missing",
            })?;
        let count = checked_product(&shape[1..])?;
        let mut sliced = Vec::with_capacity(count);
        for bond_assignment in 0..count {
            let index = bond_assignment
                .checked_mul(shape[0])
                .and_then(|x| x.checked_add(coordinate))
                .ok_or(TreeAciError::SizeOverflow {
                    context: "skeleton node offset",
                })?;
            sliced.push(*values.get(index).ok_or(TreeAciError::InternalInvariant {
                message: "skeleton node offset exceeds tensor storage",
            })?);
        }
        let mut indices = Vec::with_capacity(incidents.len());
        for (axis, &(edge, _)) in incidents.iter().enumerate() {
            let source = &problem.directed_edges[2 * edge].from;
            let bond = if problem.node_order[node_position] == *source {
                &reverse_bonds[edge]
            } else {
                &forward_bonds[edge]
            };
            if shape[axis + 1] != tensors.gauge[edge].nrows()
                || tensors.gauge[edge].nrows() != tensors.gauge[edge].ncols()
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "skeleton bond state exceeds node axis dimension",
                });
            }
            indices.push(bond.clone());
        }
        cores.push(IdxTensor::from_dense(indices, sliced).map_err(tensor_error)?);
    }
    for (edge, gauge) in tensors.gauge.iter().enumerate() {
        cores.push(
            IdxTensor::from_dense(
                vec![reverse_bonds[edge].clone(), forward_bonds[edge].clone()],
                gauge.as_col_major_slice().to_vec(),
            )
            .map_err(tensor_error)?,
        );
    }
    let n = cores.len();
    let network = TreeTN::from_tensors(cores, (0..n).collect::<Vec<_>>())?;
    let values = network.to_dense()?.to_vec::<T>().map_err(tensor_error)?;
    values
        .first()
        .copied()
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton contraction has no scalar value",
        })
}

fn incident_edges<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    node_position: usize,
) -> Result<Vec<(usize, DirectedEdgeId)>> {
    let node = problem
        .node_order
        .get(node_position)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton references an unknown node position",
        })?;
    let mut result = Vec::new();
    for edge_number in 0..problem.directed_edges.len() / 2 {
        let forward = &problem.directed_edges[2 * edge_number];
        if &forward.from == node {
            result.push((edge_number, forward.reverse));
        } else if &forward.to == node {
            result.push((edge_number, forward.id));
        }
    }
    Ok(result)
}

fn oriented_pivot_ids(
    pivots: &PivotPairs,
    edge_number: usize,
    directed: DirectedEdgeId,
) -> Result<Vec<SampleId>> {
    let forward = 2 * edge_number;
    if directed == forward {
        Ok(pivots.forward_ids(edge_number))
    } else if directed == forward + 1 {
        Ok(pivots.reverse_ids(edge_number))
    } else {
        Err(TreeAciError::InternalInvariant {
            message: "skeleton directed edge is not an orientation of its edge pair",
        })
    }
}

fn materialize_node_point<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    node_position: usize,
    local_coordinate: usize,
    incoming: &[(DirectedEdgeId, SampleId)],
) -> Result<Vec<usize>> {
    let physical = problem
        .physical
        .get(node_position)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton node has no physical plan",
        })?;
    if local_coordinate >= physical.local_dim {
        return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
            node: node_position,
            coordinate: local_coordinate,
            local_dim: physical.local_dim,
        });
    }
    let mut point = vec![0; problem.node_order.len()];
    let mut visited = vec![false; problem.node_order.len()];
    point[node_position] = local_coordinate;
    visited[node_position] = true;
    for &(directed, sample) in incoming {
        write_component(problem, arena, directed, sample, &mut point, &mut visited)?;
    }
    if visited.iter().any(|visited| !visited) {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton node samples do not cover the full tree",
        });
    }
    Ok(point)
}

fn write_component<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    directed: DirectedEdgeId,
    sample: SampleId,
    point: &mut [usize],
    visited: &mut [bool],
) -> Result<()> {
    let edge = problem
        .directed_edges
        .get(directed)
        .ok_or(TreeAciError::InternalInvariant {
            message: "skeleton component references an unknown directed edge",
        })?;
    let node =
        problem
            .node_positions
            .get(&edge.from)
            .copied()
            .ok_or(TreeAciError::InternalInvariant {
                message: "skeleton component source has no node position",
            })?;
    if visited[node] {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton component samples overlap",
        });
    }
    let record = arena.record(directed, sample)?;
    visited[node] = true;
    point[node] = record.local_coordinate;
    for &(incoming, child_sample) in &record.incoming {
        write_component(problem, arena, incoming, child_sample, point, visited)?;
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(TreeAciError::SizeOverflow {
                context: "skeleton tensor elements",
            })
    })
}

fn decode_mixed_radix(mut flat: usize, shape: &[usize]) -> Result<Vec<usize>> {
    let mut coordinates = Vec::with_capacity(shape.len());
    for &dimension in shape {
        if dimension == 0 {
            return Err(TreeAciError::InternalInvariant {
                message: "skeleton tensor has a zero-sized axis",
            });
        }
        coordinates.push(flat % dimension);
        flat /= dimension;
    }
    if flat != 0 {
        return Err(TreeAciError::InternalInvariant {
            message: "skeleton mixed-radix decode overflowed its shape",
        });
    }
    Ok(coordinates)
}

#[cfg(test)]
mod tests;
