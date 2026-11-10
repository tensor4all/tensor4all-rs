//! The numerical route of one TreeACI run.
//!
//! Every numerical step that leaves this crate's own loops goes through
//! [`Exec`]: matrix products, the pivoted LU, the construction and readback of
//! tensors, the evaluation of networks at points, and the final
//! canonicalization. [`Exec::Global`] is the compatibility route of the plain
//! entry points, unchanged; [`Exec::Session`] runs every step on a caller's
//! explicit session and publishes tensors as compact host storage, so a run
//! enters no process-global or thread-local runtime.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::{
    matrix_luci_factors_from_matrix_owned, matrix_luci_factors_from_matrix_owned_on_session,
    permute_indices_on_session, ColMajorArrayRef, DynIndex, IdxTensor, IdxTensorError,
    MatrixLuciFactors, RrLUOptions,
};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::{
    einsum_native_tensor_reads, mat_mul, mat_mul_in, mat_mul_owned, Matrix, MatrixMulError,
    NativeTensorReadInput,
};
use tensor4all_treetn::{
    CanonicalizationOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator, TreeTNOperationError,
};

use crate::TreeAciScalar;

/// Where the numerical steps of a run execute.
pub(crate) enum Exec<'a, 's> {
    /// The process-global route of the plain entry points.
    Global,
    /// A caller's explicit session, held for the whole run.
    Session(&'a mut Session<'s>),
}

impl<'s> Exec<'_, 's> {
    /// `a * b` with borrowed operands.
    pub(crate) fn mat_mul<T: TreeAciScalar>(
        &mut self,
        a: &Matrix<T>,
        b: &Matrix<T>,
    ) -> Result<Matrix<T>, MatrixMulError> {
        match self {
            Self::Global => mat_mul(a, b),
            Self::Session(session) => mat_mul_in(session.engine(), a, b),
        }
    }

    /// `a * b`, handing the operand buffers over where the route can use them.
    pub(crate) fn mat_mul_owned<T: TreeAciScalar>(
        &mut self,
        a: Matrix<T>,
        b: Matrix<T>,
    ) -> Result<Matrix<T>, MatrixMulError> {
        match self {
            Self::Global => mat_mul_owned(a, b),
            Self::Session(session) => mat_mul_in(session.engine(), &a, &b),
        }
    }

    /// The dense column-major values of one einsum over read-only native
    /// operands, each with its labels.
    pub(crate) fn einsum_reads<T: TreeAciScalar>(
        &mut self,
        operands: &[(&NativeTensorReadInput<'_>, &[usize])],
        output_ids: &[usize],
    ) -> Result<Vec<T>, String> {
        let contracted = match self {
            Self::Global => {
                einsum_native_tensor_reads(operands, output_ids).map_err(|error| error.to_string())
            }
            Self::Session(session) => {
                let reads: Vec<_> = operands.iter().map(|(read, _)| read.as_read()).collect();
                let ids: Vec<&[usize]> = operands.iter().map(|(_, ids)| *ids).collect();
                session
                    .einsum_reads(&reads, &ids, output_ids)
                    .map_err(|error| error.to_string())
            }
        }?;
        T::into_dense_values_from_native_col_major(contracted).map_err(|error| error.to_string())
    }

    /// The pivoted-LU cross-interpolation factors of `matrix`.
    pub(crate) fn luci_factors<T: TreeAciScalar>(
        &mut self,
        matrix: Matrix<T>,
        options: RrLUOptions,
    ) -> crate::Result<MatrixLuciFactors<T>> {
        match self {
            Self::Global => matrix_luci_factors_from_matrix_owned(matrix, Some(options)),
            Self::Session(session) => {
                matrix_luci_factors_from_matrix_owned_on_session(matrix, Some(options), session)
            }
        }
        .map_err(|error| crate::TreeAciError::Numerical {
            message: error.to_string(),
        })
    }

    /// A dense tensor: adopted by the process-global runtime on the global
    /// route, compact host storage on a session.
    pub(crate) fn tensor<T: TreeAciScalar>(
        &self,
        indices: Vec<DynIndex>,
        values: Vec<T>,
    ) -> Result<IdxTensor, IdxTensorError> {
        match self {
            Self::Global => IdxTensor::from_dense(indices, values),
            Self::Session(_) => IdxTensor::from_dense_compact(indices, values),
        }
    }

    /// The dense column-major values of `tensor`, read without entering a
    /// runtime on a session.
    pub(crate) fn values<T: TreeAciScalar>(
        &self,
        tensor: &IdxTensor,
    ) -> Result<Vec<T>, IdxTensorError> {
        match self {
            Self::Global => tensor.to_vec::<T>(),
            Self::Session(_) => tensor.dense_host_values::<T>(),
        }
    }

    /// `tensor` with its axes in the order of `indices`.
    pub(crate) fn permute(
        &mut self,
        tensor: &IdxTensor,
        indices: &[DynIndex],
    ) -> Result<IdxTensor, IdxTensorError> {
        match self {
            Self::Global => tensor.permute_indices(indices),
            Self::Session(session) => permute_indices_on_session(session, tensor, indices),
        }
    }

    /// Evaluate a network at a batch of points.
    pub(crate) fn evaluate<T, V>(
        &mut self,
        evaluator: &mut TreeTNCachedEvaluator<'_, V>,
        values: ColMajorArrayRef<'_, usize>,
        hint: EvaluationHint<V>,
    ) -> Result<Vec<T>, TreeTNOperationError>
    where
        T: TreeAciScalar,
        V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
    {
        match self {
            Self::Global => evaluator.evaluate_batched_typed::<T>(values, hint),
            Self::Session(session) => {
                evaluator.evaluate_batched_typed_on_session::<T>(session, values, hint)
            }
        }
    }

    /// Canonicalize `tree` in place toward `region`.
    pub(crate) fn canonicalize_mut<V>(
        &mut self,
        tree: &mut TreeTN<IdxTensor, V>,
        region: Vec<V>,
        options: CanonicalizationOptions,
    ) -> Result<(), TreeTNOperationError>
    where
        V: Clone + Eq + Hash + Ord + Debug + Send + Sync,
    {
        match self {
            Self::Global => tree.canonicalize_mut(region, options),
            Self::Session(session) => tree.canonicalize_mut_on_session(region, options, session),
        }
    }
}
