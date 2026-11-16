use tensor4all_core::ColMajorArray;

use crate::error::{invalid_argument, Result};

pub(crate) fn core_from_fn<T>(
    dims: [usize; 3],
    mut value: impl FnMut([usize; 3]) -> T,
) -> Result<ColMajorArray<T>> {
    let mut data = core_buffer(dims)?;
    for right in 0..dims[2] {
        for site in 0..dims[1] {
            for left in 0..dims[0] {
                data.push(value([left, site, right]));
            }
        }
    }
    Ok(ColMajorArray::new(data, dims.to_vec())?)
}

pub(crate) fn core_buffer<T>(dims: [usize; 3]) -> Result<Vec<T>> {
    let len = dims
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| invalid_argument(format!("core shape product overflows usize: {dims:?}")))?;
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|err| invalid_argument(format!("cannot allocate core shape {dims:?}: {err}")))?;
    Ok(data)
}

// INVARIANT: core construction and TreeTN readout validate the rank and size.
// Callers use coordinates from those dimensions or from bounded subblocks.
fn offset(dims: &[usize], left: usize, site: usize, right: usize) -> usize {
    left + dims[0] * (site + dims[1] * right)
}

pub(crate) fn core_at<T: Copy>(
    core: &ColMajorArray<T>,
    left: usize,
    site: usize,
    right: usize,
) -> T {
    core.data()[offset(core.shape(), left, site, right)]
}

pub(crate) fn set_core<T>(
    core: &mut ColMajorArray<T>,
    left: usize,
    site: usize,
    right: usize,
    value: T,
) {
    let index = offset(core.shape(), left, site, right);
    core.data_mut()[index] = value;
}
