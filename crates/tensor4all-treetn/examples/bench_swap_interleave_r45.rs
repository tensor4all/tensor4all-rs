use std::collections::HashMap;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
use tensor4all_treetn::{SwapOptions, TreeTN};

fn run(r: usize) {
    let n = 2 * r;
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    let x_inds: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let y_inds: Vec<DynIndex> = (0..r).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..n - 1).map(|_| DynIndex::new_dyn(1)).collect();
    for i in 0..n {
        let site = if i < r { x_inds[i].clone() } else { y_inds[i - r].clone() };
        let mut indices = Vec::new();
        if i > 0 { indices.push(bonds[i - 1].clone()); }
        indices.push(site);
        if i < n - 1 { indices.push(bonds[i].clone()); }
        let size: usize = indices.iter().map(|idx| idx.dim()).product();
        let t = TensorDynLen::from_dense_data(indices, vec![1.0; size]);
        tn.add_tensor(i.to_string(), t).unwrap();
    }
    for i in 0..n - 1 {
        let ni = tn.node_index(&i.to_string()).unwrap();
        let nj = tn.node_index(&(i + 1).to_string()).unwrap();
        tn.connect(ni, &bonds[i], nj, &bonds[i]).unwrap();
    }
    let mut target = HashMap::new();
    for k in 0..r {
        target.insert(x_inds[k].id().to_owned(), (2 * k).to_string());
        target.insert(y_inds[k].id().to_owned(), (2 * k + 1).to_string());
    }
    let start = std::time::Instant::now();
    tn.swap_site_indices(&target, &SwapOptions::default()).unwrap();
    let elapsed = start.elapsed();
    eprintln!("R={r:3} ({n:3} nodes): {elapsed:.3?}");
}

fn main() {
    for r in [5, 10, 15, 20, 25, 30] {
        run(r);
    }
}
