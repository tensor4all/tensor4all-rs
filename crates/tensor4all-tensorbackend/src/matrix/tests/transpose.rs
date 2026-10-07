use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn check_layout<T: Clone + Zero>(
    rows: usize,
    cols: usize,
    data: Vec<T>,
    bits: impl Fn(&T) -> [u64; 2],
) {
    let input = Matrix::from_col_major_vec(rows, cols, data);
    let output = transpose(&input);
    assert_eq!((output.nrows(), output.ncols()), (cols, rows));
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                bits(&output.as_col_major_slice()[j + cols * i]),
                bits(&input.as_col_major_slice()[i + rows * j])
            );
        }
    }
    let twice = transpose(&output);
    assert_eq!(
        input
            .as_col_major_slice()
            .iter()
            .map(&bits)
            .collect::<Vec<_>>(),
        twice
            .as_col_major_slice()
            .iter()
            .map(&bits)
            .collect::<Vec<_>>()
    );
}

#[test]
fn transpose_preserves_bits_at_dispatch_and_partial_tile_boundaries() {
    let values64 = [
        0.0_f64,
        -0.0,
        1.5,
        -2.25,
        f64::INFINITY,
        f64::from_bits(0x7ff8000000000738),
    ];
    let values32 = [
        0.0_f32,
        -0.0,
        1.5,
        -2.25,
        f32::INFINITY,
        f32::from_bits(0x7fc00738),
    ];
    for (rows, cols) in [
        (0, 0),
        (0, 7),
        (7, 0),
        (1, 8193),
        (8193, 1),
        (63, 65),
        (64, 64),
        (65, 65),
        (16, 257),
        (31, 133),
        (128, 129),
        (129, 128),
        (129, 145),
        (4096, 32),
        (4097, 32),
    ] {
        let len = rows * cols;
        check_layout(
            rows,
            cols,
            (0..len).map(|i| values64[i % values64.len()]).collect(),
            |v| [v.to_bits(), 0],
        );
        check_layout(
            rows,
            cols,
            (0..len).map(|i| values32[i % values32.len()]).collect(),
            |v| [v.to_bits() as u64, 0],
        );
        check_layout(
            rows,
            cols,
            (0..len)
                .map(|i| Complex64::new(values64[i % 6], values64[(i + 1) % 6]))
                .collect(),
            |v| [v.re.to_bits(), v.im.to_bits()],
        );
        check_layout(
            rows,
            cols,
            (0..len)
                .map(|i| Complex32::new(values32[i % 6], values32[(i + 1) % 6]))
                .collect(),
            |v| [v.re.to_bits() as u64, v.im.to_bits() as u64],
        );
    }
}

#[derive(Debug)]
struct PanicClone {
    value: usize,
    clones: Arc<AtomicUsize>,
    panic_at: usize,
}

impl Clone for PanicClone {
    fn clone(&self) -> Self {
        assert_ne!(
            self.clones.fetch_add(1, Ordering::Relaxed),
            self.panic_at,
            "intentional clone failure"
        );
        Self {
            value: self.value,
            clones: self.clones.clone(),
            panic_at: self.panic_at,
        }
    }
}

impl std::ops::Add for PanicClone {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        self
    }
}

impl Zero for PanicClone {
    fn zero() -> Self {
        Self {
            value: 0,
            clones: Arc::new(AtomicUsize::new(0)),
            panic_at: usize::MAX,
        }
    }
    fn is_zero(&self) -> bool {
        self.value == 0
    }
}

#[test]
fn blocked_transpose_drops_only_initialized_clones_on_unwind_and_success() {
    for (rows, cols) in [(128, 129), (129, 128), (1, 8193), (513, 257)] {
        let len = rows * cols;
        for panic_at in [0_usize, 17, 256, len - 1, usize::MAX] {
            let clones = Arc::new(AtomicUsize::new(0));
            let input = Matrix::from_col_major_vec(
                rows,
                cols,
                (0..len)
                    .map(|value| PanicClone {
                        value,
                        clones: clones.clone(),
                        panic_at,
                    })
                    .collect(),
            );
            let result = std::panic::catch_unwind(|| transpose(&input));
            if panic_at == usize::MAX {
                let output = result.unwrap();
                assert_eq!(Arc::strong_count(&clones), 2 * len + 1);
                for i in 0..rows {
                    for j in 0..cols {
                        assert_eq!(output[[j, i]].value, input[[i, j]].value);
                    }
                }
                drop(output);
            } else {
                assert!(result.is_err());
            }
            assert_eq!(
                Arc::strong_count(&clones),
                len + 1,
                "no initialized clone may leak or be dropped twice"
            );
            drop(input);
            assert_eq!(Arc::strong_count(&clones), 1);
        }
    }
}

#[test]
fn blocked_transpose_supports_zero_sized_values() {
    #[derive(Clone, Debug)]
    struct Z;
    impl std::ops::Add for Z {
        type Output = Self;
        fn add(self, _: Self) -> Self {
            Z
        }
    }
    impl Zero for Z {
        fn zero() -> Self {
            Z
        }
        fn is_zero(&self) -> bool {
            true
        }
    }
    let matrix = Matrix::from_col_major_vec(128, 129, vec![Z; 128 * 129]);
    let output = transpose(&matrix);
    assert_eq!(
        (
            output.nrows(),
            output.ncols(),
            output.as_col_major_slice().len()
        ),
        (129, 128, 128 * 129)
    );
}

#[test]
fn blocked_transpose_preserves_zero_sized_destructors() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone)]
    struct Z;
    impl Drop for Z {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }
    impl std::ops::Add for Z {
        type Output = Self;
        fn add(self, _: Self) -> Self {
            Z
        }
    }
    impl Zero for Z {
        fn zero() -> Self {
            Z
        }
        fn is_zero(&self) -> bool {
            true
        }
    }
    let len = 128 * 129;
    let input = Matrix::from_col_major_vec(128, 129, (0..len).map(|_| Z).collect());
    let before = DROPS.load(Ordering::Relaxed);
    let output = transpose(&input);
    assert_eq!(DROPS.load(Ordering::Relaxed), before);
    drop(output);
    assert_eq!(DROPS.load(Ordering::Relaxed), before + len);
    drop(input);
    assert_eq!(DROPS.load(Ordering::Relaxed), before + 2 * len);
}
