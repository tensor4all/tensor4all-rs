//! Macros for HDF5 operations.

#![allow(unused_macros)]

macro_rules! fail {
    ($err:expr) => (
        return Err(From::from($err))
    );

    ($fmt:expr, $($arg:tt)*) => (
        fail!(format!($fmt, $($arg)*))
    );
}

macro_rules! try_ref_clone {
    ($expr:expr) => {
        match $expr {
            Ok(ref val) => val,
            Err(ref err) => return Err(From::from(err.clone())),
        }
    };
}

macro_rules! ensure {
    ($expr:expr, $err:expr) => (
        if !($expr) {
            fail!($err);
        }
    );
    ($expr: expr, $fmt:expr, $($arg:tt)*) => (
        if !($expr) {
            fail!(format!($fmt, $($arg)*));
        }
    );
}

/// Run code containing HDF5 calls in a closure synchronized by a global reentrant mutex.
#[macro_export]
#[doc(hidden)]
macro_rules! h5lock {
    ($expr:expr) => {{
        #[allow(clippy::redundant_closure)]
        #[allow(unused_unsafe)]
        unsafe {
            $crate::sync::sync(|| $expr)
        }
    }};
}

/// Convert result of an HDF5 call to `Result` (guarded by a global reentrant mutex).
#[macro_export]
#[doc(hidden)]
macro_rules! h5call {
    ($expr:expr) => {
        $crate::h5lock!($crate::error::h5check($expr))
    };
}

/// `h5try!(..)` is a convenience shortcut for `try!(h5call!(..))`.
#[macro_export]
#[doc(hidden)]
macro_rules! h5try {
    ($expr:expr) => {
        match $crate::h5call!($expr) {
            Ok(value) => value,
            Err(err) => return Err(From::from(err)),
        }
    };
}
