//! Explicit and optional process-global tenferro CPU execution contexts.

use std::cell::RefCell;
#[cfg(feature = "global-defaults")]
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "global-defaults")]
use std::sync::Weak;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro::{
    CompiledGraph, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TracedGraph,
};
#[cfg(feature = "global-defaults")]
use tenferro_ad::ContextId;
use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::{BufferPoolStats, CpuEngine};

/// Linux CPU-affinity FFI, kept in one place so every `unsafe` block has one
/// auditable owner.
///
/// The mask is a growable word buffer in the kernel's `sched_*affinity` layout.
/// No fixed `cpu_set_t` is used: a mask larger than glibc's 1024-CPU set is
/// retried with a larger buffer instead of failing, and CPU ids are validated
/// before they index the buffer.
#[cfg(target_os = "linux")]
mod affinity {
    use std::io;

    use super::CpuThreadError;

    // Declared here rather than through `libc`, whose Linux binding for
    // `sched_getaffinity` is not present in every supported version. The
    // signatures use the kernel's raw byte-buffer contract, not `cpu_set_t`.
    unsafe extern "C" {
        fn sched_getaffinity(
            pid: libc::pid_t,
            cpusetsize: libc::size_t,
            cpuset: *mut libc::c_void,
        ) -> libc::c_int;
        fn sched_setaffinity(
            pid: libc::pid_t,
            cpusetsize: libc::size_t,
            cpuset: *const libc::c_void,
        ) -> libc::c_int;
    }

    const BITS_PER_WORD: usize = libc::c_ulong::BITS as usize;
    const INITIAL_WORDS: usize = 1;
    /// 512 KiB of mask on a 64-bit host, far beyond any current CPU count.
    const MAX_WORDS: usize = 1 << 16;

    /// One CPU set in the layout `sched_getaffinity` and `sched_setaffinity`
    /// expect. The words are owned by the `Vec`, so the buffer is word-aligned.
    #[derive(Clone, Debug)]
    pub(super) struct CpuMask {
        words: Vec<libc::c_ulong>,
    }

    impl CpuMask {
        /// The mask holding exactly `cpu`, with at least `min_words` words.
        ///
        /// Callers validate `cpu` against the allowed set first, so the small
        /// allocation is bounded by the current kernel mask. `min_words` keeps
        /// the buffer at the kernel's mask size on hosts where the highest
        /// allowed CPU is in a low word.
        pub(super) fn single(cpu: usize, min_words: usize) -> Self {
            let word = cpu / BITS_PER_WORD;
            let bit = cpu % BITS_PER_WORD;
            let mut words = vec![0; (word + 1).max(min_words)];
            words[word] |= 1 << bit;
            Self { words }
        }

        /// The number of words in this mask's kernel buffer.
        pub(super) fn words_len(&self) -> usize {
            self.words.len()
        }

        /// The CPU ids in this mask, in ascending order.
        pub(super) fn cpus(&self) -> Vec<usize> {
            let mut cpus = Vec::new();
            for (index, &word) in self.words.iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    cpus.push(index * BITS_PER_WORD + bit);
                    bits &= bits - 1;
                }
            }
            cpus
        }

        /// Whether this mask holds exactly the same CPUs as `other`.
        pub(super) fn equals(&self, other: &Self) -> bool {
            let len = self.words.len().max(other.words.len());
            (0..len).all(|index| self.word(index) == other.word(index))
        }

        fn word(&self, index: usize) -> libc::c_ulong {
            self.words.get(index).copied().unwrap_or(0)
        }

        fn as_bytes(&self) -> usize {
            std::mem::size_of_val(self.words.as_slice())
        }

        fn as_ptr(&self) -> *const libc::c_void {
            self.words.as_ptr().cast()
        }
    }

    /// Read the mask of the thread identified by `pid` (`0` is the calling
    /// thread).
    fn get_mask(pid: libc::pid_t) -> io::Result<CpuMask> {
        let mut words = vec![0 as libc::c_ulong; INITIAL_WORDS];
        loop {
            let bytes = std::mem::size_of_val(words.as_slice());
            // SAFETY: `words` is an owned, word-aligned buffer and `bytes` is
            // its exact byte length; the kernel writes at most that many bytes.
            let result = unsafe { sched_getaffinity(pid, bytes, words.as_mut_ptr().cast()) };
            if result == 0 {
                // Keep the successful kernel buffer size: `sched_setaffinity`
                // requires at least that many bytes even when the high words
                // are empty.
                return Ok(CpuMask { words });
            }
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINVAL) {
                return Err(error);
            }
            let next = words.len().checked_mul(2).filter(|&len| len <= MAX_WORDS);
            let Some(next) = next else {
                return Err(io::Error::other(
                    "CPU affinity mask is larger than the supported buffer",
                ));
            };
            words.resize(next, 0);
        }
    }

    /// Replace the mask of the thread identified by `pid` (`0` is the calling
    /// thread) with `mask`.
    fn set_mask(pid: libc::pid_t, mask: &CpuMask) -> io::Result<()> {
        // SAFETY: `mask` owns a word-aligned buffer and `as_bytes` is its exact
        // byte length; the kernel reads at most that many bytes.
        let result = unsafe { sched_setaffinity(pid, mask.as_bytes(), mask.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// The main thread's mask at construction, a snapshot taken with
    /// `sched_getaffinity(getpid())`.
    ///
    /// The process id names the thread-group leader, so a pool built lazily
    /// from a worker of an already-pinned pool does not inherit that worker's
    /// single CPU.
    pub(super) fn main_thread_mask() -> Result<CpuMask, CpuThreadError> {
        get_mask(std::process::id() as libc::pid_t).map_err(|source| CpuThreadError::Affinity {
            operation: "sched_getaffinity",
            source,
        })
    }

    /// Set the calling thread's mask to `mask`.
    pub(super) fn set_current_thread_to(mask: &CpuMask) -> Result<(), CpuThreadError> {
        set_mask(0, mask).map_err(|source| CpuThreadError::Affinity {
            operation: "sched_setaffinity",
            source,
        })
    }

    /// Pin the calling thread to exactly `cpu` and verify the effective mask.
    pub(super) fn pin_current_thread(cpu: usize) -> io::Result<()> {
        let min_words = get_mask(0)?.words_len();
        let expected = CpuMask::single(cpu, min_words);
        set_mask(0, &expected)?;
        let actual = get_mask(0)?;
        if actual.equals(&expected) {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "worker mask is not exactly CPU {cpu}"
            )))
        }
    }

    /// The calling thread's effective CPU ids; used by the affinity tests.
    #[cfg(test)]
    pub(super) fn current_thread_cpus() -> Result<Vec<usize>, CpuThreadError> {
        get_mask(0)
            .map(|mask| mask.cpus())
            .map_err(|source| CpuThreadError::Affinity {
                operation: "sched_getaffinity",
                source,
            })
    }
}

/// Failure reported when building a [`CpuBackend`] pool or applying its
/// affinity policy.
///
/// A pool build and, on Linux, the affinity snapshot are the fallible steps of a
/// [`CpuBackend`]: tenferro itself never builds threads, so the caller's thread
/// policy is the caller's error to
/// carry.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuThreadError;
///
/// let error = CpuThreadError::ZeroThreads;
/// assert!(error.to_string().contains("at least one"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CpuThreadError {
    /// A [`CpuBackend`] was asked for zero workers.
    #[error("a CPU backend needs at least one worker thread")]
    ZeroThreads,
    /// The Rayon pool for the requested worker count could not be built.
    #[error("failed to build the CPU worker pool: {source}")]
    Pool {
        /// Original Rayon diagnostic.
        #[source]
        source: Box<rayon::ThreadPoolBuildError>,
    },
    /// [`CpuBackend::pinned`] was given an empty CPU list.
    #[error("a pinned CPU backend needs at least one CPU")]
    EmptyCpuList,
    /// [`CpuBackend::pinned`] was given the same CPU twice.
    #[error("the pinned CPU list repeats CPU {cpu}")]
    DuplicateCpu {
        /// The repeated CPU id.
        cpu: usize,
    },
    /// [`CpuBackend::pinned`] named a CPU outside the main thread's mask.
    #[error("CPU {cpu} is outside the main thread's allowed CPU set")]
    CpuNotAllowed {
        /// The rejected CPU id.
        cpu: usize,
    },
    /// CPU affinity is only implemented on Linux.
    #[error("CPU affinity is not supported on this platform")]
    UnsupportedPlatform,
    /// An affinity system call failed.
    #[error("{operation} failed: {source}")]
    Affinity {
        /// The system call that failed.
        operation: &'static str,
        /// Original operating-system diagnostic.
        #[source]
        source: std::io::Error,
    },
    /// The short-lived thread that builds a pool could not be started.
    #[error("failed to start the CPU pool construction thread: {source}")]
    ConstructionThread {
        /// Original thread-spawn diagnostic.
        #[source]
        source: std::io::Error,
    },
    /// The CPU pool construction thread panicked before returning the pool.
    #[error("the CPU pool construction thread panicked")]
    ConstructionThreadPanic,
    /// A pinned worker could not be confined to its CPU.
    #[error("failed to pin CPU worker {worker} to its CPU: {source}")]
    WorkerAffinity {
        /// Rayon worker index.
        worker: usize,
        /// Original operating-system or verification diagnostic.
        #[source]
        source: std::io::Error,
    },
    /// A pinned worker terminated before reporting its affinity result.
    #[error("a pinned CPU worker did not report its affinity: {source}")]
    WorkerStartupClosed {
        /// Channel receive failure from the worker startup handshake.
        #[source]
        source: std::sync::mpsc::RecvError,
    },
}

/// Failure reported by [`set_default_cpu_threads`] when the default is fixed.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::AlreadyInitialized;
///
/// let error = AlreadyInitialized;
/// assert_eq!(
///     error.to_string(),
///     "the process-wide default CPU thread count is already initialized"
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the process-wide default CPU thread count is already initialized")]
pub struct AlreadyInitialized;

static DEFAULT_CPU_THREADS: OnceLock<usize> = OnceLock::new();

/// Fix the worker count [`CpuBackend::new`] uses for the rest of the process.
///
/// The value is read once, by the first [`CpuBackend::new`] call; a backend
/// built later keeps whatever policy it was built with. Call this before any
/// other thread can construct a default backend.
///
/// # Errors
///
/// Returns [`AlreadyInitialized`] when the default thread configuration was
/// already fixed, so a change is unsupported: either an earlier call or the
/// first [`CpuBackend::new`] fixed it.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use tensor4all_tensorbackend::{set_default_cpu_threads, AlreadyInitialized};
///
/// // Whichever call sets the default first wins, and every later one is
/// // rejected typed instead of panicking.
/// let _ = set_default_cpu_threads(NonZeroUsize::new(3).unwrap());
/// assert_eq!(
///     set_default_cpu_threads(NonZeroUsize::MAX),
///     Err(AlreadyInitialized)
/// );
/// ```
pub fn set_default_cpu_threads(threads: NonZeroUsize) -> Result<(), AlreadyInitialized> {
    DEFAULT_CPU_THREADS
        .set(threads.get())
        .map_err(|_| AlreadyInitialized)
}

/// Build one Rayon pool with `threads` (or Rayon's own default when `None`).
///
/// On Linux the build runs on a short-lived construction thread whose mask is
/// first set to the main thread's mask snapshot. Rayon resolves its default
/// worker count (`available_parallelism`) before the workers start, so this
/// ordering keeps a pool built from a singly pinned worker from inheriting that
/// worker's single CPU. The workers inherit the construction thread's mask.
///
/// Off Linux the pool is built on the calling thread and no affinity call is
/// made.
fn build_pool(threads: Option<usize>) -> Result<rayon::ThreadPool, CpuThreadError> {
    #[cfg(target_os = "linux")]
    {
        let mask = affinity::main_thread_mask()?;
        std::thread::Builder::new()
            .name("tensor4all-cpu-pool".to_owned())
            .spawn(move || {
                affinity::set_current_thread_to(&mask)?;
                build_pool_here(threads)
            })
            .map_err(|source| CpuThreadError::ConstructionThread { source })?
            .join()
            .map_err(|_| CpuThreadError::ConstructionThreadPanic)?
    }
    #[cfg(not(target_os = "linux"))]
    {
        build_pool_here(threads)
    }
}

/// Build one Rayon pool on the calling thread.
fn build_pool_here(threads: Option<usize>) -> Result<rayon::ThreadPool, CpuThreadError> {
    let mut builder = tenferro_cpu::thread_pool_builder();
    if let Some(threads) = threads {
        builder = builder.num_threads(threads);
    }
    builder.build().map_err(|source| CpuThreadError::Pool {
        source: Box::new(source),
    })
}

/// tensor4all's owned CPU execution handle: one [`CpuEngine`].
///
/// tenferro no longer has a backend session or a `CpuContext`; a caller builds
/// a Rayon pool, hands it to [`CpuEngine::new`], and keeps the engine for as
/// long as it wants the thread policy, the buffer pool, the prepared-plan cache
/// and the contraction scratch to stay warm. This type is that caller-held
/// owner for tensor4all: the implicit frontend's resource owner during the
/// transition to engine-explicit APIs, which will replace it later.
///
/// It is **not** [`Clone`]. Cloning an execution domain was how the previous
/// session API shared one engine between callers; with a caller-held engine the
/// caller owns it outright, and a second execution domain needs a second
/// backend.
///
/// [`CpuBackend::new`] resolves its worker count in this order: the value
/// [`set_default_cpu_threads`] fixed, else Rayon's own default (`RAYON_NUM_THREADS`,
/// see Rayon's documentation), else the CPUs in the main thread's mask. A
/// resolved count of one uses [`CpuEngine::serial`] and owns no pool. The count
/// is fixed by the first [`CpuBackend::new`] of the process, or by the setter.
/// If the pool cannot be built, [`CpuBackend::new`] falls back to
/// [`CpuEngine::serial`] instead of panicking.
///
/// Every pool this type builds on Linux confines its workers to the main
/// thread's allowed CPU set, captured as a snapshot when the pool is built.
/// This applies to the pool's workers only: operations that run inline on the
/// caller (small operations, Rayon-worker entries that use the serial inline
/// engine, and hataori lane threads) are not moved. A cpuset or CPU-hotplug
/// change needs a new backend, not a continuous watcher.
///
/// [`CpuBackend::pinned`] is an opt-in per-worker pinning mode whose benefit is
/// unmeasured. Disjoint pinned backends confine their *pools'* workers; they do
/// not isolate all execution of their engines, because an entry from any Rayon
/// worker runs inline on that worker. A provider team created from a pinned
/// worker inherits that worker's one CPU.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuBackend;
/// use tenferro_cpu::CpuEngine;
///
/// // A serial backend owns no pool and runs every operation on the caller's thread.
/// let backend = CpuBackend::with_threads(1)?;
/// assert_eq!(backend.num_threads(), 1);
/// assert!(backend.engine().pool().is_none());
///
/// // A caller that already owns an engine can hand it in unchanged.
/// let backend = CpuBackend::from(CpuEngine::serial());
/// assert_eq!(backend.num_threads(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuBackend {
    engine: CpuEngine,
}

impl std::fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("engine", &self.engine)
            .finish()
    }
}

impl CpuBackend {
    /// Build a backend with the default thread policy.
    ///
    /// The worker count is [`set_default_cpu_threads`] if it was called; else
    /// Rayon's default (`RAYON_NUM_THREADS`, see Rayon's documentation), which
    /// falls back to the CPUs in the main thread's mask; else one per CPU in
    /// that mask. A worker stack of
    /// [`tenferro_cpu::DEFAULT_WORKER_STACK_BYTES`] is reserved for each worker,
    /// as the CPU providers require. The count is fixed by the first call.
    ///
    /// On Linux the pool is built on a short-lived construction thread whose
    /// mask is first set to the main thread's mask snapshot, so building from a
    /// worker of a pinned pool does not inherit that worker's single CPU. The
    /// pool's workers inherit the snapshot; inline operations of the caller are
    /// not moved.
    ///
    /// The pool build and the Linux affinity snapshot are the fallible steps,
    /// and this constructor does not report them: if either cannot be used,
    /// the backend is [`CpuEngine::serial`] instead, which is correct but uses
    /// no worker threads. Use [`CpuBackend::with_threads`] or
    /// [`CpuBackend::pinned`] when that difference matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert!(backend.num_threads() >= 1);
    /// ```
    pub fn new() -> Self {
        if let Some(&threads) = DEFAULT_CPU_THREADS.get() {
            return Self::with_threads_or_serial(threads);
        }
        match build_pool(None) {
            Ok(pool) => {
                let resolved = pool.current_num_threads();
                let fixed = match DEFAULT_CPU_THREADS.set(resolved) {
                    Ok(()) => resolved,
                    Err(_) => *DEFAULT_CPU_THREADS.get().unwrap_or(&resolved),
                };
                if fixed != resolved {
                    return Self::with_threads_or_serial(fixed);
                }
                if resolved == 1 {
                    Self {
                        engine: CpuEngine::serial(),
                    }
                } else {
                    Self {
                        engine: CpuEngine::new(pool),
                    }
                }
            }
            Err(_) => {
                let _ = DEFAULT_CPU_THREADS.set(1);
                Self {
                    engine: CpuEngine::serial(),
                }
            }
        }
    }

    /// Build a backend with `threads` workers, or serial for one.
    fn with_threads_or_serial(threads: usize) -> Self {
        Self::with_threads(threads).unwrap_or_else(|_| Self {
            engine: CpuEngine::serial(),
        })
    }

    /// Build a backend that may use `threads` workers.
    ///
    /// One worker builds a [`CpuEngine::serial`] engine, which owns no pool, so
    /// it stays usable from inside another pool. Larger counts build a Rayon
    /// pool with [`tenferro_cpu::thread_pool_builder`], so provider code gets the
    /// worker stack it needs; the pool's workers are confined to the main
    /// thread's allowed CPU set snapshot on Linux.
    ///
    /// # Errors
    ///
    /// Returns [`CpuThreadError::ZeroThreads`] for zero,
    /// [`CpuThreadError::Affinity`] when the mask snapshot or its application
    /// fails, [`CpuThreadError::ConstructionThread`] when the construction
    /// thread cannot start, and [`CpuThreadError::Pool`] when Rayon cannot build
    /// the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuBackend, CpuThreadError};
    ///
    /// assert!(matches!(
    ///     CpuBackend::with_threads(0),
    ///     Err(CpuThreadError::ZeroThreads)
    /// ));
    /// let backend = CpuBackend::with_threads(2)?;
    /// assert_eq!(backend.num_threads(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_threads(threads: usize) -> Result<Self, CpuThreadError> {
        if threads == 0 {
            return Err(CpuThreadError::ZeroThreads);
        }
        if threads == 1 {
            return Ok(Self {
                engine: CpuEngine::serial(),
            });
        }
        let pool = build_pool(Some(threads))?;
        Ok(Self {
            engine: CpuEngine::new(pool),
        })
    }

    /// Build a backend whose worker `i` is pinned to `cpus[i]`.
    ///
    /// This is opt-in per-worker pinning on Linux; its performance benefit is
    /// unmeasured. Every listed CPU must be in the main thread's allowed set
    /// ([`Self::allowed_cpus`]), which lets a caller split that set into
    /// disjoint lists, one per engine. Each worker gets exactly its one CPU,
    /// so a provider team created from a pinned worker inherits that one CPU.
    /// A one-element list still builds a one-worker pool rather than a serial
    /// engine; inline operations of a one-worker engine still run on the
    /// caller.
    ///
    /// The mask is a snapshot: a cpuset or CPU-hotplug change needs a new
    /// backend. Only the pool's workers are pinned; lane threads and inline
    /// operations are not moved.
    ///
    /// # Errors
    ///
    /// Returns [`CpuThreadError::EmptyCpuList`] for an empty list,
    /// [`CpuThreadError::DuplicateCpu`] for a repeated CPU,
    /// [`CpuThreadError::CpuNotAllowed`] for a CPU outside the allowed set,
    /// [`CpuThreadError::UnsupportedPlatform`] off Linux,
    /// [`CpuThreadError::Affinity`] when the snapshot or a worker's mask fails,
    /// [`CpuThreadError::WorkerAffinity`] when a worker cannot be pinned to
    /// exactly its CPU, and [`CpuThreadError::Pool`] or
    /// [`CpuThreadError::WorkerStartupClosed`] when the pool cannot be built or
    /// a worker does not report.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuBackend, CpuThreadError};
    ///
    /// assert!(matches!(
    ///     CpuBackend::pinned(&[]),
    ///     Err(CpuThreadError::EmptyCpuList)
    /// ));
    /// # #[cfg(target_os = "linux")]
    /// # {
    /// let allowed = CpuBackend::allowed_cpus()?;
    /// let backend = CpuBackend::pinned(&allowed[..1])?;
    /// assert_eq!(backend.num_threads(), 1);
    /// assert!(backend.engine().pool().is_some());
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pinned(cpus: &[usize]) -> Result<Self, CpuThreadError> {
        if cpus.is_empty() {
            return Err(CpuThreadError::EmptyCpuList);
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = cpus;
            Err(CpuThreadError::UnsupportedPlatform)
        }
        #[cfg(target_os = "linux")]
        {
            let allowed = affinity::main_thread_mask()?.cpus();
            let mut seen = std::collections::HashSet::with_capacity(cpus.len());
            for &cpu in cpus {
                if !seen.insert(cpu) {
                    return Err(CpuThreadError::DuplicateCpu { cpu });
                }
                if allowed.binary_search(&cpu).is_err() {
                    return Err(CpuThreadError::CpuNotAllowed { cpu });
                }
            }

            let workers = cpus.to_vec();
            let (sender, receiver) = std::sync::mpsc::channel();
            let pool = tenferro_cpu::thread_pool_builder()
                .num_threads(workers.len())
                .start_handler(move |worker| {
                    let outcome = workers
                        .get(worker)
                        .copied()
                        .ok_or_else(|| {
                            std::io::Error::other("Rayon worker index outside the pinned CPU list")
                        })
                        .and_then(affinity::pin_current_thread);
                    let _ = sender.send((worker, outcome));
                })
                .build()
                .map_err(|source| CpuThreadError::Pool {
                    source: Box::new(source),
                })?;

            for _ in 0..cpus.len() {
                match receiver.recv() {
                    Ok((_, Ok(()))) => {}
                    Ok((worker, Err(source))) => {
                        return Err(CpuThreadError::WorkerAffinity { worker, source });
                    }
                    Err(source) => {
                        return Err(CpuThreadError::WorkerStartupClosed { source });
                    }
                }
            }
            Ok(Self {
                engine: CpuEngine::new(pool),
            })
        }
    }

    /// The CPUs the main thread is allowed to run on right now, in ascending
    /// order.
    ///
    /// This is a snapshot, read with `sched_getaffinity(getpid())` on Linux. A
    /// caller can split it into disjoint lists and pass one list to each
    /// [`CpuBackend::pinned`] engine. Off Linux this reports
    /// [`CpuThreadError::UnsupportedPlatform`].
    ///
    /// # Errors
    ///
    /// Returns [`CpuThreadError::UnsupportedPlatform`] off Linux and
    /// [`CpuThreadError::Affinity`] when the affinity query fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// # #[cfg(target_os = "linux")]
    /// # {
    /// let allowed = CpuBackend::allowed_cpus()?;
    /// assert!(!allowed.is_empty());
    /// assert!(allowed.windows(2).all(|pair| pair[0] < pair[1]));
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn allowed_cpus() -> Result<Vec<usize>, CpuThreadError> {
        #[cfg(not(target_os = "linux"))]
        {
            Err(CpuThreadError::UnsupportedPlatform)
        }
        #[cfg(target_os = "linux")]
        {
            Ok(affinity::main_thread_mask()?.cpus())
        }
    }

    /// The number of workers operations on this backend may use.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// assert_eq!(CpuBackend::with_threads(1)?.num_threads(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn num_threads(&self) -> usize {
        self.engine.num_threads()
    }

    /// Borrow the engine this backend owns.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// assert_eq!(backend.engine().buffer_pool_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn engine(&self) -> &CpuEngine {
        &self.engine
    }

    /// Borrow the engine mutably, for one operation or one session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// backend.engine_mut().reset_buffer_pool();
    /// assert_eq!(backend.engine().buffer_pool_len(), 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn engine_mut(&mut self) -> &mut CpuEngine {
        &mut self.engine
    }

    /// Consume the backend and return the engine it owns.
    ///
    /// A caller that hands engines to another owner, such as a lane executor,
    /// uses this instead of borrowing and cloning: [`CpuBackend`] deliberately
    /// does not clone an execution domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let engine = CpuBackend::with_threads(1)?.into_engine();
    /// assert_eq!(engine.num_threads(), 1);
    /// assert!(engine.pool().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn into_engine(self) -> CpuEngine {
        self.engine
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for CpuBackend {
    type Target = CpuEngine;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for CpuBackend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

impl From<CpuEngine> for CpuBackend {
    fn from(engine: CpuEngine) -> Self {
        Self { engine }
    }
}

/// Caller-owned execution domain used by context-aware tensor algorithms.
///
/// Values are validated against the exact runtime represented by the selected
/// context; no implicit host/device transfer is performed by this enum.
#[derive(Clone, Debug)]
pub enum ExecutionContext {
    /// Host execution through one caller-owned CPU context.
    Cpu(Arc<CpuExecutionContext>),
    /// CUDA execution through one caller-owned CUDA context.
    #[cfg(feature = "tenferro-cuda")]
    Cuda(Arc<crate::cuda::CudaExecutionContext>),
}

impl ExecutionContext {
    /// Check whether this is the process-global CPU context.
    ///
    /// Compatibility boundary: legacy CPU-global callers route through the
    /// historical host code paths (bitwise-identical numerics) while explicit
    /// contexts use the scoped primitives. Without the global-defaults
    /// feature there is no global context, so this always reports false.
    pub fn is_global_default_cpu(&self) -> bool {
        #[cfg(feature = "global-defaults")]
        {
            match self {
                ExecutionContext::Cpu(context) => {
                    let own = context.eager_runtime().map(|runtime| runtime.id());
                    let global = defaults::default_eager_ctx().map(|runtime| runtime.id());
                    matches!((own, global), (Ok(a), Ok(b)) if a == b)
                }
                #[cfg(feature = "tenferro-cuda")]
                ExecutionContext::Cuda(_) => false,
            }
        }
        #[cfg(not(feature = "global-defaults"))]
        {
            let _ = self;
            false
        }
    }
}

/// Error returned by explicit CPU context graph or eager-runtime operations.
///
/// The original tenferro diagnostic is retained as the error source.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use std::sync::Arc;
/// use tensor4all_tensorbackend::CpuExecutionContextError;
///
/// let error = CpuExecutionContextError::Initialization {
///     component: "graph runtime",
///     source: Arc::new(std::io::Error::other("registration failed")),
/// };
/// assert!(error.source().is_some());
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CpuExecutionContextError {
    /// A context-owned graph or eager runtime could not be initialized.
    #[error("failed to initialize {component}: {source}")]
    Initialization {
        /// Context component being initialized.
        component: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// The calling thread is already inside a session of this context.
    ///
    /// One context owns one engine and lends it to one caller at a time. A
    /// session entered from inside another session on the same thread would
    /// wait for a lock that thread already holds, so it is reported as this
    /// typed error before any lock is taken.
    #[error("recursive CPU execution-context session entry on one thread")]
    Reentered,
    /// Graph compilation or execution failed.
    #[error("CPU graph {operation} failed: {source}")]
    Graph {
        /// Graph operation that failed.
        operation: &'static str,
        /// Original tenferro diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

thread_local! {
    /// Identities of the execution contexts whose engines this thread holds.
    ///
    /// Re-entry is per context: holding one context's engine must not block an
    /// implicit entry that uses another context's engine.
    static ACTIVE_CONTEXTS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_context_id() -> u64 {
    NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) struct CanonicalSessionGuard {
    context_id: u64,
}

impl CanonicalSessionGuard {
    /// Whether this context's engine is already held on this thread.
    pub(crate) fn is_active(context_id: u64) -> bool {
        ACTIVE_CONTEXTS.with(|active| active.borrow().contains(&context_id))
    }

    pub(crate) fn enter(context_id: u64) -> Self {
        ACTIVE_CONTEXTS.with(|active| active.borrow_mut().push(context_id));
        Self { context_id }
    }
}

impl Drop for CanonicalSessionGuard {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|active| {
            let mut active = active.borrow_mut();
            if active.last() == Some(&self.context_id) {
                active.pop();
            } else {
                active.retain(|context_id| *context_id != self.context_id);
            }
        });
    }
}

impl CpuExecutionContextError {
    fn initialization(
        component: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Initialization {
            component,
            source: Arc::new(source),
        }
    }

    fn graph(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Graph {
            operation,
            source: Arc::new(source),
        }
    }
}

/// Failure entering the engine of the context that owns an eager runtime.
///
/// The implicit frontend resolves the engine from the runtime of the operand it
/// acts on; the owner may be a CPU or a CUDA execution context, so this error
/// keeps the owning context's typed diagnostic instead of flattening both into
/// one string.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{
///     CpuExecutionContextError, ExecutionContextEntryError,
/// };
///
/// let error = ExecutionContextEntryError::from(CpuExecutionContextError::Reentered);
/// assert!(matches!(
///     error,
///     ExecutionContextEntryError::Cpu(CpuExecutionContextError::Reentered)
/// ));
/// ```
#[cfg(feature = "global-defaults")]
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutionContextEntryError {
    /// The owning CPU context rejected the session entry.
    #[error(transparent)]
    Cpu(#[from] CpuExecutionContextError),
    /// The owning CUDA context rejected the session entry.
    #[cfg(feature = "tenferro-cuda")]
    #[error(transparent)]
    Cuda(#[from] crate::cuda::CudaExecutionContextError),
}

/// The engine a context owns, plus the graph runtime derived from it.
///
/// The registration the graph runtime holds is derived from the engine, and
/// `Runtime::run` needs `&mut engine`, so both live under one lock. The
/// [`GraphCompiler`] is not here: it never touches the engine, so it has its own
/// lock and a compile never waits for a running graph.
struct EngineState {
    backend: CpuBackend,
    runtime: Option<Runtime>,
}

impl EngineState {
    fn new(backend: CpuBackend) -> Self {
        Self {
            backend,
            runtime: None,
        }
    }

    /// The graph runtime of this context, built from the engine on first use.
    fn runtime(&mut self) -> Result<&mut Runtime, CpuExecutionContextError> {
        let Self { backend, runtime } = self;
        match runtime {
            Some(runtime) => Ok(runtime),
            None => {
                let built = build_graph_runtime(backend)?;
                Ok(runtime.insert(built))
            }
        }
    }
}

/// The runtime engine id `engine` is registered under.
///
/// An engine built by [`CpuEngine::with_runtime_engine_id`] runs under its own
/// id, so every registration and extension module installed on its behalf must
/// use that id rather than the canonical [`tenferro_cpu::runtime_engine_id`].
fn engine_runtime_id(engine: &CpuEngine) -> tenferro::EngineId {
    ExecutionEngine::target(engine).engine_id().clone()
}

/// The engine state one context owns, behind a handle that can also be reached
/// through the eager-runtime owner table.
///
/// A [`CpuExecutionContext`] holds this in an [`Arc`]; the process-wide owner
/// table holds only a [`Weak`](std::sync::Weak), so the entry dangles as soon as the context is
/// dropped instead of keeping the engine alive.
pub(crate) struct EngineOwner {
    id: u64,
    /// The engine's registered runtime id, captured once at construction.
    ///
    /// The eager runtime installs its extension modules under this id, and it is
    /// read without taking the engine lock so building the eager runtime can
    /// never wait for a running session.
    engine_id: tenferro::EngineId,
    /// The engine's prepared-plan cache, captured once at construction.
    ///
    /// A session entered from a Rayon worker runs inline on a fresh serial
    /// engine that shares this cache (tensor4all-rs#830). Holding the handle
    /// here keeps that entry off the engine lock: the worker never waits for an
    /// engine its own pool may be holding.
    plans: tenferro_cpu::CpuPlanCache,
    engine: Mutex<EngineState>,
}

impl EngineOwner {
    fn new(backend: CpuBackend) -> Self {
        Self {
            id: next_context_id(),
            engine_id: engine_runtime_id(backend.engine()),
            plans: backend.engine().plan_cache().clone(),
            engine: Mutex::new(EngineState::new(backend)),
        }
    }

    /// Lock the engine state, recovering a lock poisoned by a panic.
    ///
    /// A panic while the engine was borrowed leaves tenferro's engine marked
    /// broken, and the lock poisoned. The engine is kept: it carries the
    /// caller's pool, allocation domain, batch policy and runtime engine id, so
    /// replacing it with a default engine would silently change the caller's
    /// execution policy. Recovering the poison only makes the lock usable again;
    /// operations on the broken engine are refused with tenferro's typed
    /// `CpuEngineError::Broken`, and the caller builds a new context for a new
    /// engine.
    fn lock_engine(&self) -> std::sync::MutexGuard<'_, EngineState> {
        match self.engine.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                self.engine.clear_poison();
                guard
            }
        }
    }

    /// Check the per-context re-entry guard and install it for one entry.
    ///
    /// Every fallible entry checks before it locks: a nested entry would wait
    /// for an engine lock this thread already holds, so it is reported as
    /// [`CpuExecutionContextError::Reentered`] instead.
    fn enter_session(&self) -> Result<CanonicalSessionGuard, CpuExecutionContextError> {
        if CanonicalSessionGuard::is_active(self.id) {
            return Err(CpuExecutionContextError::Reentered);
        }
        Ok(CanonicalSessionGuard::enter(self.id))
    }

    /// The single guarded entry that locks this owner's engine.
    ///
    /// This is the only way to lock the engine from a fallible entry: the
    /// re-entry guard is checked and installed here, so no caller can hold the
    /// lock without holding the guard too. [`CpuExecutionContext::with_backend`]
    /// is the one remaining locking entry, and it installs the same guard
    /// without being able to report the rejection.
    fn begin_session(&self) -> Result<EngineEntry<'_>, CpuExecutionContextError> {
        let guard = self.enter_session()?;
        Ok(EngineEntry {
            state: self.lock_engine(),
            _guard: guard,
        })
    }

    /// Run `f` in one session of this owner's engine.
    ///
    /// This is the single session-entry implementation shared by the owning
    /// context and the eager-runtime route: the same per-context re-entry guard
    /// and the same inline-serial rule for Rayon workers apply either way.
    fn with_session<R>(
        &self,
        f: impl FnOnce(&mut CpuEngine) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        // A worker runs inline on its own serial engine instead of waiting for a
        // pool it belongs to (tensor4all-rs#830). The fresh engine shares the
        // plan cache, whose entries are immutable prepared plans, so a stage
        // entered per item does not re-plan. It does **not** share the context
        // engine's buffer pool, extension caches, batch policy or allocation
        // domain: an entry from a worker therefore allocates and batches
        // independently, and only while it runs.
        if rayon::current_thread_index().is_some() {
            let _guard = self.enter_session()?;
            let mut engine = CpuEngine::serial().with_plan_cache(self.plans.clone());
            return Ok(f(&mut engine));
        }
        let mut entry = self.begin_session()?;
        Ok(f(entry.backend.engine_mut()))
    }
}

/// One held engine session: the re-entry guard and the locked engine state.
///
/// The guard lives exactly as long as the lock it makes safe, so dropping the
/// entry releases the engine before the thread is allowed to enter again.
struct EngineEntry<'a> {
    state: std::sync::MutexGuard<'a, EngineState>,
    _guard: CanonicalSessionGuard,
}

impl Deref for EngineEntry<'_> {
    type Target = EngineState;

    fn deref(&self) -> &EngineState {
        &self.state
    }
}

impl DerefMut for EngineEntry<'_> {
    fn deref_mut(&mut self) -> &mut EngineState {
        &mut self.state
    }
}

/// Caller-owned CPU execution domain for plain, graph, and eager-AD work.
///
/// The context owns one [`CpuBackend`] and lends its engine to one caller at a
/// time. A session entered from a Rayon worker runs inline on a fresh serial
/// engine that shares the context engine's prepared-plan cache instead of
/// waiting for the context engine: a worker that waits for a pool install is
/// handed more of the enclosing pool's work, and that work may enter a session
/// itself (tensor4all-rs#830). Graph preparation caches, the engine's buffer
/// pool and the eager runtime are owned by this context and are released when
/// it is dropped.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tensor4all_tensorbackend::CpuBackend;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let threads = context.with_backend(|backend| backend.num_threads());
/// assert_eq!(threads, 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuExecutionContext {
    owner: Arc<EngineOwner>,
    compiler: Mutex<GraphCompiler>,
    eager: OnceLock<Result<Arc<EagerRuntime>, CpuExecutionContextError>>,
}

impl std::fmt::Debug for CpuExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExecutionContext")
            .field("graph_initialized", &self.engine_initialized())
            .field("eager_initialized", &self.eager.get().is_some())
            .finish_non_exhaustive()
    }
}

impl CpuExecutionContext {
    /// Create an execution context from a caller-selected CPU backend.
    ///
    /// Runtime construction is lazy, so creating a context cannot fail and does
    /// not allocate another executor or consult environment configuration.
    pub fn from_backend(backend: CpuBackend) -> Self {
        Self {
            owner: Arc::new(EngineOwner::new(backend)),
            compiler: Mutex::new(GraphCompiler::new()),
            eager: OnceLock::new(),
        }
    }

    fn engine_initialized(&self) -> bool {
        self.owner
            .engine
            .try_lock()
            .is_ok_and(|state| state.runtime.is_some())
    }

    /// Lock the compiler, discarding one a panic left behind.
    ///
    /// A `GraphCompiler` holds planning state that a panic may have left half
    /// written; rebuilding it costs nothing a compile would not pay for the next
    /// graph anyway. The poison is cleared after the rebuild so later compiles
    /// reuse the fresh compiler instead of rebuilding it on every access.
    fn lock_compiler(&self) -> std::sync::MutexGuard<'_, GraphCompiler> {
        match self.compiler.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                *guard = GraphCompiler::new();
                self.compiler.clear_poison();
                guard
            }
        }
    }

    /// Run a plain tensor operation with this context's engine.
    ///
    /// The closure runs while the context-local engine lock is held. A lock
    /// poisoned by a panic is recovered, and the engine is kept: it carries the
    /// caller's pool, allocation domain, batch policy and runtime engine id, so
    /// a broken engine keeps refusing operations with tenferro's typed
    /// `CpuEngineError::Broken` instead of being silently replaced by a default
    /// one. The caller builds a new context when it needs a new engine. The
    /// session entries recover the same way.
    ///
    /// This borrows the engine for the whole closure, so a session entry from
    /// inside `f` on this thread is rejected with
    /// [`CpuExecutionContextError::Reentered`] instead of waiting for the engine.
    /// Calling `with_backend` from inside a session of this context is the
    /// caller's obligation not to do: this entry cannot report an error, so it
    /// would wait for the engine the session already holds.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        // The guard makes a nested session entry of this context a typed error,
        // not a wait; another context's engine stays independently enterable.
        let _guard = CanonicalSessionGuard::enter(self.owner.id);
        f(&mut self.owner.lock_engine().backend)
    }

    /// Run `f` in one session of this context's engine.
    ///
    /// This is the engine-borrow entry for callers that need the concrete
    /// engine (for example to hand it to an eager AD session). The context lends
    /// its engine to one caller at a time for the whole of `f`.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
    ///
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let threads = context.with_session(|engine| engine.num_threads())?;
    /// assert_eq!(threads, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_session<R>(
        &self,
        f: impl FnOnce(&mut CpuEngine) -> R,
    ) -> Result<R, CpuExecutionContextError> {
        self.owner.with_session(f)
    }

    /// Compile a backend-neutral traced graph using this context's compiler cache.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Graph`] when graph compilation fails.
    pub fn compile_graph(
        &self,
        graph: &TracedGraph,
    ) -> Result<CompiledGraph, CpuExecutionContextError> {
        self.lock_compiler()
            .compile_traced_graph(graph)
            .map_err(|source| CpuExecutionContextError::graph("compilation", source))
    }

    /// Execute a compiled graph in this context's runtime and prepared-plan cache.
    ///
    /// `CompiledGraph` is backend-neutral. Backend-prepared executables and
    /// workspaces never leave this context-owned runtime.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context, and
    /// [`CpuExecutionContextError::Initialization`], [`CpuExecutionContextError::Graph`]
    /// when runtime initialization, preparation, or execution fails.
    pub fn run_graph(
        &self,
        graph: &CompiledGraph,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>, CpuExecutionContextError> {
        let reads = inputs
            .iter()
            .map(|input| TensorRead::from_tensor(input))
            .collect::<Vec<_>>();
        self.run_graph_reads(graph, &reads)
    }

    /// Execute a compiled graph on borrowed tensor views.
    ///
    /// The read form is what the bridge uses: an operand that is already a
    /// borrowed view stays borrowed instead of being copied into an owner first.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context, and
    /// [`CpuExecutionContextError::Initialization`], [`CpuExecutionContextError::Graph`]
    /// when runtime initialization, preparation, or execution fails.
    pub(crate) fn run_graph_reads(
        &self,
        graph: &CompiledGraph,
        inputs: &[TensorRead<'_>],
    ) -> Result<Vec<Tensor>, CpuExecutionContextError> {
        let mut state = self.owner.begin_session()?;
        let EngineState { backend, runtime } = &mut *state;
        let runtime = match runtime {
            Some(runtime) => &*runtime,
            None => &*runtime.insert(build_graph_runtime(backend)?),
        };
        let target = backend.engine().target();
        let prepared = runtime
            .prepare(graph, inputs, std::slice::from_ref(&target))
            .map_err(|source| CpuExecutionContextError::graph("preparation", source))?;
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [backend.engine_mut()];
            runtime
                .run(&mut engines, &prepared, inputs)
                .map_err(|source| CpuExecutionContextError::graph("execution", source))?
        };
        outputs
            .into_iter()
            .map(|output| {
                output
                    .into_tensor(backend.engine_mut())
                    .map_err(|source| CpuExecutionContextError::graph("output", source))
            })
            .collect()
    }

    /// Return this context's eager reverse-AD runtime.
    ///
    /// The runtime is engine-independent: it is built once from an
    /// [`AdContext`] with the linalg AD rules and the einsum/linalg extension
    /// modules installed under this context engine's runtime id, and every call
    /// takes the caller's engine ([`EagerRuntime::with_eager_session`]). Repeated
    /// calls return the same runtime and therefore the same eager compilation
    /// cache. The runtime registers itself with the eager-runtime owner table once,
    /// when it is built.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError`] when linalg AD-rule or eager-runtime
    /// registration fails.
    pub fn eager_runtime(&self) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
        self.eager
            .get_or_init(|| {
                let runtime = build_eager_runtime(&self.owner.engine_id)?;
                #[cfg(feature = "global-defaults")]
                register_eager_runtime_owner(
                    &runtime,
                    RuntimeOwner::Cpu(Arc::downgrade(&self.owner)),
                );
                Ok(runtime)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    /// Return statistics for this context's runtime-owned graph caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context, and
    /// [`CpuExecutionContextError`] when graph initialization or the cache
    /// statistics query fails.
    pub fn graph_cache_stats(
        &self,
    ) -> Result<tenferro::RuntimeCacheStats, CpuExecutionContextError> {
        let mut state = self.owner.begin_session()?;
        let runtime = state.runtime()?;
        runtime
            .cache_stats()
            .map_err(|source| CpuExecutionContextError::graph("cache statistics", source))
    }

    /// Return retained-buffer statistics for this context's engine.
    ///
    /// `buffers` counts the retained allocations and `capacity_bytes` their
    /// total element capacity. A context whose pool has just been released by
    /// [`Self::reset_graph_buffer_pool`] reports zero.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tenferro::program::{CoreSemanticOp, ProgramInputSpec};
    /// use tenferro::{DType, Tensor, TraceContext};
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    ///
    /// let mut trace = TraceContext::new();
    /// let input = trace.input(ProgramInputSpec::new(DType::F64, [2_usize.into()]))?;
    /// let output = trace.add_op(CoreSemanticOp::Neg, &[input])?[0];
    /// let graph = trace.finish(&[output])?;
    /// let compiled = context.compile_graph(&graph)?;
    /// let values = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0])?;
    /// let output = context.run_graph(&compiled, &[&values])?;
    /// assert_eq!(output[0].as_slice::<f64>()?, &[-1.0, 2.0]);
    ///
    /// // Releasing the pool empties it.
    /// context.reset_graph_buffer_pool()?;
    /// let after = context.graph_buffer_pool_stats()?;
    /// assert_eq!(after.buffers, 0);
    /// assert_eq!(after.capacity_bytes, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn graph_buffer_pool_stats(&self) -> Result<BufferPoolStats, CpuExecutionContextError> {
        Ok(self
            .owner
            .begin_session()?
            .backend
            .engine()
            .buffer_pool_stats())
    }

    /// Release retained buffers owned by this context's engine.
    ///
    /// The graph runtime, its compiled plans and its prepared-plan cache are
    /// kept, so callers can keep running previously compiled graphs afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// context.reset_graph_buffer_pool()?;
    ///
    /// // The pool retains nothing afterwards, and the runtime stays usable.
    /// let stats = context.graph_buffer_pool_stats()?;
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn reset_graph_buffer_pool(&self) -> Result<(), CpuExecutionContextError> {
        self.owner
            .begin_session()?
            .backend
            .engine_mut()
            .reset_buffer_pool();
        Ok(())
    }

    /// Recreate this context's graph runtime and release its prepared caches.
    ///
    /// # Errors
    ///
    /// Returns [`CpuExecutionContextError::Reentered`] when this thread is
    /// already inside a session of this context.
    pub fn reset_graph_runtime(&self) -> Result<(), CpuExecutionContextError> {
        {
            let mut state = self.owner.begin_session()?;
            state.runtime = None;
            state.backend.engine_mut().reset_buffer_pool();
        }
        *self.lock_compiler() = GraphCompiler::new();
        Ok(())
    }
}

fn build_graph_runtime(backend: &CpuBackend) -> Result<Runtime, CpuExecutionContextError> {
    // Register and extend the runtime under the engine's *own* runtime id: an
    // engine built with `CpuEngine::with_runtime_engine_id` would not be found
    // by a preparation targeting the canonical CPU id.
    let engine_id = engine_runtime_id(backend.engine());
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration_with_id(backend.engine(), engine_id.clone())
                .map_err(|source| {
                    CpuExecutionContextError::initialization("graph CPU engine", source)
                })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("graph CPU engine", source))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuEngine>(engine_id).map_err(|source| {
                CpuExecutionContextError::initialization("einsum extension", source)
            })?,
        )
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    builder
        .build()
        .map_err(|source| CpuExecutionContextError::initialization("graph runtime", source))
}

/// Build an AD context with the built-in linalg semantic rules installed.
///
/// The context is backend-neutral: CPU and CUDA eager runtimes share the same
/// rule set and differ only in the execution extension modules installed on the
/// runtime.
pub(crate) fn build_ad_context(
) -> Result<AdContext, tenferro_ad::semantic_extension::SemanticExtensionRegistryError> {
    let rules = tenferro_linalg::semantic_ad_rules()?;
    let builder = AdContext::builder().with_semantic_extension_rules(rules)?;
    Ok(match builder.build() {
        Ok(context) => context,
        Err(never) => match never {},
    })
}

fn build_eager_runtime(
    engine_id: &tenferro::EngineId,
) -> Result<Arc<EagerRuntime>, CpuExecutionContextError> {
    let ad_context = build_ad_context()
        .map_err(|source| CpuExecutionContextError::initialization("linalg AD rules", source))?;
    let runtime = EagerRuntime::with_ad_context(&ad_context)
        .map_err(|source| CpuExecutionContextError::initialization("eager runtime", source))?;
    // [AI Supplied] Install the built-in extension modules before publishing
    // the shared context. Lazy first use reconfigures the runtime and advances
    // its epoch; doing that after tensors have prepared AD derivatives can
    // invalidate those prepared programs under parallel first use.
    //
    // The modules are installed under the *engine's own* runtime id: an engine
    // built with `CpuEngine::with_runtime_engine_id` would otherwise not find
    // its einsum/linalg extensions.
    let einsum_module = tenferro_einsum::extension_module::<CpuEngine>(engine_id.clone())
        .map_err(|source| CpuExecutionContextError::initialization("einsum extension", source))?;
    runtime
        .install_extension_module(einsum_module)
        .map_err(|source| CpuExecutionContextError::initialization("einsum runtime", source))?;
    let linalg_module = tenferro_linalg::extension_module::<CpuEngine>(engine_id.clone())
        .map_err(|source| CpuExecutionContextError::initialization("linalg extension", source))?;
    runtime
        .install_extension_module(linalg_module)
        .map_err(|source| CpuExecutionContextError::initialization("linalg runtime", source))?;
    Ok(runtime)
}

/// A context that owns an eager runtime, held weakly by the process-wide table.
#[cfg(feature = "global-defaults")]
pub(crate) enum RuntimeOwner {
    Cpu(Weak<EngineOwner>),
    #[cfg(feature = "tenferro-cuda")]
    Cuda(Weak<crate::cuda::CudaEngineOwner>),
}

#[cfg(feature = "global-defaults")]
impl RuntimeOwner {
    fn upgrade(&self) -> Option<LiveRuntimeOwner> {
        match self {
            Self::Cpu(owner) => owner.upgrade().map(LiveRuntimeOwner::Cpu),
            #[cfg(feature = "tenferro-cuda")]
            Self::Cuda(owner) => owner.upgrade().map(LiveRuntimeOwner::Cuda),
        }
    }

    fn is_live(&self) -> bool {
        match self {
            Self::Cpu(owner) => owner.strong_count() > 0,
            #[cfg(feature = "tenferro-cuda")]
            Self::Cuda(owner) => owner.strong_count() > 0,
        }
    }
}

/// An upgraded [`RuntimeOwner`] that keeps its context alive for one session.
#[cfg(feature = "global-defaults")]
enum LiveRuntimeOwner {
    Cpu(Arc<EngineOwner>),
    #[cfg(feature = "tenferro-cuda")]
    Cuda(Arc<crate::cuda::CudaEngineOwner>),
}

/// Process-wide map from an eager runtime's id to the context engine that owns it.
///
/// The entry holds a [`Weak`] handle: a context registers its runtime when it
/// builds it, and the entry dangles when the context (the only strong owner) is
/// dropped. Lookups prune the dead entry and fall back to the process-global
/// context, so no background cleanup is needed.
#[cfg(feature = "global-defaults")]
static EAGER_RUNTIME_OWNERS: OnceLock<Mutex<HashMap<ContextId, RuntimeOwner>>> = OnceLock::new();

#[cfg(feature = "global-defaults")]
fn eager_runtime_owners() -> std::sync::MutexGuard<'static, HashMap<ContextId, RuntimeOwner>> {
    EAGER_RUNTIME_OWNERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "global-defaults")]
pub(crate) fn register_eager_runtime_owner(runtime: &EagerRuntime, owner: RuntimeOwner) {
    let mut owners = eager_runtime_owners();
    owners.retain(|_, owner| owner.is_live());
    owners.insert(runtime.id(), owner);
}

/// Run `f` on the engine of the context that owns `runtime`'s eager runtime.
///
/// This is the transitional implicit route: an implicit (context-free) eager
/// operation has no context argument, so it resolves the engine from the
/// runtime of the tensor it acts on instead of always taking the
/// process-global default context's engine. That keeps a caller-owned context's
/// thread policy, device domain, and engine serialization intact for
/// context-free readouts and intermediate operations, including on a context
/// that is not the global one.
///
/// The closure receives the owning engine as [`ExecutionEngine`]: the owner may
/// be a CPU context ([`CpuEngine`]) or, with the `tenferro-cuda` feature, a CUDA
/// context (`tenferro_gpu::cuda::CudaEngine`). The closure runs through the
/// owning context's normal session entry, so a nested entry on the same thread
/// is reported as [`ExecutionContextEntryError`]'s typed CPU or CUDA re-entry
/// variant instead of waiting for the engine. An entry from a Rayon worker runs
/// inline on a serial CPU engine that shares the context's prepared-plan cache;
/// a CUDA context keeps its single engine and never opens a second one.
/// When no live context owns `runtime` (for example after the context was
/// dropped), the entry falls back to the process-global CPU context; the global
/// context's own runtime resolves to itself.
///
/// # Errors
///
/// Returns [`ExecutionContextEntryError`] when the owning context's session
/// entry is rejected.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{with_runtime_engine, CpuBackend, CpuExecutionContext};
/// use tenferro_cpu::CpuEngine;
///
/// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
/// let runtime = context.eager_runtime()?;
/// let threads = with_runtime_engine(&runtime, |engine| {
///     engine
///         .as_any_mut()
///         .downcast_mut::<CpuEngine>()
///         .map(|engine| engine.num_threads())
/// })?;
/// assert_eq!(threads, Some(1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "global-defaults")]
pub fn with_runtime_engine<R>(
    runtime: &EagerRuntime,
    f: impl FnOnce(&mut dyn ExecutionEngine) -> R,
) -> Result<R, ExecutionContextEntryError> {
    let owner = {
        let mut owners = eager_runtime_owners();
        match owners.get(&runtime.id()).and_then(RuntimeOwner::upgrade) {
            Some(owner) => owner,
            None => {
                owners.remove(&runtime.id());
                LiveRuntimeOwner::Cpu(Arc::clone(&defaults::default_context().owner))
            }
        }
    };
    match owner {
        LiveRuntimeOwner::Cpu(owner) => owner
            .with_session(|engine| f(engine))
            .map_err(ExecutionContextEntryError::Cpu),
        #[cfg(feature = "tenferro-cuda")]
        LiveRuntimeOwner::Cuda(owner) => owner
            .with_execution_engine(f)
            .map_err(ExecutionContextEntryError::Cuda),
    }
}

#[cfg(feature = "global-defaults")]
mod defaults {
    use super::*;

    static DEFAULT_CONTEXT: OnceLock<Arc<CpuExecutionContext>> = OnceLock::new();

    #[cfg(test)]
    thread_local! {
        static FORCE_EAGER_CONTEXT_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Borrow the process-global CPU execution context.
    ///
    /// New code must take a caller-owned context; this accessor exists for the
    /// process-global convenience path and for callers that map the session-entry
    /// rejection into their own error type.
    pub(crate) fn default_context() -> &'static Arc<CpuExecutionContext> {
        DEFAULT_CONTEXT
            .get_or_init(|| Arc::new(CpuExecutionContext::from_backend(CpuBackend::new())))
    }

    /// Error returned when the process-global eager AD runtime cannot be initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::EagerContextError;
    ///
    /// let error = EagerContextError::Registration {
    ///     source: Arc::new(std::io::Error::other("registration failed")),
    /// };
    /// assert!(error.source().is_some());
    /// ```
    #[derive(Debug, Clone, thiserror::Error)]
    pub enum EagerContextError {
        /// The tenferro linalg AD extension rule could not be registered.
        #[error("failed to register tenferro linalg AD rule: {source}")]
        Registration {
            /// Original diagnostic returned by tenferro.
            #[source]
            source: Arc<dyn std::error::Error + Send + Sync + 'static>,
        },
    }

    /// Run a closure against the optional process-global CPU backend.
    pub fn with_default_backend<R>(f: impl FnOnce(&mut CpuBackend) -> R) -> R {
        default_context().with_backend(f)
    }

    /// Run `f` in one session of the process-global context's engine.
    ///
    /// The entry rejection stays inside tenferro's error type: it is reported as
    /// a `tenferro_tensor::Error` classified as runtime state, so callers keep
    /// one error type for session work.
    ///
    /// # Errors
    ///
    /// Returns the tenferro diagnostic when the entry is rejected or when `f`
    /// fails.
    pub(crate) fn with_default_session<T: Send>(
        f: impl FnOnce(&mut CpuEngine) -> Result<T, tenferro_tensor::Error> + Send,
    ) -> Result<T, tenferro_tensor::Error> {
        default_context()
            .with_session(f)
            .map_err(|source| {
                tenferro_tensor::Error::runtime_state_source("canonical session entry", source)
            })
            .and_then(std::convert::identity)
    }

    /// Run `f` against the process-global graph compiler.
    ///
    /// The compiler lock is released before the caller runs the compiled graph,
    /// so a compile never waits for a context engine that is busy running one.
    pub(crate) fn with_default_graph_compiler<R>(f: impl FnOnce(&mut GraphCompiler) -> R) -> R {
        f(&mut default_context().lock_compiler())
    }

    pub(crate) fn default_engine_buffer_pool_stats() -> anyhow::Result<BufferPoolStats> {
        default_context()
            .graph_buffer_pool_stats()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine_buffer_pool() -> anyhow::Result<()> {
        default_context()
            .reset_graph_buffer_pool()
            .map_err(anyhow::Error::new)
    }

    pub(crate) fn reset_default_engine() -> anyhow::Result<()> {
        default_context()
            .reset_graph_runtime()
            .map_err(anyhow::Error::new)
    }

    /// Return the optional process-global eager context used by convenience APIs.
    ///
    /// # Errors
    ///
    /// Returns [`EagerContextError::Registration`] when eager runtime
    /// initialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_tensorbackend::default_eager_ctx;
    ///
    /// let first = default_eager_ctx().unwrap();
    /// let second = default_eager_ctx().unwrap();
    /// assert!(Arc::ptr_eq(&first, &second));
    /// ```
    pub fn default_eager_ctx() -> Result<Arc<EagerRuntime>, EagerContextError> {
        #[cfg(test)]
        if FORCE_EAGER_CONTEXT_FAILURE.with(std::cell::Cell::get) {
            return Err(EagerContextError::Registration {
                source: Arc::new(std::io::Error::other(
                    "forced default eager context registration failure",
                )),
            });
        }
        default_context()
            .eager_runtime()
            .map_err(|source| EagerContextError::Registration {
                source: Arc::new(source),
            })
    }

    /// Borrow the process-global CPU execution context.
    ///
    /// Compatibility entry for CPU-global convenience APIs (e.g. the legacy
    /// context-free SRC entry): host tensors constructed through the global
    /// default belong to this exact context, so they validate against it.
    /// New code must take a caller-owned context instead of consulting this.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{default_cpu_execution_context, ExecutionContext};
    ///
    /// let context = ExecutionContext::Cpu(default_cpu_execution_context());
    /// assert!(matches!(context, ExecutionContext::Cpu(_)));
    /// ```
    pub fn default_cpu_execution_context() -> Arc<CpuExecutionContext> {
        Arc::clone(default_context())
    }

    #[cfg(test)]
    pub(crate) fn with_forced_eager_context_failure<T>(f: impl FnOnce() -> T) -> T {
        let previous = FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.replace(true));
        let result = f();
        FORCE_EAGER_CONTEXT_FAILURE.with(|failure| failure.set(previous));
        result
    }
}

#[cfg(feature = "global-defaults")]
pub(crate) use defaults::default_context;
#[cfg(all(test, feature = "global-defaults"))]
pub(crate) use defaults::with_forced_eager_context_failure;
#[cfg(feature = "global-defaults")]
pub use defaults::{
    default_cpu_execution_context, default_eager_ctx, with_default_backend, EagerContextError,
};
#[cfg(feature = "global-defaults")]
pub(crate) use defaults::{
    default_engine_buffer_pool_stats, reset_default_engine, reset_default_engine_buffer_pool,
    with_default_graph_compiler, with_default_session,
};

#[cfg(test)]
mod tests;
