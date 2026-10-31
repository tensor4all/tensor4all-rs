# Issue Intake Workflow

Use this workflow when creating, refining, or reviewing a tensor4all-rs issue.
It covers bug reports, feature requests, design discussions, and documentation
issues. New public features must start as accepted issues rather than direct
implementation pull requests.

The canonical `ai-ready` readiness checklist, authorization boundary, and stop
conditions are defined in the shared
`tensor4all-agent-rules/rules/common/repository.md` section "AI-Ready Issue
Burn". This workflow adds tensor4all-rs-specific intake details.

## Operating Rules

- Read the root `README.md` and the relevant issue template first. Read
  `REPOSITORY_RULES.md` when the issue may affect public APIs, crate
  boundaries, dependencies, backends, numerical behavior, AD, C API, or
  language bindings.
- Prefer repository and GitHub inspection over asking for facts that can be
  discovered locally.
- Do not create an issue until the user has approved the final draft unless the
  user explicitly requested direct creation.
- Treat maintainers as the authority for labels, acceptance, roadmap priority,
  and merge decisions.
- Treat `ai-ready` as maintainer-controlled authorization for autonomous AI
  issue burn, not as a label inferred or applied during ordinary intake.

## Drafting Checklist

Every issue should state:

- the intended behavior and why it matters;
- the relevant current behavior, reproducer, or limitation;
- affected crates, Rust APIs, C APIs, and language bindings;
- dependency, backend, dtype, numerical, AD, cache, feature, and provenance
  impact, including explicit statements when an area is not applicable;
- implementation scope and non-goals;
- acceptance criteria and focused tests, benchmarks, or documentation checks;
- links to related issues, pull requests, specs, source locations, and external
  references.

Bug reports should include the smallest practical reproducer, exact failure,
version or commit, environment, and expected regression coverage. Feature and
design issues should identify every public API or architecture decision needed
before implementation. Cross-repository changes should state the landing order
and link the related repository issues.

If prototype or external code is relevant, record its URL, license, copyright,
whether it may be used as an implementation reference, and whether it is only
illustrative and must not be copied or derived from.

## Create Or Hand Off

After approval, use the matching template under `.github/ISSUE_TEMPLATE/`.
Check that requested labels exist before applying them. Ordinary issue intake
must not apply `ai-ready`.

## Mark An Issue Ready For AI Burn

Only a maintainer may add or remove the `ai-ready` label. Apply the complete
shared checklist and additionally confirm that the issue explicitly covers:

- every affected tensor4all-rs crate and public Rust surface;
- C API, Tensor4all.jl, and Python binding impact or non-applicability;
- numerical tolerances, AD preservation, backend behavior, and dense-layout or
  materialization consequences where relevant;
- cross-repository landing order and linked issues when another Tensor4all
  repository is involved;
- the focused release-mode tests, documentation checks, and benchmark gates
  required by `AGENTS.md` and `REPOSITORY_RULES.md`.

If any shared or repository-specific requirement is unresolved, do not apply
the label. If a burn later needs a new decision, stop and have a maintainer
remove `ai-ready` until the issue is ready again.
