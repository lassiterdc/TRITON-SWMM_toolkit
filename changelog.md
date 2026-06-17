# Changelog

## [Unreleased]

### Changed — BREAKING: package renamed `TRITON_SWMM_toolkit` → `hhemt`

The Python package, CLI command, distribution name, and conda environment are
renamed from `TRITON_SWMM_toolkit` to `hhemt`. `import hhemt`, `hhemt --help`,
and `conda activate hhemt` replace their `TRITON_SWMM_toolkit` equivalents.

**Clean-break notes (no automatic migration):**

- **Cache / example-data orphan.** Per-user caches and example data move from
  `~/.cache/TRITON_SWMM_toolkit`, `~/.local/share/TRITON_SWMM_toolkit`, and
  `<user_cache_dir>/TRITON_SWMM_toolkit/...` to the `hhemt`-named equivalents.
  Pre-rename caches are **orphaned** (not deleted); example data re-downloads
  into the new `hhemt` paths on first use. To reclaim disk, remove the old
  `TRITON_SWMM_toolkit` cache/data directories manually.
- **Pre-rename on-disk Snakefiles.** Any `{analysis_dir}/Snakefile` or
  `Snakefile.reprocess` generated before the rename bakes the old
  `-m TRITON_SWMM_toolkit.` module path and will raise `ModuleNotFoundError`
  on `resume`/`reprocess`. Regenerate via a fresh `run()` — the stale module
  path is not rewritten in place.
- **Environment variables renamed** `TRITON_SWMM_*` → `HHEMT_*` (e.g.
  `TRITON_SWMM_REQUIRE_EXAMPLE_DATA` → `HHEMT_REQUIRE_EXAMPLE_DATA`,
  `TRITON_SWMM_DISABLE_PROVENANCE_AUDIT` → `HHEMT_DISABLE_PROVENANCE_AUDIT`).
- The coupled-model **domain identifiers are unchanged** — `TRITONSWMM_*`
  class names and the `TRITON_SWMM_<word>` API/config identifiers (e.g.
  `TRITON_SWMM_example`, `triton_swmm_configuration_template`,
  `TRITON_SWMM_definition_template`) refer to the coupled TRITON+SWMM model,
  not the toolkit's package identity, and were deliberately preserved.
- On-disk analysis-tree layout is **unchanged** (no `LAYOUT_VERSION` bump, no
  migration); the rename is a pure Layer-1 package-identity change.
