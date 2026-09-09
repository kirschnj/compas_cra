# AGENTS.md

Guidance for coding agents working in `compas_cra`.

These rules favor careful, minimal changes over speed. Use judgment for trivial
edits, but apply the full verification workflow to behavioral or numerical
changes.

## Think Before Coding

- Inspect the relevant implementation, tests, and documentation before editing.
- State material assumptions when requirements or behavior are ambiguous.
- Do not silently choose between multiple plausible interpretations.
- Raise conflicts between the request, existing behavior, tests, and docs.
- Ask before changing public APIs, solver formulations, numerical tolerances,
  dependencies, or supported Python versions.
- Prefer the simplest approach that satisfies the stated goal.

## Keep Changes Focused

- Every changed line should relate directly to the requested work.
- Do not perform unrelated refactoring, cleanup, reformatting, or renaming.
- Match the existing architecture and style, even when another style is
  personally preferable.
- Do not introduce abstractions, configuration, or flexibility without a
  demonstrated need.
- Remove code made obsolete by your own change, but leave pre-existing dead code
  alone unless its removal is requested.
- Preserve existing comments unless they are incorrect because of the change.
- Never overwrite or discard uncommitted work that you did not create.

## Work Toward Verifiable Goals

- Convert requests into explicit, testable success criteria.
- For bug fixes, reproduce the bug with a focused test when practical.
- For behavior changes, add or update tests covering the new behavior.
- For refactors, preserve behavior and run relevant tests before and after.
- For multi-step work, use a short plan with a verification step for each part.
- Continue until the requested behavior is verified or a concrete blocker is
  identified.
- Do not claim that checks passed unless they were actually run.

## Repository Overview

- `src/compas_cra/`: Python package source.
- `src/compas_cra/equilibrium/`: CRA, penalty, and RBE solver implementations.
- `src/compas_cra/datastructures/`: CRA assembly data structures.
- `src/compas_cra/algorithms/`: assembly and interface algorithms.
- `src/compas_cra/geometry/`: geometry helpers.
- `tests/`: automated tests, including IPOPT-dependent solver tests.
- `docs/`: Sphinx documentation, tutorials, and examples.
- `scripts/`: conversion and tutorial scripts.
- `data/`: Rhino and Grasshopper source models.
- `tasks.py`: Invoke task definitions.

Do not manually edit generated build artifacts under `build/`.

## Environment And Dependencies

- The project supports Python 3.9 and newer versions declared in
  `pyproject.toml`.
- Runtime dependencies are defined in `requirements.txt`.
- Development dependencies are defined in `requirements-dev.txt`.
- Prefer the existing Conda environment files for platform-specific setup.
- IPOPT is an external solver required by the solver tests.
- Do not add, remove, or upgrade dependencies without approval.
- Never commit credentials, local environment files, or machine-specific paths.

## Code Conventions

- Follow the Ruff and Black configuration in `pyproject.toml`.
- Use a maximum line length of 119 characters.
- Preserve compatibility with Python 3.9 unless explicitly instructed
  otherwise.
- Follow existing naming, import, type annotation, and docstring patterns.
- Keep imports one per line, consistent with the Ruff configuration.
- New public functions and classes should be exported through the appropriate
  second-level package `__init__.py`.
- Treat solver equations, variable bounds, constraints, termination checks, and
  tolerances as behavior-sensitive code.
- Do not modify numerical defaults merely to make a failing test pass.

## Verification

Run checks proportionate to the change:

- Focused test: `pytest tests/<relevant_test>.py`
- Full tests: `invoke test`
- Lint: `invoke lint`
- Style checks: `invoke check`
- Documentation build, when docs or public APIs change: `invoke docs`

Start with focused tests for fast feedback. Before completion, run the relevant
lint and full test commands when the environment supports them.

Tests involving Pyomo solvers require a working IPOPT installation. If IPOPT is
unavailable:

- Run all relevant checks that do not require IPOPT.
- Do not weaken or skip tests simply to obtain a passing result.
- Report exactly which checks could not run and why.
- Do not describe the work as fully verified.

Review the final diff for unintended edits after running formatting or build
commands.

## Documentation And Public Changes

- Update documentation when public behavior or usage changes.
- Add an entry under the appropriate `CHANGELOG.md` `Unreleased` heading for
  user-visible changes.
- Keep examples aligned with the current public API.
- Preserve backward compatibility unless a breaking change is explicitly
  requested and approved.

## Git Safety

- Assume the working tree may contain user changes.
- Do not revert, replace, or reformat unrelated modifications.
- Do not use destructive Git commands or rewrite history without approval.
- Do not commit, push, merge, or create releases unless explicitly requested.
- Keep generated files and local environment artifacts out of commits.

## Completion Report

When finishing a task, report:

- What changed and why.
- Which tests and checks were run and their results.
- Any checks that could not be run.
- Remaining risks, assumptions, or follow-up work.
