# v0.2.0 release checklist

This checklist prepares a release candidate without creating a Git tag or GitHub Release.

## Source and package

- Confirm `pyproject.toml` is the only authoritative version source and reports `0.2.0`.
- Confirm `python -m cc_lite --version` and `cc-lite --version` report `0.2.0` after installation.
- Run `python scripts/release_check.py` from a clean checkout.
- Confirm the wheel and source distribution pass `twine check`.

## Rules and data

- Confirm the claimed rules behavior matches `docs/rules-profile.md`.
- Run the complete rules fixture and randomized reversibility suite.
- Confirm every training dataset has provenance, license notes, rejection output, split fingerprints, and content hashes.
- Keep downloaded datasets, model caches, engine binaries, and generated checkpoints out of Git.

## Evaluation evidence

- Validate every committed evaluation bundle from its raw games with `python -m eval.evidence validate`.
- Confirm all reported comparisons use the pinned opening suite, mirrored positions, paired colors, and declared seeds.
- Confirm confidence intervals and Elo estimates regenerate from raw terminal games.
- Confirm truncated games remain separate and never count as wins, draws, or losses.
- Confirm the sequential stopping decision is recorded and cannot use truncated games as favorable evidence.

## Claims boundary

- Do not claim engine strength from the committed debug artifact.
- Do not publish Elo or superiority claims without a completed paired multi-seed experiment and valid evidence bundle.
- Record the exact reference engine binary hash, command, options, depth, threads, and time control when an engine is used.
- Record unavailable external resources as blockers instead of substituting unsupported results.

## Maintainer actions after merge

- Review all CI checks on the merged commit.
- Create the `v0.2.0` tag only after every checklist gate passes.
- Create a GitHub Release only after the tag, artifacts, provenance, and claims are approved.
