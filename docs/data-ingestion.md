# ICCS and UCCI move-record ingestion

cc-lite accepts a deliberately small JSONL envelope around ICCS or UCCI coordinate moves.
It does not scrape, download, or bundle third-party game collections.
The operator must supply records whose provenance and license are known.

## Input records

Each non-empty input line is one JSON object.
The required fields are `record_id`, `moves`, and `result`.
The optional `notation` field is either `iccs` or `ucci` and defaults to `iccs`.
The optional `initial_fen` field defaults to the standard starting position.

Both supported notations use four-character source and destination coordinates in this interface.
Files are `a` through `i`.
Ranks are `0` through `9`, counted from Black's home rank.
This interface accepts move tokens only, not complete UCCI protocol command transcripts.

```json
{"record_id":"example-001","notation":"iccs","moves":["a6a5","a3a4"],"result":"1/2-1/2"}
{"record_id":"example-002","notation":"ucci","moves":"c6c5 c3c4","result":"1-0"}
```

The accepted results are `1-0`, `0-1`, and `1/2-1/2`.
Records without a known result are rejected because they cannot provide an honest value target.
A decisive result may represent resignation before a terminal board position.
If the final board is terminal, the declared result must agree with the board result.

## Provenance

Every run requires a separate provenance JSON object.
This keeps source and license evidence attached to every generated artifact through a provenance fingerprint.

```json
{
  "schema_version": 1,
  "dataset_name": "My licensed Xiangqi records",
  "source": "Publisher archive or user-supplied records",
  "source_url": "https://example.invalid/records",
  "license": "License identifier or full license name",
  "license_url": "https://example.invalid/license",
  "license_notes": "Explain the applicable terms and any redistribution limits.",
  "attribution": "Required attribution text",
  "redistribution_permitted": false
}
```

`dataset_name`, `source`, `license`, `license_notes`, and `attribution` must be non-empty strings.
`source_url` and `license_url` are optional because user-supplied local records may not have public URLs.
`redistribution_permitted` is optional, but recording it is recommended.
The CLI records these statements without deciding whether a license interpretation is legally correct.
Do not commit generated records unless their terms clearly allow redistribution.

## Run the CLI

```bash
export PYTHONPATH=src
python -m data.ingest \
  --input path/to/games.jsonl \
  --provenance path/to/provenance.json \
  --output-dir runs/ingested/my-dataset \
  --validation-fraction 0.1 \
  --split-seed research-v1
```

The CLI refuses to replace known output artifacts unless `--force` is passed.
It validates each move against the board's legal moves before advancing the position.
It also rejects malformed FENs, missing kings, malformed move tokens, inconsistent terminal results, duplicate result conflicts, and reused record IDs that refer to different move sequences.
The rejection `code` distinguishes these cases with values such as `invalid_fen`, `invalid_move`, `illegal_move`, `terminal_result_conflict`, `duplicate_game`, and `record_id_conflict`.

## Determinism and outputs

The normalized initial FEN and move sequence define a game identity.
The identity is the SHA-256 digest of their canonical JSON representation.
Exact duplicate games keep the lexicographically smallest `record_id` and reject the other copies.
If copies declare different results, every conflicting copy is rejected.

The split is assigned at game level so positions from one game cannot cross between train and validation data.
The implementation hashes the split seed and game identity, reads the first 64 bits as a fraction, and compares it with `--validation-fraction`.
Changing input order does not change accepted game identities or split assignments.

The output directory contains:

- `games.jsonl` with normalized accepted games, their split, game hash, and provenance fingerprint.
- `train.jsonl` with training-ready positions assigned to the train split.
- `validation.jsonl` with training-ready positions assigned to the validation split.
- `rejections.jsonl` with stable reason codes, line numbers, input-line hashes, and diagnostic details.
- `manifest.json` with the original input hash, complete provenance, artifact hashes, counts, split configuration, and game-set fingerprints.

The training files contain `fen`, `move`, one-hot `policy`, `value`, `game_hash`, `record_id`, `ply`, and `provenance_fingerprint` fields.
Their `policy` and `value` fields are compatible with the supervised JSONL loader.
The value is from the side-to-move perspective at that position.

An empty split has the SHA-256 fingerprint of an empty byte string.
This is valid for tiny inputs or boundary fractions of `0` and `1`.
For meaningful model selection, use enough independently sourced games to populate both splits.
