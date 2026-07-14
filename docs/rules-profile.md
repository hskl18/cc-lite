# CC-Lite Research Adjudication Profile 1

This document defines the one Xiangqi rules profile supported by cc-lite v0.2.0.
It is intended to make local self-play and evaluation reproducible.
It is not a claim of complete compatibility with any national, regional, online, or tournament rulebook.

## Board and legal moves

The board has 10 ranks and 9 files.
FEN ranks run from Black's home rank at row 0 to Red's home rank at row 9.
Files are named `a` through `i`.
Red moves first in the standard starting position.

Piece movement follows the movement implemented by `xiangqi.board.Board`.
Kings and advisors stay in their own three-by-three palace.
Elephants cannot cross the river and an occupied elephant eye blocks the move.
Horses cannot move through an occupied horse leg.
Cannons require exactly one screen to capture and no screen for a quiet move.
Pawns move forward before crossing the river and may also move sideways after crossing it.
Kings may capture each other along an otherwise empty file under the flying-general rule.
A move is illegal when it leaves the moving side's king attacked, including when it exposes the two kings on the same file.

`Board.push()` accepts only a legal move for the side to move.
It rejects a move after adjudication, a move by the wrong side, an impossible piece move, and a move that leaves the moving king in check.
`Board.pop()` accepts only the most recently pushed move and its matching captured piece.

## Accepted position notation

The supported FEN form is ten slash-separated placement ranks followed by `r` or `b` for the side to move.
The side token may be omitted, in which case Red is to move.
Each rank must expand to exactly nine squares.
Empty-square digits must be from 1 through 9.
Piece letters are `KAEHRCP` for Red and the corresponding lowercase letters for Black.

A parsed nonterminal position must contain exactly one king for each side.
A terminal king-capture position may omit the captured king and place the surviving king on the capture square so recorded final FEN can be loaded and inspected.
A position cannot contain multiple kings for either side or omit both kings.
When both kings are present, each must be inside its own palace.
Piece counts cannot exceed the standard per-side inventory of one king, two advisors, two elephants, two horses, two rooks, two cannons, and five pawns.

The parser deliberately does not prove that every non-king piece could have reached its square from the standard starting position.
This keeps composed analysis positions usable while still rejecting malformed or structurally impossible king states.
FEN stores the placement and side to move only.
It does not store repetition history or the no-progress counter, so loading a FEN starts both histories at that position.

## Terminal conditions and precedence

`Board.adjudication()` returns the first applicable terminal decision in this order:

1. A missing king loses with reason `king_capture`.
2. A repeated position is decided under the repetition rule below.
3. A position at 120 no-progress plies is a draw with reason `no_progress_120_plies`.
4. A side with no legal move loses with reason `no_legal_moves`, whether or not its king is currently in check.

The returned `Adjudication` contains a `reason` and a `winner`.
The winner is `red` or `black` for a decisive result and `None` for a draw.
After a terminal decision, `legal_moves()` returns an empty list, `is_game_over()` returns true, and `result_for(color)` returns `1.0`, `0.0`, or `-1.0` from that color's perspective.

King capture remains a supported terminal representation because existing cc-lite search and engine adapters may produce it.
Normal play can also terminate one ply earlier when the checked side has no legal response.

## Repetition and perpetual check

A position identity is the complete piece placement plus the side to move.
The no-progress counter is not part of the repetition identity.
A position is repeated when the same identity appears three times in the current in-memory game history.

The repetition interval begins at the earliest of the three counted occurrences and ends at the current position.
For each color, cc-lite examines every move that color made in that interval.
A color is a perpetual checker when all of its moves in the interval gave check.

If exactly one color is a perpetual checker, that color loses with reason `perpetual_check`.
If neither color or both colors meet that condition, the game is a draw with reason `threefold_repetition`.

This rule is deterministic and intentionally narrower than full tournament long-check and long-capture adjudication.
It does not classify chase targets, protected versus unprotected pieces, alternating checks and chases, or tournament-specific exceptions.

## No-progress move limit

The no-progress counter increases after every move that is neither a capture nor a pawn move.
Any capture resets the counter to zero.
Any pawn move resets the counter to zero, including a pawn capture.
At 120 consecutive no-progress plies, the game is a draw with reason `no_progress_120_plies`.

There is no separate total-ply win or loss rule in this profile.
Experiment-level safety caps are truncations, not rule-based game results, and must be reported separately from wins, draws, and losses.

## Known tournament differences

Tournament rulebooks can assign responsibility for long checks and long chases using richer move classification than cc-lite records.
Some rulebooks use different repetition thresholds, move counters, claim procedures, or exemptions.
Some protocols encode additional FEN clock fields or adjudication state.
This profile does not implement those variations.

Results produced under this profile must be labeled as CC-Lite Research Adjudication Profile 1 results.
They must not be described as universally tournament-compliant Xiangqi results.
