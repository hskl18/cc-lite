import math

from eval.engine import (
    cc_move_to_engine,
    command_argv,
    engine_move_to_cc,
    parse_bestmove_output,
    score_to_value,
    to_engine_fen,
)


def test_parse_bestmove_output_with_cp_score():
    result = parse_bestmove_output("info depth 4 score cp 123 pv a0a1\nbestmove b2e2")
    assert result.move.uci() == "b7e7"
    assert result.score_cp == 123


def test_parse_bestmove_output_with_mate_score():
    result = parse_bestmove_output("info depth 4 score mate -3\nbestmove h9h0")
    assert result.move.uci() == "h0h9"
    assert result.score_cp == -100000
    assert score_to_value(result.score_cp) == -1.0


def test_multipv_output_keeps_each_line_and_the_best_line_score():
    output = "\n".join(
        [
            "info depth 9 multipv 1 score cp 40 pv b2e2 h9g7",
            "info depth 9 multipv 2 score cp 10 pv h2e2",
            "info depth 10 multipv 1 score cp 35 pv b2e2 h9g7",
            "info depth 10 multipv 2 score cp -5 pv h2e2",
            "info depth 10 multipv 3 score mate -4 pv a0a1",
            "bestmove b2e2",
        ]
    )
    result = parse_bestmove_output(output)
    assert result.move.uci() == "b7e7"
    assert result.score_cp == 35
    assert [(move.uci(), score) for move, score in result.lines] == [
        ("b7e7", 35.0),
        ("h7e7", -5.0),
        ("a9a8", -100000.0),
    ]


def test_multipv_lines_skip_bound_scores_and_lines_without_a_score():
    output = "\n".join(
        [
            "info depth 10 multipv 1 score cp 35 pv b2e2",
            "info depth 10 multipv 2 score cp -5 pv h2e2",
            "info depth 11 multipv 1 score cp 90 lowerbound pv b2e2",
            "info depth 11 multipv 2 score cp -40 upperbound pv h2e2",
            "info depth 11 multipv 2 pv h2e2",
            "bestmove b2e2",
        ]
    )
    result = parse_bestmove_output(output)
    assert [(move.uci(), score) for move, score in result.lines] == [("b7e7", 35.0), ("h7e7", -5.0)]
    assert result.score_cp == 35


def test_multipv_value_is_line_one_even_when_the_other_line_is_only_a_bound():
    output = "info depth 10 multipv 1 score cp 35 pv b2e2\ninfo depth 10 multipv 2 score cp -40 upperbound pv h2e2\nbestmove b2e2"
    result = parse_bestmove_output(output)
    assert result.score_cp == 35
    assert [(move.uci(), score) for move, score in result.lines] == [("b7e7", 35.0)]


def test_multipv_without_an_exact_line_one_has_no_lines_and_no_score():
    output = "info depth 10 multipv 1 pv b2e2\ninfo depth 10 multipv 2 score cp 20 pv h2e2\nbestmove b2e2"
    result = parse_bestmove_output(output)
    assert result.lines == ()
    assert result.score_cp is None


def test_inconsistent_multipv_reports_have_no_lines():
    # Line 1 at the last exact depth is not the move played.
    stale = "\n".join(
        [
            "info depth 10 multipv 1 score cp 35 pv b2e2",
            "info depth 10 multipv 2 score cp -5 pv h2e2",
            "info depth 11 multipv 1 score cp 90 lowerbound pv h2e2",
            "info depth 11 multipv 2 score cp 10 pv b2e2",
            "bestmove h2e2",
        ]
    )
    assert parse_bestmove_output(stale).lines == ()
    repeated = "info depth 10 multipv 1 score cp 35 pv b2e2\ninfo depth 10 multipv 2 score cp 10 pv b2e2\nbestmove b2e2"
    assert parse_bestmove_output(repeated).lines == ()


def test_a_single_line_keeps_the_last_score_printed_as_before():
    output = "info depth 9 multipv 1 score cp 40 pv b2e2\ninfo depth 10 multipv 1 score cp 80\nbestmove b2e2"
    assert parse_bestmove_output(output).score_cp == 80


def test_engine_move_to_cc_flips_ranks():
    assert engine_move_to_cc("c6c5") == "c3c4"


def test_to_engine_fen_converts_piece_letters_and_side_marker():
    fen = "rheakaehr/9/9/9/9/9/9/9/9/RHEAKAEHR r"
    assert to_engine_fen(fen) == "rnbakabnr/9/9/9/9/9/9/9/9/RNBAKABNR w"


def test_engine_command_is_parsed_as_argv_without_a_shell():
    assert command_argv('python "fake engine.py" --mode ucci') == [
        "python",
        "fake engine.py",
        "--mode",
        "ucci",
    ]


def test_fairy_moves_use_one_based_ranks_including_rank_ten():
    assert engine_move_to_cc("h10g8", rank_base=1) == "h0g2"
    assert parse_bestmove_output("info depth 4 score cp 77\nbestmove b3e3", rank_base=1).move.uci() == "b7e7"


def test_engine_moves_off_the_board_are_rejected():
    for text, rank_base in (("a0a1", 1), ("a10a9", 0), ("j1a1", 1), ("a1a", 0)):
        try:
            engine_move_to_cc(text, rank_base=rank_base)
        except ValueError:
            continue
        raise AssertionError(f"{text!r} with rank base {rank_base} should be rejected")


def test_tanh_mapping_keeps_large_scores_apart_and_unknown_mappings_fail():
    assert score_to_value(500, 1000.0, "tanh") == math.tanh(0.5)
    assert score_to_value(3000, 1000.0, "tanh") < score_to_value(5000, 1000.0, "tanh") < 1.0
    assert score_to_value(3000, 1000.0) == score_to_value(5000, 1000.0) == 1.0
    try:
        score_to_value(1.0, 1000.0, "cubic")
    except ValueError:
        return
    raise AssertionError("unknown mappings must be rejected")


def test_cc_move_to_engine_inverts_engine_move_to_cc():
    assert cc_move_to_engine("c3c4") == "c6c5"
    assert cc_move_to_engine("h0g2", rank_base=1) == "h10g8"
    for move in ("a0a9", "h9g7", "e3e4", "i6i5"):
        for rank_base in (0, 1):
            assert engine_move_to_cc(cc_move_to_engine(move, rank_base), rank_base) == move
