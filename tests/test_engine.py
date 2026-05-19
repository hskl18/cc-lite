from eval.engine import engine_move_to_cc, parse_bestmove_output, score_to_value, to_engine_fen


def test_parse_bestmove_output_with_cp_score():
    result = parse_bestmove_output("info depth 4 score cp 123 pv a0a1\nbestmove b2e2")
    assert result.move.uci() == "b7e7"
    assert result.score_cp == 123


def test_parse_bestmove_output_with_mate_score():
    result = parse_bestmove_output("info depth 4 score mate -3\nbestmove h9h0")
    assert result.move.uci() == "h0h9"
    assert result.score_cp == -100000
    assert score_to_value(result.score_cp) == -1.0


def test_engine_move_to_cc_flips_ranks():
    assert engine_move_to_cc("c6c5") == "c3c4"


def test_to_engine_fen_converts_piece_letters_and_side_marker():
    fen = "rheakaehr/9/9/9/9/9/9/9/9/RHEAKAEHR r"
    assert to_engine_fen(fen) == "rnbakabnr/9/9/9/9/9/9/9/9/RNBAKABNR w"
