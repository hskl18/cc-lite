from eval.engine import parse_bestmove_output, score_to_value


def test_parse_bestmove_output_with_cp_score():
    result = parse_bestmove_output("info depth 4 score cp 123 pv a0a1\nbestmove b2e2")
    assert result.move.uci() == "b2e2"
    assert result.score_cp == 123


def test_parse_bestmove_output_with_mate_score():
    result = parse_bestmove_output("info depth 4 score mate -3\nbestmove h9h0")
    assert result.move.uci() == "h9h0"
    assert result.score_cp == -100000
    assert score_to_value(result.score_cp) == -1.0

