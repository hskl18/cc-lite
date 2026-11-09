from mcts.search import MCTS, SearchConfig, UniformEvaluator
from xiangqi.board import Board, Move


def test_mcts_uniform_smoke(monkeypatch):
    def reject_public_push(*_args, **_kwargs):
        raise AssertionError("MCTS must use the trusted path for legal tree edges")

    board = Board.start()
    search = MCTS(UniformEvaluator(), SearchConfig(simulations=4))
    monkeypatch.setattr(Board, "push", reject_public_push)
    move, policy = search.run(board)
    assert move in board.legal_moves()
    assert abs(sum(policy.values()) - 1.0) < 1e-6
    assert search.last_nodes_per_second > 0


def test_mcts_backs_up_repetition_as_a_draw():
    board = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    for move_text in ["a9b9", "a0b0", "b9a9", "b0a0"] * 2:
        board.push(Move.from_uci(move_text))
    assert MCTS._terminal_value(board) == 0.0


def test_search_yields_only_boards_that_need_an_evaluation():
    board = Board.start()
    search = MCTS(UniformEvaluator(), SearchConfig(simulations=8, temperature=0.0))
    steps = search.search(board)
    yielded = [next(steps)]
    while True:
        try:
            yielded.append(steps.send(UniformEvaluator().evaluate(yielded[-1])))
        except StopIteration as done:
            move, policy = done.value
            break
    assert yielded[0] is board
    assert all(item.adjudication() is None for item in yielded)
    assert len(yielded) == 9
    assert move == MCTS(UniformEvaluator(), SearchConfig(simulations=8, temperature=0.0)).run(board)[0]
    assert abs(sum(policy.values()) - 1.0) < 1e-6
