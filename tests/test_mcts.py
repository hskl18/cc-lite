from mcts.search import MCTS, Node, SearchConfig, UniformEvaluator
from xiangqi.board import Board, Move


def test_mcts_uniform_smoke():
    board = Board.start()
    search = MCTS(UniformEvaluator(), SearchConfig(simulations=4))
    move, policy = search.run(board)
    assert move in board.legal_moves()
    assert abs(sum(policy.values()) - 1.0) < 1e-6
    assert search.last_nodes_per_second > 0


def test_mcts_backs_up_repetition_as_a_draw():
    board = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    for move_text in ["a9b9", "a0b0", "b9a9", "b0a0"] * 2:
        board.push(Move.from_uci(move_text))
    search = MCTS(UniformEvaluator(), SearchConfig(simulations=1))

    value = search._evaluate_terminal_or_expand(Node(prior=1.0), board)

    assert value == 0.0
