from mcts.search import MCTS, SearchConfig, UniformEvaluator
from xiangqi.board import Board


def test_mcts_uniform_smoke():
    board = Board.start()
    search = MCTS(UniformEvaluator(), SearchConfig(simulations=4))
    move, policy = search.run(board)
    assert move in board.legal_moves()
    assert abs(sum(policy.values()) - 1.0) < 1e-6
    assert search.last_nodes_per_second > 0

