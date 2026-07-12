import torch

from data.replay import ReplaySample
from model.network import ModelConfig, PolicyValueNet
from train.loop import train_samples
from train.self_play import _game_outcome
from xiangqi.board import BLACK, RED, Board
from xiangqi.encoding import MoveCodec


def test_training_smoke():
    board = Board.start()
    move = board.legal_moves()[0]
    samples = [
        ReplaySample(fen=board.to_fen(), policy={MoveCodec.encode(move): 1.0}, value=0.0)
        for _ in range(4)
    ]
    model = PolicyValueNet(ModelConfig(channels=16, blocks=1))
    stats = train_samples(
        model,
        samples,
        device=torch.device("cpu"),
        batch_size=2,
        epochs=1,
        lr=0.001,
    )
    assert stats.policy_loss > 0
    assert 0.0 <= stats.legal_move_accuracy <= 1.0


def test_cutoff_does_not_turn_material_advantage_into_self_play_win():
    board = Board.from_fen("4k4/9/9/9/9/4A4/9/9/9/4K4 r")

    assert board.legal_moves()
    assert board.material_score(RED) == 20
    assert _game_outcome(board, max_plies_reached=True) == {RED: 0.0, BLACK: 0.0}
