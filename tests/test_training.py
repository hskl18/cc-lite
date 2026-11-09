import torch

from data.replay import ReplaySample
from model.network import ModelConfig, PolicyValueNet
from train.loop import train_samples
from train.self_play import _game_outcome
from xiangqi.board import BLACK, RED, Board, Move
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


def test_rules_profile_draw_stays_a_draw_in_self_play_targets():
    board = Board.from_fen("r3k4/9/9/9/9/4P4/9/9/9/R3K4 r")
    for move_text in ["a9b9", "a0b0", "b9a9", "b0a0"] * 2:
        board.push(Move.from_uci(move_text))

    assert _game_outcome(board, max_plies_reached=False) == {RED: 0.0, BLACK: 0.0}


def test_batched_self_play_is_seeded_and_labels_every_move():
    from train.self_play import generate_self_play_batched

    model = PolicyValueNet(ModelConfig(channels=8, blocks=1))
    kwargs = dict(device=torch.device("cpu"), games=3, simulations=4, max_plies=6, seed=5, temperature_plies=2)
    first, metrics = generate_self_play_batched(model, **kwargs)
    again, _ = generate_self_play_batched(model, **kwargs)
    assert [(s.fen, s.policy, s.value) for s in first] == [(s.fen, s.policy, s.value) for s in again]
    assert metrics["positions"] == len(first) == 18
    assert metrics["capped_games"] == 3
    # Capped games are draws, never material-based wins.
    assert {sample.value for sample in first} == {0.0}
    assert all(abs(sum(sample.policy.values()) - 1.0) < 1e-6 for sample in first)
