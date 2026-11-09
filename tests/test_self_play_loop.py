from __future__ import annotations

import json
import sys

from data.replay import ReplaySample, write_jsonl
from model.checkpoint import load_checkpoint, save_checkpoint
from model.network import ModelConfig, PolicyValueNet
from train import self_play_loop
from xiangqi.board import START_FEN, Board
from xiangqi.encoding import MoveCodec


def test_split_games_spreads_games_without_empty_workers():
    assert self_play_loop.split_games(10, 4) == [3, 3, 2, 2]
    assert self_play_loop.split_games(2, 4) == [1, 1]


def test_one_generation_writes_a_checkpoint_and_a_manifest(tmp_path, monkeypatch):
    start = tmp_path / "start.pt"
    save_checkpoint(start, PolicyValueNet(ModelConfig(channels=8, blocks=1)))
    move = Board.from_fen(START_FEN).legal_moves()[0]
    anchor = tmp_path / "anchor.jsonl"
    write_jsonl(anchor, [ReplaySample(fen=START_FEN, policy={MoveCodec.encode(move): 1.0}, value=0.1)] * 4)
    config = tmp_path / "loop.yaml"
    config.write_text(
        "seed: 3\ndevice: cpu\n"
        "self_play: {games: 2, simulations: 2, max_plies: 4, workers: 1, temperature_plies: 2}\n"
        "loop: {generations: 1, replay_generations: 1, anchor_ratio: 0.5}\n"
        "training: {batch_size: 4, epochs: 1, lr: 0.001}\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["self_play_loop", "--config", str(config), "--checkpoint", str(start),
                                      "--anchor-labels", str(anchor), "--output-dir", str(output)])
    self_play_loop.main()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    [generation] = manifest["generations"]
    assert generation["self_play"]["positions"] == 8
    assert generation["training"]["anchor_positions"] == 4
    load_checkpoint(output / "gen-01" / "checkpoint.pt")
