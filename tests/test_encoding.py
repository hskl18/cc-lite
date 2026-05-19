from xiangqi.board import Board
from xiangqi.encoding import MoveCodec


def test_move_codec_round_trip():
    for move in Board.start().legal_moves():
        assert MoveCodec.decode(MoveCodec.encode(move)) == move


def test_legal_mask_marks_all_and_only_legal_indices():
    board = Board.start()
    mask = MoveCodec.legal_mask(board)
    legal_indices = set(MoveCodec.legal_indices(board))
    assert int(mask.sum()) == len(legal_indices)
    for index in legal_indices:
        assert mask[index]

