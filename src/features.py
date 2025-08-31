import numpy as np
import chess

def mirror_square(sq):
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))

def extract_features(board, move=None):
    features = np.zeros(64 * 12, dtype=np.float32)
    perspective = board.turn

    for square, piece in board.piece_map().items():
        sq = square if perspective == chess.WHITE else mirror_square(square)
        pt = piece.piece_type - 1 + (0 if piece.color == perspective else 6)
        features[sq * 12 + pt] = 1

    extras = [
        int(board.is_check()),
        int(board.has_castling_rights(board.turn)),
        int(board.has_castling_rights(not board.turn)),
        board.fullmove_number / 100,
        len(list(board.legal_moves)) / 100
    ]

    move_features = np.zeros(64 * 64, dtype=np.float32)
    if move is not None:
        from_sq = move.from_square
        to_sq = move.to_square
        move_index = from_sq * 64 + to_sq
        move_features[move_index] = 1.0

    return np.concatenate([features, np.array(extras, dtype=np.float32), move_features])
