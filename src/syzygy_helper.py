import chess
import chess.syzygy

_syzygy_tb = None

def setup_syzygy(path="syzygy"):
    global _syzygy_tb
    if _syzygy_tb is None:
        _syzygy_tb = chess.syzygy.open_tablebase(path)

def close_syzygy():
    global _syzygy_tb
    if _syzygy_tb is not None:
        _syzygy_tb.close()
        _syzygy_tb = None

def probe_wdl(board):
    if _syzygy_tb is None:
        return None
    try:
        return _syzygy_tb.probe_wdl(board)
    except Exception:
        return None

def is_syzygy_position(board):
    return _syzygy_tb is not None and len(board.piece_map()) <= 6
