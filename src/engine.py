import chess
import random
import torch
import numpy as np
from nnue import NNUEModel
from features import extract_features
import os
from syzygy_helper import setup_syzygy, probe_wdl, is_syzygy_position

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_default_model = None

def load_default_model():
    global _default_model
    if _default_model is None or not os.path.isfile("model.pt"):
        print("Loading default model...")
        _default_model = NNUEModel().to(device)
        _default_model.load_state_dict(torch.load("model.pt", map_location=device))
        _default_model.eval()
    return _default_model

def is_game_over(board):
    return board.is_game_over()

def game_status(board):
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return f"checkmate {winner}"
    if board.is_stalemate():
        return "stalemate draw"
    if board.is_insufficient_material():
        return "draw insufficient material"
    if board.can_claim_threefold_repetition():
        return "draw threefold repetition"
    if board.can_claim_fifty_moves():
        return "draw 50-move rule"
    syzygy_tb.close()
    return None

def evaluate(board, model=None):
    if is_syzygy_position(board):
        wdl = probe_wdl(board)
        if wdl is not None:
            return {1: 1.0, 0: 0.0, -1: -1.0}.get(wdl, 0.0)

    if model is None:
        model = load_default_model()
    if not model:
        return random.uniform(-1, 1)

    with torch.no_grad():
        features = extract_features(board)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        _, value = model(input_tensor)
        return value.item()

def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: (board.is_capture(move), board.gives_check(move)), reverse=True)
    return moves

def alphabeta(board, depth, alpha, beta, maximizing, model=None, stop_event=None):
    if stop_event and stop_event.is_set():
        return evaluate(board, model), None

    if depth == 0 or board.is_game_over():
        return evaluate(board, model), None

    best_move = None
    ordered_moves = order_moves(board)

    if maximizing:
        max_eval = float("-inf")
        for move in ordered_moves:
            board.push(move)
            eval, _ = alphabeta(board, depth - 1, alpha, beta, False, model, stop_event)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval, _ = alphabeta(board, depth - 1, alpha, beta, True, model, stop_event)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def select_move(board, model=None, depth=None, sims=None, stop_event=None):
    if board.is_game_over():
        print(game_status(board))
        return None

    if model is None:
        model = load_default_model()

    if sims is not None:
        from mcts import select_best_move
        return select_best_move(board, model=model, sims=sims)
    else:
        depth_to_use = depth if depth is not None else 3
        _, move = alphabeta(board, depth_to_use, float("-inf"), float("inf"),
        board.turn == chess.WHITE, model=model, stop_event=stop_event)
        return move
