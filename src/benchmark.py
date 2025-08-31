import torch
import chess
from engine import select_move
from nnue import NNUEModel

def load_model(path, device):
    model = NNUEModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def play_game(model_white, model_black, device):
    board = chess.Board()
    while not board.is_game_over():
        model = model_white if board.turn == chess.WHITE else model_black
        move = select_move(board.copy(), model, depth=3, sims=400)
        if move not in board.legal_moves:
            # Illegal move (e.g., crash/fail), treat as loss
            return "0-1" if board.turn == chess.WHITE else "1-0"
        board.push(move)
    return board.result()

def play_matches(new_model_path, old_model_path, num_games=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_model = load_model(new_model_path, device)
    old_model = load_model(old_model_path, device)

    results = {"wins_new": 0, "wins_old": 0, "draws": 0}

    for game_num in range(num_games):
        print(f"[Benchmark] Game {game_num+1}/{num_games}")

        if game_num % 2 == 0:
            result = play_game(new_model, old_model, device)
        else:
            result = play_game(old_model, new_model, device)
            # Flip perspective
            if result == "1-0":
                result = "0-1"
            elif result == "0-1":
                result = "1-0"

        if result == "1-0":
            results["wins_new"] += 1
        elif result == "0-1":
            results["wins_old"] += 1
        else:
            results["draws"] += 1

    print(f"\n[Benchmark Complete] New Model Wins: {results['wins_new']}, "
          f"Old Model Wins: {results['wins_old']}, Draws: {results['draws']}")
    return results

if __name__ == "__main__":
    results = play_matches("checkpoints/model_v2.pt", "checkpoints/model_v1.pt", num_games=4)