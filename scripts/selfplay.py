import chess
import chess.pgn
import sys
import engine
import time
import os

def run_selfplay_session(game_index, think_time=0.1, verbose=True):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Mallard Selfplay"
    game.headers["Site"] = "localhost"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_index + 1)
    game.headers["White"] = "Mallard"
    game.headers["Black"] = "Mallard"

    node = game
    moves_played = 0

    claimed_draw = False

    while not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("No legal moves available.")
            break

        move = engine.select_move(board, depth=3, sims=600)
        if move is None:
            if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                print("Draw claimed successfully")
                claimed_draw = True
                break
            else:
                print("Mallard ERROR: No move returned and no draw available")
                raise RuntimeError("Engine failed to return move and can't claim draw")

        try:
            san = board.san(move)
        except Exception:
            san = "(invalid SAN)"

        board.push(move)
        node = node.add_variation(move)
        moves_played += 1

        if verbose:
            print(f"Move {moves_played}: {san}")
            print("-" * 30)
            time.sleep(think_time)

    if claimed_draw:
        final_result = "1/2-1/2"
    else:
        final_result = board.result()

    game.headers["Result"] = final_result

    os.makedirs("games", exist_ok=True)
    with open(f"games/{game_index}.pgn", "w") as pgn_file:
        print(game, file=pgn_file)

    if verbose:
        print(f"Game finished with result: {final_result}")

    return board, final_result, game

def run_selfplay_and_save(game_index):
    board, result, game = run_selfplay_session(game_index, verbose=False)
    with open(f"games/{game_index}.pgn", "w") as pgn_file:
        print(game, file=pgn_file)

def get_next_game_index():
    os.makedirs("games", exist_ok=True)
    files = [f for f in os.listdir("games") if f.endswith(".pgn") and f.split('.')[0].isdigit()]
    indices = [int(f.split('.')[0]) for f in files]
    return max(indices, default=0) + 1

if __name__ == "__main__":
    game_index = get_next_game_index()
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

    print(f"=== Starting Selfplay game {game_index} ===")
    board, result, game = run_selfplay_session(game_index, verbose=True)
    if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        result = "1/2-1/2"
    elif result in results and not board.can_claim_threefold_repetition() and not board.can_claim_fifty_moves():
        results[result] += 1
    else:
        results["*"] += 1

    print(f"Result of game {game_index}: {result}")
