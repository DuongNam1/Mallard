import chess.pgn
import chess.engine
import os
import json

ENGINE_PATH = "stockfish.exe"
GAMES_DIR = "games"
DEPTH = 16
TRAINING_OUTPUT = "training_data.json"

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

all_data = []

def get_eval(engine, board):
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=2)
        scores = []

        for entry in info:
            move = entry["pv"][0]  # principal variation move
            score = entry["score"].relative
            if score.is_mate():
                return None  # skip mate scores
            scores.append((move, score.score()))

        if len(scores) < 2:
            return None  # not enough data

        best_move, best_score = scores[0]
        second_move, second_score = scores[1]

        return best_move, best_score
    except Exception as e:
        print(f"Engine error: {e}")
        return None

def analyze_game(game):
    board = game.board()
    node = game
    game_data = []

    while node.variations:
        move = node.variation(0).move
        fen_before = board.fen()

        result = get_eval(engine, board)
        if result is None:
            break

        best_move, best_score = result
        board.push(move)

        played_result = get_eval(engine, board)
        if played_result is None:
            break

        _, played_score = played_result

        regret = best_score - played_score  # regret vs best move
        value = max(min(best_score / 1000.0, 1.0), -1.0)
        
        game_data.append({
            "fen": fen_before,
            "move": best_move.uci(),
            "value": value
        })

        node = node.variation(0)

    return game_data

pgn_files = sorted(
    [f for f in os.listdir(GAMES_DIR) if f.endswith(".pgn")],
    key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x
)

game_count = 0
for filename in pgn_files:
    pgn_path = os.path.join(GAMES_DIR, filename)
    with open(pgn_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            print(f"Analyzing Game {game_count + 1}: {filename}")
            all_data.extend(analyze_game(game))
            game_count += 1

with open(TRAINING_OUTPUT, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\nTraining data saved to {TRAINING_OUTPUT} ({len(all_data)} positions across {game_count} games)")
engine.quit()
