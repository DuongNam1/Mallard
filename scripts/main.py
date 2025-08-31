import sys
import chess
from engine import select_move
import threading

search_thread = None

board = chess.Board()

def search_move(board, depth, sims):
    global best_move
    best_move = select_move(board, depth=depth, sims=sims, stop_event=stop_search_event)

def parse_go_command(line):
    depth = None
    nodes = None
    infinite = True

    parts = line.split()
    i = 1
    while i < len(parts):
        if parts[i] == "depth" and i + 1 < len(parts):
            try:
                depth = int(parts[i + 1])
                infinite = False
            except ValueError:
                pass
            i += 2
        elif parts[i] == "nodes" and i + 1 < len(parts):
            try:
                nodes = int(parts[i + 1])
                infinite = False
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    return depth, nodes, infinite

def uci_loop():
    global search_thread, best_move
    best_move = None
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line != "uci" and line != "isready" and line != "help" and line != "quit" and not line.startswith("position") and not line.startswith("go"):
            print(f"Unknown command: '{line}'. Type 'help' for more information.")
        if line == "help":
            print("Mallard is a chess engine built in Python entirely used for playing.")
            print("It is released as free software licensed under the GNU GPLv3 License.")
            print("Mallard is normally used with a graphical user interface (GUI) and implements")
            print("the Universal Chess Interface (UCI) protocol to communicate with a GUI, an API, etc.")
            print("\nCommands:")
            print("- 'uci': Ensure the engine implements the Universal Chess Interface (UCI) protocol. Should be")
            print("responded with the engine's identity, available options and 'uciok'.")
            print("- 'isready': Ensure the engine is ready to receive commands. Should")
            print("be responded with 'readyok'.")
            print("- 'position [fen <FEN>] moves <move1> <move2> ...: Tell the engine what position to analyze.")
            print("FEN should be input as 'startpos' if the position is the default position of the")
            print("chessboard. E.g.: position startpos e2e4")
            print("- 'go': Find the best move from the provided position. This command has multiple arguments such as:")
            print("+ 'depth <n>: Analyze the position n moves deep.")
            print("+ 'nodes <n>': Visit n positions.")
            print("- 'stop': Stop an infinite search.")
            print("- 'quit': Quit.")
        if line == "uci":
            print("id name Mallard")
            print("id author MallardDev")
            print("uciok")
        if line == "isready":
            print("readyok")
        if line == "quit":
            break
        if line.startswith("position"):
            set_position(line)
            continue
        if line == "stop":
            stop_search_event.set()
            if search_thread and search_thread.is_alive():
                search_thread.join()
            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove 0000")
            continue
        elif line.startswith("go"):
            # Parse depth and sims from the go command
            depth = 3
            sims = 600
            tokens = line.split()
            if "depth" in tokens:
                depth = int(tokens[tokens.index("depth") + 1])
            if "nodes" in tokens:
                sims = int(tokens[tokens.index("nodes") + 1])
            
            best_move = select_move(board, depth=depth, sims=sims)
            if best_move is None:
                print("bestmove 0000")
                continue
            
            # Create a copy of the board and push best move to get ponder move
            board_after_best = board.copy()
            board_after_best.push(best_move)
            
            # Get ponder move (best reply for opponent)
            ponder_move = select_move(board_after_best, depth=depth, sims=sims)
            
            if ponder_move:
                print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}")
            else:
                print(f"bestmove {best_move.uci()}")


def set_position(command):
    global board
    tokens = command.split()
    idx = tokens.index("position")
    if "startpos" in tokens:
        board = chess.Board()
        moves_start = tokens.index("moves") if "moves" in tokens else len(tokens)
    elif "fen" in tokens:
        fen = " ".join(tokens[tokens.index("fen") + 1 : tokens.index("moves")])
        board = chess.Board(fen)

    if "moves" in tokens:
        moves = tokens[tokens.index("moves") + 1 :]
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)

if __name__ == "__main__":
    uci_loop()
