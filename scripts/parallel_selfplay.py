import multiprocessing
from selfplay import run_selfplay_and_save, get_next_game_index

if __name__ == "__main__":
    num_games = 10
    num_workers = 4

    start_idx = get_next_game_index()
    indices = list(range(start_idx, start_idx + num_games))

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_selfplay_and_save, indices)
