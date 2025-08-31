# The training has been giving me a huge pain in the dick. I'd rather stick my face to an elephant's butt than babysitting this fucking moronic code. FUCKKKK FUCKKK FUCKK FUCKKKKKK FUCKKK BITCH SHIT CUNT CUNT CUNT FUCKKK

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import chess
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from features import extract_features
from nnue import NNUEModel
from collections import Counter

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_FILE = "training_data.json"
RL_GAMES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(TRAIN_FILE) as f:
    data = json.load(f)

X = []
policy_y = []
value_y = []

print("Loading and encoding data...")

for entry in data:
    fen = entry["fen"]
    move = entry["move"]
    value = entry["value"]

    board = chess.Board(fen)
    move = chess.Move.from_uci(move)

    if board.is_legal(move):
        features = extract_features(board, move)
        X.append(features)

        # Map move to unique index in [0..4095] = from*64 + to
        move_index = move.from_square * 64 + move.to_square
        policy_y.append(move_index)
        value_y.append(value)

print(f"Total usable samples: {len(X)}")

X = np.array(X, dtype=np.float32)
policy_y = np.array(policy_y, dtype=np.int64)
value_y = np.array(value_y, dtype=np.float32)

print("Class distribution:", Counter(policy_y))

X_train, X_val, y_policy_train, y_policy_val, y_value_train, y_value_val = train_test_split(
    X, policy_y, value_y, test_size=0.1, random_state=42
)

X_train = torch.tensor(X_train).to(device)
y_policy_train = torch.tensor(y_policy_train).to(device)
y_value_train = torch.tensor(y_value_train).to(device)

X_val = torch.tensor(X_val).to(device)
y_policy_val = torch.tensor(y_policy_val).to(device)
y_value_val = torch.tensor(y_value_val).to(device)

train_dataset = TensorDataset(X_train, y_policy_train, y_value_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NNUEModel().to(device)
if os.path.exists("model.pt"):
    print("Loading existing model checkpoint...")
    model.load_state_dict(torch.load("model.pt", map_location=device))
else:
    print("No existing checkpoint found. Training from scratch.")

checkpoint_dir = "checkpoints"
version_file = "model_version.txt"
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(version_file):
    with open(version_file) as f:
        model_version = int(f.read())
else:
    model_version = 0

ELO_LOG_PATH = "elo_log.json"
if os.path.exists(ELO_LOG_PATH):
    with open(ELO_LOG_PATH) as f:
        elo_log = json.load(f)
else:
    elo_log = {}

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

best_val_loss = float("inf")

def run_benchmark(new_model_path, old_model_path, num_games=20):
    from benchmark import play_matches
    result = play_matches(new_model_path, old_model_path, num_games)
    score = result['wins_new'] + 0.5 * result['draws']
    return score

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch_x, batch_policy_y, batch_value_y in train_loader:
        optimizer.zero_grad()
        policy_logits, value_pred = model(batch_x)

        loss_policy = policy_loss_fn(policy_logits, batch_policy_y)
        loss_value = value_loss_fn(value_pred, batch_value_y)

        loss = loss_policy + 0.5*loss_value
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        policy_logits_val, value_pred_val = model(X_val)
        val_policy_loss = policy_loss_fn(policy_logits_val, y_policy_val)
        val_value_loss = value_loss_fn(value_pred_val, y_value_val)
        val_acc = (policy_logits_val.argmax(dim=1) == y_policy_val).float().mean()

    print(f"Epoch {epoch}: "
          f"Train Loss={total_loss/len(train_loader):.4f} | "
          f"Val Policy Loss={val_policy_loss.item():.4f} | "
          f"Val Value Loss={val_value_loss.item():.4f} | "
          f"Val Acc={val_acc:.4f}")

    if (val_policy_loss.item() + val_value_loss.item()) < best_val_loss:
        best_val_loss = val_policy_loss.item() + val_value_loss.item()

        # Increment model version
        model_version += 1
        with open(version_file, "w") as f:
            f.write(str(model_version))

        # Save versioned checkpoint
        model_path = os.path.join(checkpoint_dir, f"model_v{model_version}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model: {model_path}")

        prev_model_path = os.path.join(checkpoint_dir, f"model_v{model_version - 1}.pt")
        score = run_benchmark(model_path, prev_model_path, num_games=20)

        # Elo calculation from win rate
        expected_score = 0.5  # Elo baseline (same strength)
        k = 32
        prev_elo = elo_log.get(f"model_v{model_version - 1}.pt", 1500)
        elo_change = k * (score - expected_score)
        real_elo = int(prev_elo + elo_change)

        # Update Elo log
        elo_log[f"model_v{model_version}.pt"] = real_elo
        with open(ELO_LOG_PATH, "w") as f:
            json.dump(elo_log, f, indent=2)

        print(f"Logged Elo: {real_elo}")

        # Promote if Elo improved
        if model_version > 1:
            if real_elo >= prev_elo:
                print("Promoted new model to model.pt")
                torch.save(model.state_dict(), "model.pt")
            else:
                print("New model has lower Elo, not promoting.")
        else:
            torch.save(model.state_dict(), "model.pt")

def self_play_rl(model, games=20, gamma=1.0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for game_num in range(games):
        board = chess.Board()
        game_history = []

        while not board.is_game_over():
            state = extract_features(board)
            legal_moves = list(board.legal_moves)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            policy_logits, _ = model(state_tensor)
            logits = policy_logits.detach().cpu().squeeze()

            # restrict logits to legal moves
            legal_indices = [m.from_square * 64 + m.to_square for m in legal_moves]
            move_logits = logits[legal_indices]
            probs = torch.softmax(move_logits, dim=0)

            # sample move
            move_idx = torch.multinomial(probs, 1).item()
            selected_move = legal_moves[move_idx]

            game_history.append((state, legal_indices[move_idx], board.turn))
            board.push(selected_move)

        # rewards
        result = board.result()
        if result == '1-0':
            reward_white, reward_black = 1, -1
        elif result == '0-1':
            reward_white, reward_black = -1, 1
        else:
            reward_white = reward_black = 0

        # reinforce
        policy_loss = 0
        for t, (state, move_index, is_white) in enumerate(reversed(game_history)):
            reward = (reward_white if is_white else reward_black) * (gamma ** t)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            policy_logits, _ = model(state_tensor)
            log_probs = torch.log_softmax(policy_logits[0], dim=0)
            log_prob = log_probs[move_index]

            policy_loss += -log_prob * reward

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"[RL Game {game_num+1}] Result={result}, Loss={policy_loss.item():.4f}")

    model.eval()

print("\nStarting reinforcement learning via self-play...")
self_play_rl(model, games=RL_GAMES)