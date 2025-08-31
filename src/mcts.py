import math
import random
import threading
import chess
from evaluation import cached_batched_evaluate, extract_features, get_model
import torch
import numpy as np
from engine import load_default_model as get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.prior = 1.0
        self.virtual_loss = 0
        self.lock = threading.Lock()

    def is_expanded(self):
        return bool(self.children)

    def backpropagate(self, value):
        with self.lock:
            self.visits += 1
            self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)

    def apply_virtual_loss(self, loss=1):
        with self.lock:
            self.virtual_loss += loss
        if self.parent:
            self.parent.apply_virtual_loss(loss)

    def revert_virtual_loss(self, loss=1):
        with self.lock:
            self.virtual_loss -= loss
        if self.parent:
            self.parent.revert_virtual_loss(loss)


def uct_score(child, total_visits, c_puct):
    q_value = 0 if child.visits == 0 else child.total_value / child.visits
    u_value = c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
    return q_value + u_value - child.virtual_loss


def select_leaf(node, c_puct):
    current = node
    while current.is_expanded() and not current.board.is_game_over():
        total_visits = sum(child.visits for child in current.children.values())
        current = max(
            current.children.values(),
            key=lambda child: uct_score(child, total_visits, c_puct)
        )
    return current


def expand_with_prior(node):
    model = get_model()
    board = node.board
    legal_moves = list(board.legal_moves)

    # Extract features
    features = extract_features(board)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(input_tensor)
        policy_logits = policy_logits.squeeze(0).cpu()
        policy_probs = torch.softmax(policy_logits, dim=0).numpy()

    for move in legal_moves:
        new_board = board.copy()
        new_board.push(move)
        move_index = move.from_square * 64 + move.to_square
        prior = policy_probs[move_index] if 0 <= move_index < len(policy_probs) else 1e-8
        child = Node(new_board, parent=node, move=move)
        child.prior = prior
        node.children[move] = child


def lazy_expand(node):
    legal_moves = list(node.board.legal_moves)
    random.shuffle(legal_moves)
    for move in legal_moves:
        if move not in node.children:
            new_board = node.board.copy()
            new_board.push(move)
            child = Node(new_board, parent=node, move=move)
            node.children[move] = child
            return child
    return None


def add_dirichlet_noise(node, alpha=0.3, epsilon=0.25):
    legal_moves = list(node.board.legal_moves)
    if not legal_moves:
        return
    noise = np.random.dirichlet([alpha] * len(legal_moves))
    for i, move in enumerate(legal_moves):
        if move in node.children:
            node.children[move].prior = (
                (1 - epsilon) * node.children[move].prior + epsilon * noise[i]
            )


def select_best_move(board, evaluate_fn=cached_batched_evaluate, sims=100, c_puct=1.4, stop_event=None):
    root = Node(board.copy())

    expand_with_prior(root)
    add_dirichlet_noise(root)

    def simulate():
        nonlocal root
        node = select_leaf(root, c_puct)
        node.apply_virtual_loss()

        if not node.board.is_game_over():
            expand_with_prior(node)

        value = evaluate_fn(node.board)
        if isinstance(value, torch.Tensor):
            value = value.item()
        value = float(torch.tanh(torch.tensor(value)))

        node.backpropagate(value)
        node.revert_virtual_loss()

    for _ in range(sims):
        if stop_event and stop_event.is_set():
            break
        simulate()

    if not root.children:
        return random.choice(list(board.legal_moves))

    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

