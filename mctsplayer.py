import numpy as np
import random
import time
from player import Player, ROWS, COLS, CONNECT_NUMBER
from connect4 import Connect4Board

class Node:
    def __init__(self, board, move, parent, player, connect4board):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_valid_moves(board)
        self.player = player
        self.connect4board = connect4board

    def get_valid_moves(self, board):
        return [col for col in range(board.shape[1]) if board[0, col] == 0]

    def is_terminal(self):
        # Directly use the board state without multiplying by self.player
        winner_board = self.connect4board.check_if_winner(self.board)
        if winner_board is not None:
            return True  # Terminal state (win)
        if not np.any(self.board == 0):
            return True  # Terminal state (draw)
        return False  # Not a terminal state

    def uct_select_child(self):
        # Use the UCT formula to select a child node
        best_value = -float('inf')
        best_node = None
        for child in self.children:
            uct_value = child.wins / child.visits + np.sqrt(2 * np.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node
    
    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.simulate_move(self.board, move, self.player)
        child_node = Node(new_board, move, self, -self.player, self.connect4board)
        self.children.append(child_node)
        return child_node

    def simulate_move(self, board, move, player):
        new_board = board.copy()
        for row in range(board.shape[0] - 1, -1, -1):
            if new_board[row, move] == 0:
                new_board[row, move] = player
                return new_board
        return new_board

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTSPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connect4board = Connect4Board(rows=self.rows, cols=self.cols, connect_number=self.connect_number)

    def select(self, node):
        while not node.is_terminal():
            if node.untried_moves:
                return node.expand()
            else:
                node = node.uct_select_child()
        return node

    def rollout(self, node):
        current_board = node.board.copy()
        current_player = node.player
        while True:
            valid_moves = node.get_valid_moves(current_board)
            if not valid_moves or node.is_terminal(): 
                break
            move = random.choice(valid_moves)
            current_board = node.simulate_move(current_board, move, current_player)
            current_player = -current_player
        return self.game_result(current_board, node.player)

    def backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent
            result = -result  # The result alternates between players

    def play(self, board):
        self.connect4board._board = board

        root = Node(board, move=None, parent=None, player=1, connect4board=self.connect4board)

        start_time = time.time()
        while time.time() - start_time < self.timeout_move:
            node = self.select(root)
            result = self.rollout(node)
            self.backpropagate(node, result)

        # Select the move with the highest win ratio
        best_move = max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else -float('inf')).move
        return best_move

    def simulate_move(self, board, move, player):
        # Use the process_move method from Connect4Board to simulate the move
        is_valid, new_board = self.connect4board.process_move(move, board.copy())
        if not is_valid:
            raise ValueError(f"Simulated move {move} is not valid.")
        return new_board * player  # Return the board with the move made

    def game_result(self, board, player):
        # Use the check_if_winner method from Connect4Board to get the game result
        result_board = self.connect4board.check_if_winner(board)
        if result_board is not None:
            # Check for a win
            if np.any(result_board == 2 * player):
                return 1  # Current player won
            elif np.any(result_board == 2 * -player):
                return -1  # Opponent won
        elif not np.any(board == 0):
            return 0  # Draw
        return None  # Game is ongoing
