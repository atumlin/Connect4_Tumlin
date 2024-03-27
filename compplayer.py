import numpy as np
import random
import time
from player import Player, ROWS, COLS, CONNECT_NUMBER

MAX_DEPTH = 2

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=1, cols=COLS, rows=ROWS,connect_number=CONNECT_NUMBER, cylinder=True):
        self.board = np.copy(board)
        self.parent = parent
        self.move = move
        self.player = player  # 1 for one player, -1 for the opponent
        self.cols = cols
        self.rows = rows
        self.connect_number = connect_number
        self.cylinder = cylinder
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_valid_moves(self.board)
    
    def setup(self, piece_color):
        self.piece_color = piece_color

    def get_valid_moves(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]

    def simulate_move(self, board, move, player):
        new_board = np.copy(board)
        for row in range(new_board.shape[0] - 1, -1, -1):
            if new_board[row, move] == 0:
                new_board[row, move] = player
                break
        return new_board
    
    # Method utilizied from connect4.py
    def check_config(self, board, config):
        b1, b2 = board.shape
        c1, c2 = config.shape
        for i in range(b1 - c1 + 1):
            for j in range(b2 - c2 + 1):
                if np.sum(board[i:i+c1, j:j+c2] * config) == self.connect_number:
                    board[i:i+c1, j:j+c2][config == 1] = 2
                    if self.cylinder:
                        board[:, :self.connect_number-1][board[:, -self.connect_number+1:] == 2] = 2
                        board = board[:, :-self.connect_number+1]
                    return True, board
        return False, board

    # Method utilizied from connect4.py
    def check_if_winner(self, board):
        if self.cylinder:
            board = np.concatenate((board, board[:, :self.connect_number-1]), axis=1)

        # Horizontal checking
        config = np.ones((1, self.connect_number), dtype=int)
        end, board = self.check_config(board, config)
        if end:
            return board

        # Vertical checking
        config = np.ones((self.connect_number, 1), dtype=int)
        end, board = self.check_config(board, config)
        if end:
            return board

        # Diagonal checking
        config = np.eye(self.connect_number, dtype=int)
        end, board = self.check_config(board, config)
        if end:
            return board

        # Anti-diagonal checking
        config = np.fliplr(np.eye(self.connect_number, dtype=int))
        end, board = self.check_config(board, config)
        if end:
            return board

        return None

    # Not utilized when using heuristic evaluation 
    def simulate_random_playout(self):
        current_board = self.board
        current_player = self.player
        while True:
            valid_moves = self.get_valid_moves(current_board)
            if not valid_moves:  # Draw
                return 0

            move = random.choice(valid_moves)
            current_board = self.simulate_move(current_board, move, current_player)
            winner_board = self.check_if_winner(current_board)
            if winner_board is not None:  # A player wins
                if np.any(winner_board == 2) and current_player == self.player:
                    return 1  # Current node's player wins
                else:
                    return -1  # Opponent wins

            current_player *= -1  # Switch player
    
    # Used when using heuristic evaluation
    def simulate_heuristic_playout(self, depth=0):
        if depth >= MAX_DEPTH:
            return 0
        current_board = self.board
        current_player = self.player
        while True:
            valid_moves = self.get_valid_moves(current_board)
            if not valid_moves:  # Draw
                return 0

            heuristic_scores = []
            for move in valid_moves:
                simulated_board = self.simulate_move(current_board, move, current_player)
                score = self.heuristic_evaluation(simulated_board * current_player)  # Multiply by player to evaluate from the current player's perspective
                heuristic_scores.append(score)
            
            # Choose the move with the best score for the current player
            best_move_index = np.argmax(heuristic_scores) if current_player == 1 else np.argmin(heuristic_scores)
            best_move = valid_moves[best_move_index]
            current_board = self.simulate_move(current_board, best_move, current_player)
            
            winner_board = self.check_if_winner(current_board)
            if winner_board is not None:  # A player wins
                if np.any(winner_board == 2) and current_player == self.player:
                    return 1  # Current node's player wins
                else:
                    return -1  # Opponent wins
            
            current_player *= -1  # Switch player
            
            # Update for next iteration
            depth += 1

    
    def heuristic_evaluation(self, board):
        weights = {3: 10, 4: 100}  # Simplified weights
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Main directions: horizontal, vertical, diagonal

        for player in [1, -1]:  # Loop for both players
            for row in range(self.rows):
                for col in range(self.cols):
                    for d_row, d_col in directions:
                        sequence_count = self.simple_count_sequence(board, row, col, d_row, d_col, player)
                        # Update score based on found sequences
                        score += sequence_count * (weights[3] if sequence_count == 3 else 0) + (weights[4] if sequence_count >= 4 else 0)

        return score 

    def simple_count_sequence(self, board, row, col, d_row, d_col, player):
        """Counts the maximum sequence length in one direction."""
        max_length = 0
        sequence_length = 0

        for l in range(1, self.connect_number):  # Limit to max connect_number - 1 steps
            current_row = row + d_row * l
            current_col = col + d_col * l

            if 0 <= current_row < self.rows and 0 <= current_col < self.cols and board[current_row, current_col] == player:
                sequence_length += 1
                max_length = max(max_length, sequence_length)
            else:
                break  # Stop if sequence is broken

        return max_length


    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_board = self.simulate_move(self.board, move, self.player)
        child_node = MCTSNode(new_board, parent=self, move=move, player=-self.player, cols=self.cols,
                              connect_number=self.connect_number, cylinder=self.cylinder)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=1.4):
        choices_weights = [(child.wins / child.visits) + c_param * np.sqrt(2 * np.log(self.visits) / child.visits) for child in self.children]
        return self.children[np.argmax(choices_weights)]

class CompPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece_color_numeric = None  # Will hold +1 or -1 based on piece_color

    def setup(self, piece_color):
        super().setup(piece_color) 
        # Convert piece_color to numeric value
        self.piece_color_numeric = 1 if piece_color == '+' else -1

    def play(self, board):
        if self.piece_color_numeric is None:
            raise ValueError("Player color not set. Please call setup before play.")
        
        # Initialize the root of the MCTS tree with the current state
        root = MCTSNode(board, player=self.piece_color_numeric, cols=self.cols, connect_number=self.connect_number, cylinder=self.cylinder)
        
        NUM_SIMULATIONS = 20  # Adjust this number based on performance testing and time constraints
    
        for _ in range(NUM_SIMULATIONS):
            node = root
            depth = 0  # Initialize depth for each new simulation

            # Selection
            while not node.untried_moves and node.children:
                node = node.best_child()
                depth += 1
            
            # Expansion
            if node.untried_moves:
                node = node.expand()
            
            # Simulation
            outcome = node.simulate_heuristic_playout(depth)
            
            # Backpropagation
            node.backpropagate(outcome)
        
        best_move = root.best_child(c_param=0).move
        return best_move