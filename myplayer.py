import numpy as np
import random
import time
from player import Player

class MyPlayer(Player):
    def __init__(self, rows, cols, connect_number, timeout_setup, timeout_move, max_invalid_moves, cylinder):
        self.rows = rows
        self.cols = cols
        self.connect_number = connect_number
        self.timeout_setup = timeout_setup
        self.timeout_move = timeout_move
        self.max_invalid_moves = max_invalid_moves
        self.cylinder = cylinder
        self.start_time = 0

    def setup(self, piece_color):
        self.piece_color = piece_color

    def get_valid_moves(self, board):
        # Return a list of column indices where a new piece can be placed
        return [col for col in range(self.cols) if board[0, col] == 0]

    def simulate_move(self, board, move, maximizingPlayer):
        # print(f'Maximizing Player: {maximizingPlayer}')
        # Create a copy of the board to simulate the move without altering the original
        new_board = board.copy()
        # Check if the move is within the column range and the column is not full
        if isinstance(move, (int, np.int32, np.int64)) and 0 <= move < self.cols:
            n_spots = sum(new_board[:, move] == 0)  # Count empty spots in the column
            if n_spots:
                # Place the piece for the current player at the lowest empty spot
                new_board[n_spots - 1, move] = 1 if maximizingPlayer else -1
        return new_board

    def heuristic_evaluation(self, board):
        weights = {2: 10, 3: 30, 4: 90}  # Weights for sequences of length 2, 3, and 4
        blocked_sequence_penalty = -50  # Penalty for blocked sequences
        score = 0
        board_extended = np.concatenate((board, board[:, :self.connect_number-1]), axis=1) if self.cylinder else board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]  # All 8 directions

        for row in range(self.rows):
            for col in range(self.cols * (2 if self.cylinder else 1) - (self.connect_number - 1)):
                for d_row, d_col in directions:
                    for length, weight in weights.items():
                        for player in [1, -1]:  # Assuming 1 is the player's piece, -1 is the opponent's
                            extendable, blocked = self.count_sequence(board_extended, row, col, d_row, d_col, player, length)
                            if player == 1:  # Adjusting score for the player
                                score += extendable * weight
                                score += blocked * blocked_sequence_penalty
                            else:  # Adjusting score for the opponent
                                score -= extendable * weight
                                score -= blocked * blocked_sequence_penalty

        return score

    def count_sequence(self, board, row, col, d_row, d_col, player, length):
        extendable_seq = 0
        blocked_seq = 0
        for l in range(length):
            current_row = row + d_row * l
            current_col = (col + d_col * l) % self.cols  # Handle cylindrical wrap for columns
            if not (0 <= current_row < self.rows) or board[current_row, current_col] != player:
                return (0, 0)  # If out of bounds or not matching piece, not a valid sequence

        # Check for an open or blocked end after the sequence
        next_row = current_row + d_row
        next_col = (current_col + d_col) % self.cols
        if 0 <= next_row < self.rows:
            if board[next_row, next_col] == 0:
                extendable_seq = 1
            elif board[next_row, next_col] == -1*(player):  # represents the opposite player
                blocked_seq = 1

        # Similarly, check the start of the sequence for an open or blocked end
        prev_row = row - d_row
        prev_col = (col - d_col) % self.cols
        if 0 <= prev_row < self.rows:
            if board[prev_row, prev_col] == 0:
                extendable_seq = 1
            elif board[prev_row, prev_col] == -1*(player):
                blocked_seq = 1

        return (extendable_seq, blocked_seq)


    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        if time.time() - self.start_time > self.timeout_move - 0.1: 
            return 0, None  # Return a neutral value and no move if timeout is near
        
        # Get all valid moves for the current board state
        valid_moves = self.get_valid_moves(board)
        
        # Base case: no valid moves or maximum depth reached
        if not valid_moves or depth == 0:
            # Evaluate the heuristic value of the board
            return self.heuristic_evaluation(board), None

        if maximizingPlayer:
            # Initialize the maximum evaluation score and best move
            maxEval = float('-inf')
            best_move = random.choice(valid_moves)  # Choose a random move as the best move initially
            for move in valid_moves:
                # Simulate the move for the maximizing player
                new_board = self.simulate_move(board, move, maximizingPlayer)
                # Recursively call minimax for the minimizing player
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                # Update maxEval and best_move if a better evaluation is found
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                # Update alpha
                alpha = max(alpha, eval)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return maxEval, best_move
        else:
            # Initialize the minimum evaluation score and best move for minimizing player
            minEval = float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                # Simulate the move for the minimizing player
                new_board = self.simulate_move(board, move, maximizingPlayer)
                # Recursively call minimax for the maximizing player
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                # Update minEval and best_move if a better evaluation is found
                if eval < minEval:
                    minEval = eval
                    best_move = move
                # Update beta
                beta = min(beta, eval)
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            return minEval, best_move

    def play(self, board):
        self.start_time = time.time()  # Record the start time
        _, best_move = self.minimax(board, 3, float('-inf'), float('inf'), True)
        return best_move
