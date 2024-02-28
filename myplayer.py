import numpy as np
import random
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

    def setup(self, piece_color):
        self.piece_color = piece_color

    def get_valid_moves(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]
    
    def simulate_move(self, board, move, maximizingPlayer):
        new_board = board.copy() 
        if isinstance(move, (int, np.int32, np.int64)) and 0 <= move < self.cols:
            n_spots = sum(new_board[:, move] == 0)
            if n_spots:
                new_board[n_spots - 1, move] = 1 if maximizingPlayer else -1
        return new_board

    def heuristic_evaluation(self, board):
        score = 0
        board_to_check = np.concatenate((board, board[:, :self.connect_number-1]), axis=1) if self.cylinder else board
        
        for row in range(self.rows):
            for col in range(self.cols * (2 if self.cylinder else 1) - (self.connect_number - 1)):
                # Check sequences for both player and opponent
                score += self.evaluate_sequence(board_to_check, row, col, True) - self.evaluate_sequence(board_to_check, row, col, False)
        return score

    def evaluate_sequence(self, board, row, col, player):
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)] # Including all 8 directions for thoroughness
        sequence_lengths = [2, 3, 4] # Lengths of interest

        for length in sequence_lengths:
            for d_row, d_col in directions:
                if self.count_sequence(board, row, col, d_row, d_col, player, length):
                    # Update scoring based on sequence length
                    if length == 2:
                        score_increment = 10
                    elif length == 3:
                        score_increment = 50
                    else:  # length == 4
                        score_increment = 100
                    # If player's or opponent's sequence increment/decrement score
                    score += score_increment * (1 if player else -1)
        return score

    def count_sequence(self, board, row, col, d_row, d_col, player, length):
        total_seq = 0  # To keep track of sequences found
        for l in range(length):
            # Calculate current position with wrap around for cylindrical logic
            current_row = row + d_row*l
            current_col = (col + d_col*l) % self.cols  # Wrap around using modulo for columns
            
            # Out of bounds check for rows only, columns are handled by wrap around
            if not (0 <= current_row < self.rows):
                return False
            
            # Checking sequence continuity based on player
            if player:
                if board[current_row, current_col] != 1:
                    return False
            else:
                if board[current_row, current_col] != -1:
                    return False

        # Now, check for an open end considering cylindrical nature
        # Forward direction
        next_col_forward = (col + d_col*length) % self.cols
        if 0 <= row + d_row*length < self.rows and board[row + d_row*length, next_col_forward] == 0:
            total_seq += 1

        # Backward direction (considering wrap around)
        next_col_backward = (col - d_col) % self.cols
        if 0 <= row - d_row < self.rows and board[row - d_row, next_col_backward] == 0:
            total_seq += 1

        return total_seq > 0  # Return True if there's at least one open end



    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:  # No valid moves means game is over
            return 0, None
        
        if depth == 0:
            return self.heuristic_evaluation(board), None
       
        if maximizingPlayer:
            maxEval = float('-inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(board, move, maximizingPlayer)
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval, best_move
        else:
            minEval = float('inf')
            best_move = random.choice(valid_moves)
            for move in valid_moves:
                new_board = self.simulate_move(board, move, maximizingPlayer) 
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval, best_move

    def play(self, board):
        _, best_move = self.minimax(board, 3, float('-inf'), float('inf'), True)
        return best_move
