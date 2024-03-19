import numpy as np
import random
from player import Player
from myplayer import MyPlayer  # Assuming your MyPlayer class is in a file named myplayer.py

class MCTSNode:
    def __init__(self, board, move=None, parent=None, player=1, cols=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.cols = cols
        self.untried_moves = self.get_valid_moves(board)
        self.player = player  # 1 or -1

    def get_valid_moves(self, board):
        return [col for col in range(board.shape[1]) if board[0, col] == 0]
    
    def simulate_move(self, board, move, maximizingPlayer):
        # Create a copy of the board to simulate the move without altering the original
        new_board = board.copy()
        # Check if the move is within the column range and the column is not full
        if isinstance(move, (int, np.int32, np.int64)) and 0 <= move < self.cols:
            n_spots = sum(new_board[:, move] == 0)  # Count empty spots in the column
            if n_spots:
                # Place the piece for the current player at the lowest empty spot
                new_board[n_spots - 1, move] = 1 if maximizingPlayer else -1
        return new_board

    def expand(self):
        # Choose a random untried move
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_board = self.simulate_move(self.board, move, True)
        child_node = MCTSNode(new_board, move=move, parent=self, player=-self.player)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        # UCT formula to select the best child
        choices_weights = [
            (child.wins / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        # Random rollout policy
        return possible_moves[np.random.randint(len(possible_moves))]

    def rollout(self):
        current_rollout_board = self.board
        while MyPlayer.get_valid_moves(current_rollout_board):
            possible_moves = MyPlayer.get_valid_moves(current_rollout_board)
            move = self.rollout_policy(possible_moves)
            current_rollout_board = self.simulate_move(current_rollout_board, move, True)
            if MyPlayer.check_for_win(current_rollout_board, move):  # You need to implement this
                return 1
        return 0  # Draw or loss

    def backpropagate(self, result):
        self.update(result)
        if self.parent:
            self.parent.backpropagate(-result)

class MCTSPlayer(MyPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
    def piece_color_to_num(self, piece_color):
        return 1 if piece_color == '+' else -1

    def play(self, board):
        numeric_piece_color = self.piece_color_to_num(self.piece_color)
        root = MCTSNode(board, player=numeric_piece_color, cols=self.cols)
        for _ in range(self.timeout_move):  # Use the move timeout as the number of iterations
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
            # Rollout
            rollout_result = node.rollout()
            # Backpropagation
            node.backpropagate(rollout_result)

        best_move = root.best_child(c_param=0).move
        return best_move

    # TODO - implement this function
    def check_for_win(self,board,move):
        return True