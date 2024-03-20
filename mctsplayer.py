import numpy as np
import random
from player import Player, ROWS, COLS, CONNECT_NUMBER

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=1, cols=COLS, connect_number=CONNECT_NUMBER, cylinder=True):
        self.board = np.copy(board)
        self.parent = parent
        self.move = move
        self.player = player  # 1 for one player, -1 for the opponent
        self.cols = cols
        self.connect_number = connect_number
        self.cylinder = cylinder
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_valid_moves(self.board)

    def get_valid_moves(self, board):
        return [col for col in range(self.cols) if board[0, col] == 0]

    def simulate_move(self, board, move, player):
        new_board = np.copy(board)
        for row in range(new_board.shape[0] - 1, -1, -1):
            if new_board[row, move] == 0:
                new_board[row, move] = player
                break
        return new_board

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

class MCTSPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.piece_color_numeric = None  # Will hold +1 or -1 based on piece_color

    def setup(self, piece_color):
        super().setup(piece_color)  # Call the parent setup if needed
        # Convert piece_color to numeric value
        self.piece_color_numeric = 1 if piece_color == '+' else -1

    def play(self, board):
        # Ensure that piece_color_numeric is set
        if self.piece_color_numeric is None:
            raise ValueError("Player color not set. Please call setup before play.")
        
        # Initialize the root of the MCTS tree with the current state
        root = MCTSNode(board, player=self.piece_color_numeric, cols=self.cols, connect_number=self.connect_number, cylinder=self.cylinder)
        
        # Simulate within the given time or move limit
        for _ in range(self.timeout_move):
            node = root
            # Selection
            while not node.untried_moves and node.children:
                node = node.best_child()
            
            # Expansion
            if node.untried_moves:
                node = node.expand()
            
            # Simulation
            outcome = node.simulate_random_playout()
            
            # Backpropagation
            node.backpropagate(outcome)
        
        # After completing the simulations, select the best move at the root
        best_move = root.best_child(c_param=0).move
        return best_move
