# Connect 4 Game

![Game Animation](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)


This repository contains code for the Connect 4 game. It provides the interface for each player to program game logic, and then for players to be pitted against each other. 

## Rules

1. The board is a 6-row, 7-column grid erected vertically.
2. Each column is a shaft, such that a piece dropped in a column will fall to the bottom (or on top of the previous piece.)
3. Players take turns choosing which column to drop their piece in.
4. The objective of the game is to drop pieces such that and 4 of your pieces are connected horizintally, vertically, or diagonally first.
5. If there are no more empty spots left, and no player has connected 4 of their pieces, the game is a draw.


## Variations of the Game
The board can be configured for any number of columns and rows, as well as the number of disks you have to connect to win the game. In addition, a cylinder option is available, connecting the left-hand side of the board to the right-hand side.


## Installing
The code requires python=3.9 numpy. You can pip install or conda install these packages. It is recommended to use a virtual environment.

## Interface
This repository provides the Player class which includes their game logic. It is in turn used by the Connect4Board class to run the game.

The Player class has two methods which may be overridden:

setup() is called at the beginning of the game, and may be used to set up any game logic (loading stuff etc.). It is a timed method, and taking too long will cause the player to lose by default.
play(self, board: np.ndarray) -> int takes the current state of the board, and returns the column index to put the piece in. If the move is invalid, the player loses by default. A move is invalid if the column index is out of bounds, or if the column has no more space left.
When writing up your player, you may subclass the Player class, or write your own, but with these method signatures.

The Connect4Board class plays matches between two players. It has the play(p1, p2) -> str, str, list[int] method that returns the winner, reason for win, and the list of moves as a list of column indices.

p1, p2 are the string names of the modules containing the player object.

* If you have your own Player class in a file myplayer.py in the working directory, you can simply pass myplayer.
* If your player is named something else, then specify the class name like myplayer/Playa
* If the player is in a nested module. For example if you'd need to write from players.simple import Dumbo, then specify the player as players.simple/Dumbo.

A Jupiter Notebook, play.ipynb, is provided along with some dummy players to learn how to use the Connect4Board class to set up a game between two players.


## Minimax & Alpha-Beta Search
For Part 1 of the connect 5 assignment, I implemented the  minimax algorithm with alpha-beta pruning. The minimax function is a recursive algorithm that evaluates the game state to determine the best possible move for a player. It alternates between maximizing and minimizing players, assuming that both players play optimally. The function uses alpha-beta pruning to skip the evaluation of branches that cannot possibly influence the final decision. 

### Parameters
* board: The current state of the game board as a 2D array or matrix. Each element of the array represents a cell on the game board.
* depth: The maximum depth of the game tree to explore. A larger depth results in a more thorough search but increases computation time.
* alpha: The best already explored option along the path to the root for the maximizer. Initially, this is set to negative infinity.
* beta: The best already explored option along the path to the root for the minimizer. Initially, this is set to positive infinity.
* maximizingPlayer: A boolean value indicating whether the current move is for the maximizing player (True) or the minimizing player (False).

### Return Value
The minimax function returns a tuple containing two elements:

* The first element is the score of the board, representing the best achievable outcome from the current game state for the player making the move.
* The second element is the best move to achieve that outcome, represented as an index or coordinate on the game board.

### Helper Functions
* get_valid_moves(board): Returns a list of all valid moves for the current player given the state of the game board.
* simulate_move(board, move, maximizingPlayer): Returns a new game board state after applying the given move for the current player.
* heuristic_evaluation(board): Evaluates the game board and returns a score based on the current state from the perspective of the maximizing player.

## Heuristic Function
I used this [link](https://www.cs.cornell.edu/boom/2001sp/Anvari/Anvari.htm) as a resource when thinking of my heuristic function.

The function scores the board by assessing the number of contiguous lines of pieces for both the player and its opponent. Each line's score is determined based on its length, with the following rationale:

* Sequences are weighted to reflect their potential impact on the game. Shorter sequences have lower weights, while longer sequences, closer to achieving a win, are given exponentially increasing importance.
* The evaluation prioritizes open sequences where additional pieces can be added, recognizing that sequences blocked by the opponent's pieces offer limited strategic value moving forward.
* The cylindrical dynamic of the board is taken into account when measuring the count sequences. 

Weights are assigned as follows:

* 10 points for two in a row.
* 30 points for three in a row.
* 90 points for four in a row.
* -50 for blocked rows. 

This is a basic heuristic function which performs decently well against the SmartRandom player. An example of this play can be seen in play.ipynb. Additional adjustments will be needed before the competition. 