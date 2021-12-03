import copy

RED = 1
YELLOW = -1
BLANK = 0


class Game(object):
    num_cols = 7
    num_rows = 6

    def __init__(self):
        self.height = [0]*self.num_cols
        self.board = []
        for i in range(self.num_cols):
            self.board.append(copy.deepcopy([BLANK]*self.num_rows))
        self.current_turn = RED
        self.player_to_piece_map = {
            RED: [],
            YELLOW: []
        }
        self.moves = []
        self.move_priority_list = [3, 2, 4, 1, 5, 0, 6] if self.num_cols == 7 else range(self.num_cols)
        self._hash = None

    def __hash__(self):
        return self.hash

    def can_play(self, col: int) -> bool:
        return self.height[col] < self.num_rows

    def reset(self):
        self.current_turn = RED
        self.player_to_piece_map = {
            RED: [],
            YELLOW: []
        }
        self.moves = []
        self.height = [0] * self.num_cols
        self.board = []
        for i in range(self.num_cols):
            self.board.append(copy.deepcopy([BLANK] * self.num_rows))

    def get_moves(self):
        gen = (i for i in self.move_priority_list if self.can_play(i))

        if self.is_symmetric:
            return [col for col in gen if col <= self.num_cols//2]

        return [i for i in gen]

    def apply_move(self, column, player=None):
        piece_height = self.height[column]
        self.board[column][piece_height] = player if player is not None else self.current_turn
        self.moves.append((column, piece_height))
        self.player_to_piece_map[self.current_turn].append((column, piece_height))
        self.height[column] += 1
        self.current_turn *= -1
        self._hash = None

    def is_winning_move(self, col: int, man_col_height=None, man_current_player=None):
        current_player = man_current_player if man_current_player is not None else self.current_turn
        col_height = man_col_height if man_col_height is not None else self.height[col]
        # Check vertical alignment
        if (col_height >= 3
                and self.board[col][col_height-3:col_height] == [current_player]*3):
            return True

        # Check the horizontal (dy = 0), and the two diagonal directions (dy = -1 and 1)
        for dy in [-1, 0, 1]:
            num_consecutive_stones = 0
            for dx in [-1, 1]:  # Count continuous stones of current player on the left, then right of the player column
                x, y = col + dx, col_height + dx*dy
                # Search along the determined direction for consecutive stones
                while 0 <= x < self.num_cols and 0 <= y < self.num_rows and self.board[x][y] == current_player:
                    num_consecutive_stones += 1
                    x += dx
                    y += dx*dy
            if num_consecutive_stones >= 3:
                return True

        return False

    def get_all_ranges(self):
        for col in range(self.num_cols):
            for start_row in range(self.num_rows-3):
                yield [(col, start_row), (col, start_row+1), (col, start_row+2), (col, start_row+3)]

        for row in range(self.num_rows):
            for start_col in range(self.num_cols-3):
                yield [(start_col, row), (start_col+1, row), (start_col+2, row), (start_col+3, row)]

        # 3.1 Up and to the right
        for start_row in range(self.num_rows-3):
            for start_col in range(self.num_cols - 3):
                yield [(start_col, start_row), (start_col + 1, start_row + 1),
                       (start_col + 2, start_row + 2), (start_col + 3, start_row + 3)]

        # 3.2 Down and to the right
        for start_row in range(3, self.num_rows):
            for start_col in range(self.num_cols - 3):
                yield [(start_col, start_row), (start_col + 1, start_row - 1),
                       (start_col + 2, start_row - 2), (start_col + 3, start_row - 3)]

    def get_winner(self):
        # Function to determine if the game has a winner
        if self.num_moves < 7:
            return None

        col, piece_height = self.moves[-1]

        if self.is_winning_move(col, piece_height, -1*self.current_turn):
            return -1*self.current_turn

        return None

    def get_hash(self):
        return self.hash

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash((frozenset(self.player_to_piece_map[RED]), frozenset(self.player_to_piece_map[YELLOW])))
        return self._hash

    @property
    def num_moves(self):
        return len(self.moves)

    @property
    def is_terminal(self):
        return (self.get_winner() is not None) or (self.num_moves == self.num_cols * self.num_rows)

    @property
    def is_symmetric(self):
        for i in range(self.num_cols//2):
            if self.board[i] != self.board[self.num_cols-1-i]:
                return False
        return True
