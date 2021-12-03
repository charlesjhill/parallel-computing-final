import PySimpleGUI as sg
from multiprocessing import freeze_support

from Agent import RandomAgent, UCTAgent, ParallelTree, ParallelLeaf
from utils import *

import logging

piece_to_color = {
    BLANK: '#FFFDD0',
    RED: 'red',
    YELLOW: 'blue'
}

name_to_agent = {
    'Random': RandomAgent(),
    'MCTS': UCTAgent(max_generations=100000),
    'Root-MCTS': ParallelTree(max_generations=100000),
    'Leaf-MCTS': ParallelLeaf(playout_mult=4, max_generations=100000)
}

if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(level=logging.DEBUG)

    game = Game()
    opponent = ParallelTree(time=3, max_generations=10000)

    # Create the Layout
    board = list(reversed([
        [sg.RButton('', button_color=('white', piece_to_color[BLANK]), key=(i, j), size=(3, 2), auto_size_button=False)
         for j in range(game.num_cols)]
        for i in range(game.num_rows)
    ]))
    board.append([sg.T('Engine Evaluation: '), sg.T('', size=(7, 1), key='out_val'), sg.T('YOUR MOVE', visible=True, key='move_indicator')])
    button_panel = [
        [sg.T('Agent'), sg.Combo(values=list(name_to_agent.keys()), default_value='MCTS', key='agent_option')],
        [sg.Check(text='Engine First', key='engine_first')],
        [sg.T('Engine Time'), sg.Spin(values=list(range(2, 10)), initial_value=3, key='depth')],
        [sg.B('New Game')]
    ]
    layout = [
        [sg.Column(board), sg.VSeparator(), sg.Column(button_panel)]
    ]
    window = sg.Window('CONNECT FOUR', layout)

    opponent_to_move = False

    def redraw_board(game: 'Game', window: 'sg.Window'):
        for j in range(game.num_cols):
            for i in range(game.num_rows):
                window[(i, j)].Update(button_color=('white', piece_to_color[game.board[j][i]]))

        window['out_val'].Update('')
        window.refresh()


    def update_board(game: 'Game', window: 'sg.Window', computer_move_val=None):
        col, row = game.moves[-1]
        window[(row, col)].Update(button_color=('white', piece_to_color[game.board[col][row]]))
        window['move_indicator'].Update(visible=False)
        if computer_move_val is not None:
            window['out_val'].Update(f'{computer_move_val:+.2f}')
            window['move_indicator'].Update(visible=True)

        window.refresh()


    def computer_turn(opponent: 'Agent', game: 'Game', window: 'sg.Window') -> bool:
        # Opponent plays
        move, move_value = opponent.get_move(game)
        game.apply_move(move, game.current_turn)
        update_board(game, window, computer_move_val=move_value)

        winner = game.get_winner()
        if winner is not None:
            sg.popup(f'{piece_to_color[winner]} has won!')
            # Reset Board
            game.reset()
            redraw_board(game, window)
            return True

        return False


    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED,):
            break

        opponent = name_to_agent[values['agent_option']]
        opponent.set_calculation_time(values['depth'])

        if event == 'New Game':
            # Reset Board
            game.reset()
            opponent_to_move = values['engine_first']
            if opponent_to_move:
                window['move_indicator'].Update(visible=False)
            redraw_board(game, window)

        if type(event) == tuple:
            _, col = event
            game.apply_move(col, game.current_turn)
            update_board(game, window)
            opponent_to_move = True

            winner = game.get_winner()
            if winner is not None:
                sg.popup(f'{piece_to_color[winner]} has won!')
                # Reset Board
                game.reset()
                redraw_board(game, window)
                continue

        if opponent_to_move:
            opponent_to_move = False
            if computer_turn(opponent, game, window):
                continue

    window.close()