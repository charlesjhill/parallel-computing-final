import time
from statistics import mean
from multiprocessing import freeze_support
import logging

import tqdm
from Agent import UCTAgent, ParallelTree, SelectionCriteria
from utils import Game

NUM_TRIALS = 5

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    freeze_support()

    one_response_times = []
    one_playouts = []
    two_response_times = []
    two_playouts = []

    results = []

    for i in tqdm.trange(NUM_TRIALS):
        agent_one = UCTAgent(time=2, max_generations=1000000, move_criteria=SelectionCriteria.BEST_CHILD)
        agent_two = ParallelTree(time=2, max_generations=1000000, move_criteria=SelectionCriteria.BEST_CHILD)
        game = Game()

        if i % 2 == 0:
            # Half the time, make agent_two move first
            agent_one_color = -1 * game.current_turn

            start_time = time.time()
            move, _ = agent_two.get_move(game)
            two_response_times.append(time.time() - start_time)
            game.apply_move(move)
        else:
            agent_one_color = game.current_turn

        # Have the agents play the game
        while not game.is_terminal:
            start_time = time.time()
            move, _ = agent_one.get_move(game)
            one_response_times.append(time.time() - start_time)
            game.apply_move(move)

            if game.is_terminal:
                break

            start_time = time.time()
            move, _ = agent_two.get_move(game)
            two_response_times.append(time.time() - start_time)
            game.apply_move(move)

        # Collect summary statistics
        winner = game.get_winner()

        one_playouts.extend(agent_one.playouts_performed)
        two_playouts.extend(agent_two.playouts_performed)

        if winner is None:
            # Draw game
            results.append({'length': game.num_moves, 'winner': None})
        else:
            # One of the agents won
            winner = 'agent_one' if winner == agent_one_color else 'agent_two'
            results.append({'length': game.num_moves, 'winner': winner})

        del agent_two

    # Print summary statistics
    wins_for_one = sum(1 for d in results if d['winner'] == 'agent_one')
    wins_for_two = sum(1 for d in results if d['winner'] == 'agent_two')
    draws = NUM_TRIALS - wins_for_one - wins_for_two

    print(f'Agent One (UCTAgent): {wins_for_one} Wins, {wins_for_two} Losses, {draws} Draws')
    print(f'\t Average Response Time (s): {mean(one_response_times):.4f}')
    print(f'\t Average Playouts Per Move: {mean(one_playouts):.2f}')
    print(f'Agent Two (ParallelTree): {wins_for_two} Wins, {wins_for_one} Losses, {draws} Draws')
    print(f'\t Average Response Time (s): {mean(two_response_times):.4f}')
    print(f'\t Average Playouts Per Move: {mean(two_playouts):.2f}')

