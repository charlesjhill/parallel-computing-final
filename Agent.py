from copy import deepcopy
import datetime
import logging
import multiprocessing as mp
import os

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial, reduce
from math import sqrt, log
from typing import Tuple, List, Set

from numpy.random import default_rng, Generator

from utils import Game


class Agent(object):

    @abstractmethod
    def get_move(self, board_state: 'Game') -> Tuple[int, float]:
        pass

    def set_calculation_time(self, t):
        pass


class RandomAgent(Agent):
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = default_rng(seed)

    def get_move(self, board_state: 'Game'):
        return self.rng.choice(board_state.get_moves()), 0

    def __repr__(self):
        return 'RandomAgent'

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['rng'] = default_rng(self.seed)
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


@dataclass
class Node:
    """Class for storing a 'node' of the MCTS tree"""
    state: Game
    player_color: int
    parent: 'Node' = None
    fully_expanded: bool = False
    child_moves: Set[int] = field(default_factory=set)
    children: List['Node'] = field(default_factory=list)
    num_visits: int = 0
    score: float = 0.0

    @property
    def action(self):
        """
        Yield the move that generated this node
        :return: An integer move
        """
        return self.state.moves[-1][0]

    def add_child(self, move: int) -> 'Node':
        """
        Add a child node created by making a move the current state
        :param move: A valid integer move
        :return: The child node object
        """
        state_cpy = deepcopy(self.state)
        state_cpy.apply_move(move)

        child_node = Node(state_cpy, state_cpy.current_turn, parent=self)
        self.children.append(child_node)
        self.child_moves.add(move)

        return child_node

    def to_move_data(self):
        data_dict = {}
        for child in self.children:
            data_dict[child.action] = (child.score, child.num_visits)
        return data_dict


class SelectionCriteria(Enum):
    BEST_CHILD = auto()
    ROBUST_CHILD = auto()
    BEST_AND_ROBUST = auto()


class UCTAgent(Agent):
    def __init__(self,
                 generations_per_move=100,
                 max_generations=1000,
                 time=600,
                 move_criteria: SelectionCriteria = SelectionCriteria.BEST_CHILD):

        self.generations_per_move = generations_per_move
        self.max_generations = max_generations
        self.random_agent = RandomAgent()
        self.rng: 'Generator' = default_rng()
        self.default_c = 1.0 / sqrt(2)
        self.calculation_time = datetime.timedelta(seconds=time)
        self.playouts_performed = []

        self.move_criteria = move_criteria
        self.move_selector = {
            SelectionCriteria.BEST_CHILD: partial(self.best_child, c=0),
            SelectionCriteria.ROBUST_CHILD: self.robust_child,
            SelectionCriteria.BEST_AND_ROBUST: partial(self.best_child, c=0)
        }[move_criteria]

    def __repr__(self):
        return f'UCTAgent(generations_per_move={self.generations_per_move}, max_generations={self.max_generations}, ' \
               f'time={self.calculation_time}, criteria={self.move_criteria})'

    def set_calculation_time(self, seconds):
        self.calculation_time = datetime.timedelta(seconds=seconds)

    def playout(self, state: 'Game', agent_color: int) -> int:
        """Do a random 'playout' starting at the current point in the game"""
        # Make a local copy of the state to work on
        state_copy = deepcopy(state)

        # Simulate the game with random moves from both players
        while not state_copy.is_terminal:
            m, _ = self.random_agent.get_move(state_copy)
            state_copy.apply_move(m)

            if state_copy.is_terminal:
                break

            m, _ = self.random_agent.get_move(state_copy)
            state_copy.apply_move(m)

        # Yield the result
        winner = state_copy.get_winner()
        if winner is None:
            # Draw
            return 0
        else:
            # One of the two sides won
            return winner * agent_color

    def expand(self, node: 'Node') -> 'Node':
        # 1. Select some move we haven't tried yet
        #    Room for optimization here
        untried_moves = [a for a in node.state.get_moves() if a not in node.child_moves]
        untried_move = self.rng.choice(untried_moves)

        # 2. Create a child node and attach it
        child_node = node.add_child(untried_move)

        if len(untried_moves) == 1:
            # This was the last untried move
            node.fully_expanded = True

        return child_node

    def tree_policy(self, root_node: 'Node') -> 'Node':
        v = root_node
        # While the current node is not terminal
        while not v.state.is_terminal:
            if not v.fully_expanded:
                return self.expand(v)
            else:
                v = self.best_child(v)

        return v

    def default_policy(self, leaf_state: 'Game'):
        return self.playout(leaf_state, -1*leaf_state.current_turn)

    def backup(self, v: 'Node', delta):
        while v is not None:
            v.num_visits += 1
            v.score += delta
            delta = -1*delta
            v = v.parent

    def best_child(self, parent: 'Node', c=0.8):
        # Return the child node which maximizes this function:
        best_score = -float('inf')
        log_parent_visits = log(parent.num_visits)
        best_options = []
        for child_node in parent.children:
            child_ucb = (child_node.score / (child_node.num_visits + 0.0001)
                         + c * sqrt(2 * log_parent_visits / (child_node.num_visits + 0.0001)))
            if child_ucb > best_score:
                best_score = child_ucb
                best_options = [child_node]
            elif child_ucb == best_score:
                best_options.append(child_node)

        return self.rng.choice(best_options)

    def robust_child(self, parent: 'Node'):
        most_visits = -float('inf')
        best_options = []
        for child_node in parent.children:
            child_visits = child_node.num_visits
            if child_visits > most_visits:
                most_visits = child_visits
                best_options = [child_node]
            elif child_visits == most_visits:
                best_options.append(child_node)

        return self.rng.choice(best_options)

    def get_move(self, board_state: 'Game') -> Tuple[int, float]:

        # 0. Create a root node, with state "board_state"
        root_node = Node(deepcopy(board_state), board_state.current_turn)

        # While we are within our computation budget....
        #  TODO: Add some early stopping criteria if a dominant move has emerged?
        begin = datetime.datetime.utcnow()
        i = 0
        while datetime.datetime.utcnow() - begin < self.calculation_time and i < self.max_generations:
            # 1. From `board_state`, navigate, via the tree_policy, to some leaf node
            leaf = self.tree_policy(root_node)

            # 2. Perform a playout from a random child of the leaf node, via the default policy
            delta = self.default_policy(leaf.state)

            # 3. Backup the results through the tree structure
            self.backup(leaf, delta)

            i += 1

        logging.debug(f'Serial: {i} moves evaluated')
        self.playouts_performed.append(i)

        # 4. Return the action which results in the best child
        #  TODO: Compare performance versus an agent with "robust child" selection criteria.
        #  Additionally, it has been suggested that if "best child" and "most robust child" disagree, allotting
        #   more compute time can be worthwhile
        best_child = self.move_selector(root_node)
        return best_child.action, (1 + (best_child.score / best_child.num_visits)) / 2.0


def merge_move_data(data1: dict, data2: dict) -> dict:
    new_dict = dict(data1)
    for k, v_tuple in data2.items():
        if k in new_dict:
            new_dict[k] = (new_dict[k][0] + v_tuple[0], new_dict[k][1] + v_tuple[1])
        else:
            new_dict[k] = v_tuple

    return new_dict


class ParallelTree(UCTAgent):
    def __init__(self, num_processes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = os.cpu_count() if num_processes is None else num_processes
        self.pool = None

    def __del__(self):
        if 'pool' in self.__dict__ and self.pool is not None:
            self.pool.close()
            self.pool.terminate()

    def get_move_single(self, board_state: 'Game', start_time):
        # 0. Create a root node, with state "board_state"
        root_node = Node(board_state, board_state.current_turn)

        # While we are within our computation budget....
        #  TODO: Add some early stopping criteria if a dominant move has emerged?
        i = 0
        while datetime.datetime.utcnow() - start_time < self.calculation_time and i < self.max_generations:
            # 1. From `board_state`, navigate, via the tree_policy, to some leaf node
            leaf = self.tree_policy(root_node)

            # 2. Perform a playout from a random child of the leaf node, via the default policy
            delta = self.default_policy(leaf.state)

            # 3. Backup the results through the tree structure
            self.backup(leaf, delta)

            i += 1

        # Put move data in queue
        # output_q.put(root_node.to_move_data())
        return root_node.to_move_data()

    def get_move(self, board_state: 'Game') -> Tuple[int, float]:
        if self.pool is None:
            self.pool = mp.Pool(processes=self.num_processes)

        start_time = datetime.datetime.utcnow()
        results = [self.pool.apply_async(self.get_move_single,
                                         args=(deepcopy(board_state),
                                               start_time))
                   for _ in range(self.num_processes)]

        states = [res.get() for res in results]

        # Merge the move data from each process
        combined_data = reduce(merge_move_data, states)

        # 4. Return the action which results in the best child
        #  TODO: Compare performance versus an agent with "robust child" selection criteria.
        #  Additionally, it has been suggested that if "best child" and "most robust child" disagree, allotting
        #   more compute time can be worthwhile

        # Get the best child
        best_act = -1
        best_score = -1
        for action, (sc, num_moves) in combined_data.items():
            val = sc / num_moves
            if val > best_score:
                best_score = val
                best_act = action

        num_playouts_total = sum(v[1] for v in combined_data.values())
        self.playouts_performed.append(num_playouts_total)
        logging.debug(f'Parallel: {num_playouts_total} moves evaluated')
        logging.debug(f'Parallel: {len([1 for state in states if len(state) != 0])} processes contributed')
        return best_act, (1 + best_score) / 2.0

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class ParallelLeaf(UCTAgent):
    def __init__(self, num_processes=None, playout_mult=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_processes = os.cpu_count() if num_processes is None else num_processes
        self.num_playouts_per = int(playout_mult * self.num_processes)
        self.pool = None

    def __del__(self):
        if 'pool' in self.__dict__ and self.pool is not None:
            self.pool.close()
            self.pool.terminate()

    def default_policy(self, leaf_state: 'Game'):
        async_results = [self.pool.apply_async(self.playout,
                                               args=(leaf_state, -1 * leaf_state.current_turn))
                         for _ in range(self.num_playouts_per)]

        return [res.get() for res in async_results]

    def backup(self, v: 'Node', deltas: list):
        n_v = len(deltas)
        delta = sum(deltas)

        while v is not None:
            v.num_visits += n_v
            v.score += delta
            delta = -1*delta
            v = v.parent

    def get_move_single(self, board_state: 'Game', start_time):
        # 0. Create a root node, with state "board_state"
        root_node = Node(board_state, board_state.current_turn)

        # While we are within our computation budget....
        #  TODO: Add some early stopping criteria if a dominant move has emerged?
        i = 0
        while datetime.datetime.utcnow() - start_time < self.calculation_time and i < self.max_generations:
            # 1. From `board_state`, navigate, via the tree_policy, to some leaf node
            leaf = self.tree_policy(root_node)

            # 2. Perform a playout from a random child of the leaf node, via the default policy
            delta = self.default_policy(leaf.state)

            # 3. Backup the results through the tree structure
            self.backup(leaf, delta)

            i += 1

        # Put move data in queue
        return root_node.to_move_data()

    def get_move(self, board_state: 'Game') -> Tuple[int, float]:
        if self.pool is None:
            self.pool = mp.Pool(processes=self.num_processes)

        start_time = datetime.datetime.utcnow()
        combined_data = self.get_move_single(deepcopy(board_state), start_time)

        # 4. Return the action which results in the best child
        #  TODO: Compare performance versus an agent with "robust child" selection criteria.
        #  Additionally, it has been suggested that if "best child" and "most robust child" disagree, allotting
        #   more compute time can be worthwhile

        # Get the best child
        best_act = -1
        best_score = -1
        for action, (sc, num_moves) in combined_data.items():
            val = sc / num_moves
            if val > best_score:
                best_score = val
                best_act = action

        num_playouts_total = sum(v[1] for v in combined_data.values())
        self.playouts_performed.append(num_playouts_total)
        logging.debug(f'Leaf: {num_playouts_total} moves evaluated')
        return best_act, (1 + best_score) / 2.0

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict:
            del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
