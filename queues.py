from collections import deque
from typing import List

import numpy as np


class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def extend(self, elements):
        for element in elements:
            self.append(element)

    def to_list(self):
        return self[:]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        else:
            if idx >= 0:
                if idx > len(self) - 1:
                    raise ValueError("Incorrect index- out of range")
                return self.data[(self.start + idx) % len(self.data)]
            else:
                if idx < -len(self.data):
                    raise ValueError("Incorrect index- out of range")
                return self.data[(self.end - idx + 1) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ExperienceReplay:
    def __init__(self, size, n_state_frames):
        """n_state_frames determines how much frames in state."""
        # States are saved as separate frames. Each state consists of n_state_frames.
        self.n_state_frames = n_state_frames

        # Buffer for states needs to be bigger. We store there n_state_frames
        # per state and extra frame as future state
        self.states = RingBuf(size)
        self.new_states = RingBuf(size)
        self.actions = RingBuf(size)
        self.rewards = RingBuf(size)
        self.terminate_state = RingBuf(size)

    def __len__(self):
        return len(self.actions)

    def add(self, state: List[np.array], action: np.array, new_state: List[np.array], reward: int,
            terminate: bool):
        # add only new_frame, the rest of them are all already in buffer
        self.states.append(state)
        self.actions.append(action)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.terminate_state.append(terminate)

    def extend(self, states, actions, new_frames, rewards, terminates):
        for s, a, nf, r, t in zip(states, actions, new_frames, rewards, terminates):
            self.add(s, a, nf, r, t)

    def sample_batch(self, n):
        """Returns n samples as np arrays."""
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        # We are counting from 0
        for i in np.random.randint(0, len(self) - 1, n):
            state, action, reward, next_state, terminate = self.get_sample(i)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminate_state.append(terminate)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminate_state)

    def get_sample(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.new_states[idx]
        terminate = self.terminate_state[idx]

        return state, action, reward, next_state, terminate

    def get_all(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        for i in range(len(self)):
            state, action, reward, next_state, terminate = self.get_sample(i)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminate_state.append(terminate)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminate_state)


class TreeNode:
    """Node of unsorted binary tree. """

    def __init__(self, value, parent=None, left=None, right=None):
        self.parent = parent
        self.left = left
        self.right = right
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def parent_exists(self):
        return self.parent is not None

    def update_value(self, value):
        """Updates error value of node and its parents."""
        self._value = value
        if self.parent_exists:
            self.parent.__update_value()

    def __update_value(self):
        """Update error value base on child nodes."""
        self._value = self.left.value + self.right.value
        if self.parent_exists:
            self.parent.__update_value()


class PrioritizedExperienceReplayNode(TreeNode):
    """Node of unsorted binary tree. value of _error field equal to None indices unused node in tree.
    Only leaf nodes can be unused. """

    def __init__(self, value, index=None, parent=None, left=None, right=None):
        super().__init__(value=value, parent=parent, left=left, right=right)
        self.epsilon = 0.0001
        self.index = index

    @property
    def value(self):
        """Returns modified _error value. Returns _error value modified by epsilon value if its leaf node.
        If _error value is None returns 0. In other cases returns 0"""
        if self._value is None:
            return 0
        elif self.is_leaf:
            return self._value + self.epsilon
        else:
            return self._value

    @classmethod
    def from_list(cls, data):
        """Initialize binary tree for given list. Empty nodes should have value of None."""
        lower_nodes = deque()
        upper_nodes = deque()

        for i, error in enumerate(data):
            node = cls(value=error, index=i)
            lower_nodes.append(node)

        nodes = list(lower_nodes)

        # lower_nodes will have len of one when only root node will be left
        while len(lower_nodes) != 1:
            while len(lower_nodes):
                l_node = lower_nodes.pop()
                # if uneven number of nodes in queue
                if len(lower_nodes):
                    r_node = lower_nodes.pop()
                else:
                    r_node = cls(value=None)

                node = cls(value=l_node.value + r_node.value, left=l_node, right=r_node)

                l_node.parent = node
                r_node.parent = node

                upper_nodes.append(node)

            lower_nodes, upper_nodes = upper_nodes, lower_nodes
        return nodes, lower_nodes.pop()

    @classmethod
    def init_tree_with_n_leafs(cls, n):
        return cls.from_list([None] * n)

    def proportional_sample(self, value):
        """Return index of given value. Probability of sampling index is proportional to its error value."""
        if value > self.value:
            raise ValueError(f"Value is to high. Sum of tree is {self.value}")
        return self.__proportional_sample(value)

    def __proportional_sample(self, value):
        if self.is_leaf:
            return self.index

        if self.right and value > self.left.value:
            # if we go right we need to subtract value of left node, so we can work in local subtree
            return self.right.__proportional_sample(value - self.left.value)
        else:
            return self.left.__proportional_sample(value)


class PrioritizedRingBuf(RingBuf):
    def __init__(self, size):
        self.data, self.root = PrioritizedExperienceReplayNode.init_tree_with_n_leafs(size + 1)
        self.start = 0
        self.end = 0

    @property
    def error_sum(self):
        return self.root._value

    @property
    def min_value(self):
        minimum = self.data[0].value
        for element in self.data[1:]:
            if 0 < element.value < minimum:
                minimum = element.value
        return minimum

    def append(self, element):
        self.data[self.end].update_value(element)
        self.end = (self.end + 1) % len(self.data)
        self.data[self.end].update_value(None)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def to_list(self):
        return self[:]

    def __getitem__(self, idx):
        nodes = self.get_leaf(idx)
        if type(nodes) == list:
            return [node.value for node in nodes]
        else:
            return nodes.value

    def get_leaf(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        else:
            if idx >= 0:
                return self.data[(self.start + idx) % len(self.data)]
            else:
                if idx < -len(self.data):
                    raise ValueError("Incorrect index- out of range")
                return self.data[(self.end - idx + 1) % len(self.data)]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample(self, value):
        """Return buffer index of passed value from that range."""
        # index of data list without taking into account buffer shift
        idx = self.root.proportional_sample(value)

        if idx >= self.start:
            return idx - self.start
        else:
            return len(self) - 1 - self.end + idx

    def get_sample_probability(self, index):
        leaf = self.get_leaf(index)
        return leaf.value / self.error_sum


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, size, n_state_frames):
        super().__init__(size, n_state_frames)

        self.errors = PrioritizedRingBuf(size)

    @property
    def get_min_probability(self):
        return self.errors.root.epsilon / self.errors.error_sum

    def add(self, state: List[np.array], action: np.array, new_frame: np.array, reward: int,
            terminate: bool, error: float):
        super().add(state, action, new_frame, reward, terminate)
        self.errors.append(error)

    def extend(self, states, actions, new_frames, rewards, terminates, errors):
        for s, a, nf, r, t, e in zip(states, actions, new_frames, rewards, terminates, errors):
            self.add(s, a, nf, r, t, e)

    def sample_batch(self, n):
        """Returns batch of samples."""
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        indexes = []
        probabilities = []
        for value in self.random(n):
            # sample with probability proportional to error value
            index = self.errors.sample(value)
            indexes.append(index)
            state, action, reward, next_state, terminate = self.get_sample(index)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminate_state.append(terminate)
            probabilities.append(self.errors.get_sample_probability(index))

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminate_state), \
               indexes, np.array(probabilities)

    def get_sample(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.new_states[idx]
        terminate = self.terminate_state[idx]

        return state, action, reward, next_state, terminate

    def random(self, n):
        """Returns n random numbers from range <0, self.error.error_sum>."""
        return np.random.rand(n) * (self.errors.error_sum)

    def update_errors(self, indexes, new_errors):
        """Updates errors of nodes"""
        for index, error in zip(indexes, new_errors):
            node = self.errors.get_leaf(index)
            node.update_value(error)