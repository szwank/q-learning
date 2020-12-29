from typing import List

import numpy as np
from collections import deque


class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
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
        # States are saved as separete frames. Each state consists of n_state_frames.
        self.n_state_frames = n_state_frames

        # Buffer for states needs to be bigger. We store there n_state_frames
        # per state and extra frame as future state
        self.states = RingBuf(size + n_state_frames)
        self.actions = RingBuf(size)
        self.rewards = RingBuf(size)
        self.terminate_state = RingBuf(size)

    def __len__(self):
        return len(self.actions)

    def add(self, state: List[np.array], action: np.array, new_frame: np.array, reward: int,
            terminate: bool):
        # add only new_frame, the rest of them are all already in buffer
        if len(self.states) == 0:
            self.states.extend(state)
        self.states.append(new_frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminate_state.append(terminate)

    def sample_batch(self, n):
        """Returns batch of samples."""
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        # remove -2 because -1 is from counting from 0 and -1 is from counting most
        # current frame as first one and future frame as +1 frame
        for i in np.random.randint(0, len(self)-1, n):
            state, action, reward, next_state, terminate = self.get_sample(i)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminate_state.append(terminate)

        return np.swapaxes(states, 1, 3), np.array(actions), np.array(rewards), \
               np.swapaxes(next_states, 1, 3), np.array(terminate_state)

    def get_sample(self, idx):
        state = self.states[idx:idx + self.n_state_frames]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.states[idx + 1:idx + self.n_state_frames + 1]
        terminate = self.terminate_state[idx]

        return state, action, reward, next_state, terminate


class TreeNode:
    """Node of unsorted binary tree. """
    def __init__(self, value, parent=None, left=None, right=None):
        self.parent = parent
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def parent_exists(self):
        return self.parent is not None

    def update_value(self, error):
        """Updates error value of node and its parents."""
        self._error = error
        if self.parent_exists:
            self.parent.__update_value()

    def __update_value(self):
        """Update error value base on child nodes."""
        self._error = self.left.error + self.right.error
        if self.parent_exists:
            self.parent.__update_value()

    def proportional_sample(self, value):
        """Return index of given value. Probability of sampling index is proportional to its error value."""
        if value > self.error:
            raise ValueError(f"Value is to high. Sum of tree is {self.error}")
        return self.__proportional_sample

    def __proportional_sample(self, value):
        if self.is_leaf:
            return self.index

        if self.right and value > self.right.error:
            return self.right.__proportional_sample(value)
        else:
            return self.left.__proportional_sample(value-self.left.error)


class PrioritizedExperienceReplayNode(TreeNode):
    """Node of unsorted binary tree. value of _error field equal to None indices unused node in tree.
    Only leaf nodes can be unused. """
    def __init__(self, error, index=None, parent=None, left=None, right=None):
        super().__init__(value=error, parent=parent, left=left, right=right)
        self.epsilon = 0.0001
        self.index = index

    @property
    def error(self):
        """Returns modified _error value. Returns _error value modified by epsilon value if its leaf node.
        If _error value is None returns 0. In other cases returns 0"""
        if self.value is None:
            return 0
        elif self.is_leaf:
            return self.value + self.epsilon
        else:
            return self.value

    @classmethod
    def from_list(cls, data):
        """Initialize binary tree for given list. Empty nodes should have value of None."""
        lower_nodes = deque()
        upper_nodes = deque()

        for i, error in enumerate(data):
            node = cls(error=error, index=i)
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
                    r_node = cls(error=None)

                node = cls(error=l_node.error + r_node.error, left=l_node, right=r_node)

                l_node.parent = node
                r_node.parent = node

                upper_nodes.append(node)

            lower_nodes, upper_nodes = upper_nodes, lower_nodes
        return nodes, lower_nodes.pop()

    @classmethod
    def init_tree_with_n_leafs(cls, n):
        return cls.from_list([None]*n)


class PrioritizedRingBuf(RingBuf):
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data, self.root = PrioritizedExperienceReplayNode.init_tree_with_n_leafs(size+1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end].value = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def to_list(self):
        return self[:]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii].value for ii in range(*idx.indices(len(self)))]
        else:
            if idx >= 0:
                return self.data[(self.start + idx) % len(self.data)].value
            else:
                if idx < -len(self.data):
                    raise ValueError("Incorrect index- out of range")
                return self.data[(self.end - idx + 1) % len(self.data)].value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i].value


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, size, n_state_frames):
        super().__init__(size, n_state_frames)

        self.errors = TreeNode.init_n_leafs(size)

    def add(self, state: List[np.array], action: np.array, new_frame: np.array, reward: int,
            terminate: bool):
        # add only new_frame, the rest of them are all already in buffer
        if len(self.states) == 0:
            self.states.extend(state)
        self.states.append(new_frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminate_state.append(terminate)

        self.errors








