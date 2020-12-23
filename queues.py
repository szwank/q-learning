from typing import List

import numpy as np


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
        self.teminate_state = RingBuf(size)

    def __len__(self):
        return len(self.actions)

    def add(self, state: List[np.array], action: np.array, new_frame: np.array, reward: int,
            terminate: bool):
        # add only new_frame, the rest of them are all alreaty in buffer
        if len(self.states) == 0:
            self.states.extend(state)
        self.states.append(new_frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.teminate_state.append(terminate)

    def sample_batch(self, n):
        """Returns batch of samples."""
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        # remove -2 because -1 is from counting from 0 and -1 is from counting most
        # current frame as first one and foutrue frame as +1 frame
        for i in np.random.randint(0, len(self) - 1, n):
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
        terminate = self.teminate_state[idx]

        return state, action, reward, next_state, terminate