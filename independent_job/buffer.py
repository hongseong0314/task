import numpy as np
import torch
import gc
from dataclasses import dataclass

@dataclass
class BufferState:
    machine_feature: torch.Tensor = None
    task_feature: torch.Tensor = None
    D_TM: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    logpa : torch.Tensor = None

class EpisodeBuffer():
    """
    버퍼 기본
    """
    def __init__(self,
                 buffer_limit=10000,
                 batch_size=12):
        self.s_mem = np.empty(shape=(batch_size, buffer_limit), dtype=np.ndarray)
        self.max_size = buffer_limit
        self.batch_size = batch_size
        self.trajectory_pointer = 0
        self._idx = 0
        self.size = 0
        self.trajectory_max_T, self.G_ts = [], []

    def trajectory_up(self, G_t):
        self.trajectory_pointer += 1
        self.trajectory_max_T.append(self._idx)
        self.G_ts.append(G_t)
        self._idx = 0
        self.size = 0

    def put(self, state):
        self.s_mem[self.trajectory_pointer, self._idx] = state
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def get_trajectory(self, batch_size=None):
        return np.array([self.s_mem[i, :self.trajectory_max_T[i]] \
                         for i in range(self.trajectory_pointer)]), self.G_ts

    def __len__(self):
        return self.size

    def reset(self):
        self.s_mem = np.empty(shape=(self.batch_size, self.max_size), dtype=np.ndarray)
        self.trajectory_pointer = 0
        self._idx = 0
        self.size = 0

        self.trajectory_max_T, self.G_ts = [], []
        gc.collect()
        pass