import numpy as np
import torch

class ReplayBuffer():
    """
    버퍼 기본
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.s_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.s_prime_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.pre_idx = 0
        self.size = 0
    
    def put(self, prime_state, state):
        self.s_mem[self._idx] = state
        self.s_prime_mem[self._idx] = prime_state
        self.pre_idx = self._idx
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        batch_s = self.s_mem[idxs]
        batch_s_prim = self.s_prime_mem[idxs]

        machine_feature, task_feature, D_TM, ninf_mask, available_task_idx, reward, action, done = \
        [],[],[],[],[],[],[],[]
        machine_feature_s_prime, task_feature_s_prime, D_TM_s_prime,\
        ninf_mask_s_prime, available_task_idx_s_prime, reward_s_prime, action_s_prime, done_s_prime = \
        [],[],[],[],[],[],[],[]
        
        for s, s_prime in zip(batch_s, batch_s_prim):
                machine_feature.append(s.machine_feature); machine_feature_s_prime.append(s_prime.machine_feature)
                task_feature.append(s.task_feature); task_feature_s_prime.append(s_prime.task_feature)
                D_TM.append(s.D_TM); D_TM_s_prime.append(s_prime.D_TM)
                
                
                ninf_mask.append(s.ninf_mask); ninf_mask_s_prime.append(s_prime.ninf_mask)
                #available_task_idx.append(s.available_task_idx); available_task_idx_s_prime.append(s_prime.available_task_idx)
                
                reward.append(s.reward); reward_s_prime.append(s_prime.reward)
                action.append(s.action); action_s_prime.append(s_prime.action)
                done.append(s.done); done_s_prime.append(s_prime.done)

        machine_feature = torch.stack(machine_feature).squeeze(dim=1)
        task_feature = torch.stack(task_feature).squeeze(dim=1)
        D_TM = torch.stack(D_TM).squeeze(dim=1)
        ninf_mask = torch.stack(ninf_mask).squeeze(dim=1)

        reward = torch.tensor(reward)
        action = torch.tensor(action)
        done = torch.tensor(done)

        machine_feature_s_prime = torch.stack(machine_feature_s_prime).squeeze(dim=1)
        task_feature_s_prime = torch.stack(task_feature_s_prime).squeeze(dim=1)
        D_TM_s_prime = torch.stack(D_TM_s_prime).squeeze(dim=1)
        ninf_mask_s_prime = torch.stack(ninf_mask_s_prime).squeeze(dim=1)

        reward_s_prime = torch.tensor(reward_s_prime)
        action_s_prime = torch.tensor(action_s_prime)
        done_s_prime = torch.tensor(done_s_prime)

        experiences = machine_feature, \
                        task_feature, \
                        D_TM, \
                        ninf_mask, \
                        reward, \
                        action, \
                        done, \
                        machine_feature_s_prime, \
                        task_feature_s_prime, \
                        D_TM_s_prime, \
                        ninf_mask_s_prime, \
                        reward_s_prime, \
                        action_s_prime, \
                        done_s_prime
        return experiences

    def __len__(self):
        return self.size