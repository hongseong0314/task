import torch

class FitAlgorithm:
    def __init__(self, agent):
        self.agent = agent

        self.action = None
        self.feature = None
        
    def __call__(self, state):
        machine_feature = state.machine_feature
        ninf_mask = state.ninf_mask
        available_task_idx = state.available_task_idx

        m_idx, t_idx = torch.nonzero((~torch.isinf(ninf_mask)).squeeze(0), as_tuple=True)

        pair_index = self.agent.decision(machine_feature)

        machine_pointer = m_idx[pair_index]
        task_pointer = available_task_idx[int(t_idx[pair_index])].item()
        self.feature = machine_feature.clone(); self.action = pair_index;
        # if pair_index == 9:
        #     print(machine_feature.shape, pair_index, self.feature.shape, self.action)
        return machine_pointer, task_pointer