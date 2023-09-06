import torch

class MatrixAlgorithm:
    def __init__(self, agent):
        self.agent = agent
        
    def __call__(self, state):
        machine_feature = state.machine_feature
        task_feature = state.task_feature
        D_TM = state.D_TM
        ninf_mask = state.ninf_mask
        available_task_idx = state.available_task_idx
        task_num = task_feature.size(1)

        if self.agent.skip:
            task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
            machine_pointer = task_selected // (task_num+1)
            task_pointer = task_selected % (task_num+1)
            if task_pointer == 0:
                return None, None
            else:
                return machine_pointer, available_task_idx[int(task_pointer)-1].item()

        else: 
            task_selected = self.agent.decision(machine_feature, task_feature, D_TM, ninf_mask)
            machine_pointer = task_selected // task_num
            task_pointer = task_selected % task_num

            return machine_pointer, available_task_idx[int(task_pointer)].item()

class MatrixAlgorithm_Q:
    def __init__(self, agent):
        self.agent = agent
        
    def __call__(self, state):
        machine_feature = state.machine_feature
        task_feature = state.task_feature
        D_TM = state.D_TM
        ninf_mask = state.ninf_mask
        available_task_idx = state.available_task_idx
        task_num = task_feature.size(1)
        batch_size = task_feature.size(0)

        padding_task_feature = torch.zeros((batch_size, 50, 4))
        padding_task_feature[:, :task_num, :] = task_feature
        padding_task_D_TM = torch.zeros((batch_size, 50, 5, 2))
        padding_task_D_TM[:, :task_num, :, :] = D_TM
        padding_task_ninf_mask = torch.full(size=(batch_size, 5, 50),fill_value=float('-inf'))
        padding_task_ninf_mask[:, :, :task_num] = ninf_mask

        state.task_feature = padding_task_feature
        state.D_TM = padding_task_D_TM
        state.ninf_mask = padding_task_ninf_mask
        if self.agent.skip:
            task_selected = self.agent.decision(machine_feature, padding_task_feature, \
                                                padding_task_D_TM, padding_task_ninf_mask)
            machine_pointer = task_selected // (50+1)
            task_pointer = task_selected % (50+1)

            state.action = task_selected
            if task_pointer == 0:
                return None, None
            else:
                return machine_pointer, available_task_idx[int(task_pointer)-1].item()

        else: 
            task_selected = self.agent.decision(machine_feature, padding_task_feature, \
                                                padding_task_D_TM, padding_task_ninf_mask)
            machine_pointer = task_selected // 50
            task_pointer = task_selected % 50
            state.action = task_selected
            state.machine_select = machine_pointer
            state.task_select = task_pointer
            return machine_pointer, available_task_idx[int(task_pointer)].item()