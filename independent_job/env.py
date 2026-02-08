import torch
import time
from dataclasses import dataclass
from codes.machine import Machine
import copy

@dataclass
class State:
    machine_feature: torch.Tensor = None
    task_feature: torch.Tensor = None
    D_TM: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    available_task_idx : torch.Tensor = None


class EnvTest(object):
    def __init__(self, cfg):
        # machine
        self.machine_configs = cfg.machine_configs
        self.machine_num = cfg.machines_number
        self.nM = cfg.nM

        # task
        self.task_configs = cfg.task_configs
        self.nT = cfg.nT
        self.total_task_num = 0
        self.job_pointer = 0

        # time, total step
        self.terminal_time = cfg.terminal_time
        self.time = 0 
        self.step_count = 0
        self.total_make_span = 0
        # energy
        self.total_energy_consumptipn = 0
        self.max_energy = cfg.max_energy

        # agent name        
        self.model_name = cfg.model_name

        # scheduling
        self.scheduling = {}

    def setup(self):
        self.machines = [Machine(mc) for mc in self.machine_configs]
        self.job_done = False

        # state
        self.step_state =  State()
        self.machine_feature = torch.stack([m.state for m \
                                            in self.machines]).float()[None, ...].expand(1, self.machine_num, self.nM)
        # task.cpu, task.memory, task.duration, task.instances_number, idx
        self.task_full_feature = torch.zeros(size=(1, 0, self.nT + 1), dtype=torch.float32)
        self.D_TM = torch.zeros(size=(1, self.total_task_num, self.machine_num))
        self.ninf_mask = torch.full(size=(1, self.machine_num, self.total_task_num),fill_value=float('-inf'))

        # job time reset
        for st in range(1, len(self.task_configs)):
            pre_st = self.task_configs[st-1].submit_time
            self.task_configs[st].submit_time = self.task_configs[st].submit_time + pre_st
        if self.task_configs[0].submit_time != 0:
            self.task_configs[0].submit_time = 0
            

    def arrived_job_check(self):
        while self.task_configs[self.job_pointer].submit_time <= self.time:
            task_config = self.task_configs[self.job_pointer]
            task_num = len(task_config.tasks)
            
            arrived_task = torch.cat((torch.tensor(task_config.tasks, \
                                                   dtype=torch.float32)[:, [0,1,3,4]], \
                                                    torch.arange(self.total_task_num, \
                                                                 self.total_task_num+task_num)[..., None]), dim=-1)
            self.task_full_feature = torch.cat((self.task_full_feature, \
                                           arrived_task[None, ...]), dim=1)
            self.total_task_num += task_num
            self.job_pointer += 1

            if self.job_pointer == len(self.task_configs):
                self.job_done = True
                break


    def parallel_rollout(self, decision_maker):
        while True:
            self.state_update()

            if len(self.step_state.available_task_idx) == 0 or (~torch.isinf(self.step_state.ninf_mask)).sum() == 0:
                break

            machine_num, task_num = decision_maker(self.step_state)
            self.step_count += 1 

            if machine_num == None and task_num == None:
                break
            else:
                cpu, mem, duration = self.task_full_feature[0, task_num, [0, 1, 2]]
                duration_s = self.machines[machine_num].mips * duration
                task = torch.tensor([self.time, self.time + duration_s, cpu, mem], \
                                    dtype=torch.float32)[None, ...]
                self.machines[machine_num].allocation(task)
                self.scheduling[f'{self.time}'].append(f"{task_num}-{self.task_full_feature[0, task_num, -2]}")
                self.task_full_feature[0, task_num, -2] -= 1


    def episode(self, decision_maker):
        while not (self.job_done and \
            not len(self.step_state.available_task_idx) and \
            not sum([m.running() for m in self.machines])):
            self.scheduling[f'{self.time}'] = []
            if not self.job_done: 
                self.arrived_job_check()
                
            self.parallel_rollout(decision_maker)
        
            if sum([m.running() for m in self.machines]):
                self.total_energy_consumptipn += sum([m.energy_consumption() for m in self.machines])
                self.total_make_span += 1
            self.time += 1

            if (self.time >= self.terminal_time) and (decision_maker.agent.model.training):
                self.total_energy_consumptipn = self.terminal_time * self.max_energy
                break

    def state_update(self):
        ## machone update
        [m.finished_task_move(self.time) for m in self.machines]

        # task feature update [B, T, F]
        available_task = ~(self.task_full_feature[..., -2] == 0)
        available_task = available_task.squeeze(0)
        self.task_feature = self.task_full_feature[:,available_task, :-1]
        task_size = self.task_feature.size(1)

        # machine feature update [B, M, F]
        self.machine_feature = torch.stack([m.state for m \
                                            in self.machines]).float()[None, ...].expand(1, self.machine_num, self.nM)
        pfse = torch.tensor([[m.cpu_capacity, m.pf, m.ps, m.pe, m.mips] \
                                for m in self.machine_configs], \
                                    dtype=torch.float32)
        

        TASK_IDX = torch.arange(self.machine_num)[:, None].expand(self.machine_num, task_size)
        MACHINE_IDX = torch.arange(task_size)[None, :].expand(self.machine_num, task_size)

        ## mask_update
        # available_machine [B, M, T]
        available_machine = (self.machine_feature[:, TASK_IDX, :2] >= \
                             self.task_feature[:, MACHINE_IDX, :2]).all(dim=3)

        self.ninf_mask = torch.full(size=(1, self.machine_num, task_size),fill_value=float('-inf'))
        self.ninf_mask[available_machine] = 0

        self.step_state.task_feature = self.task_feature.clone()
        self.step_state.ninf_mask = self.ninf_mask.clone()
        self.step_state.available_task_idx = self.task_full_feature[:, available_task, -1].squeeze(0).int()

        if task_size == 0:
            return None

        if self.model_name == 'matrix':
            ## D_MT update [B, M, T]
            # make span
            make_span = self.task_feature[None, ..., 2].expand(1, self.machine_num, task_size)
            mips = pfse[:, 4][None, :, None]
            make_span = make_span * mips
            # energy
            cpu_capacity = pfse[:, 0][None, :, None]
            pfs = pfse[:, 1][None, :, None]
            pss = pfse[:, 2][None, :, None]
            pes = pfse[:, 3][None, :, None]
            use_cpu = (cpu_capacity - (self.machine_feature[:, TASK_IDX, 0] -\
                                    self.task_feature[:,MACHINE_IDX, 0])) / cpu_capacity
            energy = (pss + (pfs - pss) * (use_cpu ** pes))
            self.D_TM = torch.stack((make_span.transpose(2,1), energy.transpose(2,1)), dim=3)

            self.step_state.machine_feature = self.machine_feature.clone()
            self.step_state.D_TM = self.D_TM.clone()

        elif self.model_name == 'fit':
            # cpu, mem, cpu_capacity, pf, ps, pe, mips
            self.machine_feature = torch.cat([self.machine_feature, pfse[None, ...].expand(1, self.machine_num, 5)], dim=-1)

            # cpu, mem, cpu_capacity, pf, ps, pe, mips + cpu, mem, duration, instance
            features = torch.cat([self.machine_feature[:, TASK_IDX, :], self.task_feature[:, MACHINE_IDX, :]],
                        dim=-1).reshape(task_size*self.machine_num, -1)[(~torch.isinf(self.ninf_mask)).reshape(-1,)]
        
            cpu_capacity = features[:, 2]
            pfs = features[:, 3]
            pss = features[:, 4]

            use_cpu = (cpu_capacity - (features[:, 0] - features[:, 7])) / cpu_capacity
            features[:, 5] = pss + (pfs - pss) * (use_cpu ** features[:, 5])
            features[:, 9] = features[:, 9] * features[:, 6]
            # cpu, mem, energy, cpu, mem, duration, instance
            self.step_state.machine_feature = features[:, [0,1,5,7,8,9,10]].clone()

        elif self.model_name == 'random':
            pass

        elif self.model_name == 'first':
            pass