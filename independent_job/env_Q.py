import torch
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
    
    reward : int = 0
    action : int = None
    done : bool = False
    machine_select : int = None
    task_select : int = None

class Env(object):
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

        # energy
        self.total_energy_consumptipn = 0
        self.max_energy = cfg.max_energy

        # agent name        
        self.model_name = cfg.model_name
        self.buffer = cfg.buffer

        # scheduling
        self.scheduling = {}

    def setup(self):
        self.machines = [Machine(mc) for mc in self.machine_configs]
        self.job_done = False

        # state
        self.step_state =  State()
        self.pre_state = None
        self.machine_feature = torch.stack([m.state for m \
                                            in self.machines]).float()[None, ...].expand(1, self.machine_num, self.nM)
        # task.cpu, task.memory, task.duration, task.instances_number, idx
        self.task_full_feature = torch.zeros(size=(1, 0, self.nT + 1), dtype=torch.float32)
        self.D_TM = torch.zeros(size=(1, self.total_task_num, self.machine_num))
        self.ninf_mask = torch.full(size=(1, self.machine_num, self.total_task_num),fill_value=float('-inf'))

        # job time reset
        if self.task_configs[0].submit_time != 0:
            self.task_configs[0].submit_time = 0

    def arrived_job_check(self):
        while self.task_configs[self.job_pointer].submit_time <= self.time:
            task_config = self.task_configs[self.job_pointer]
            task_num = len(task_config.tasks)
            
            arrived_task = torch.cat((torch.tensor(task_config.tasks, \
                                                   dtype=torch.float32), \
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

            state = copy.deepcopy(self.step_state)
            if self.pre_state:
                self.buffer.put(state, self.pre_state)
            self.pre_state = state
            
            self.step_count += 1 
            if machine_num == None and task_num == None:
                break
            else:
                cpu, mem, duration = self.task_full_feature[0, task_num, [0, 1, 2]]
                task = torch.tensor([self.time, self.time + duration, cpu, mem], \
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
            self.total_energy_consumptipn += sum([m.energy_consumption() for m in self.machines])
        
            self.time += 1

            if self.time >= self.terminal_time:
                self.total_energy_consumptipn = self.terminal_time * self.max_energy
                break

        # final state
        pre_state = copy.deepcopy(self.buffer.s_prime_mem[self.buffer.pre_idx])
        pre_state.reward = -self.total_energy_consumptipn / (self.max_energy * self.terminal_time)
        pre_state.done = True
        self.buffer.put(pre_state,pre_state)

    def state_update(self):
        ## task feature update [B, T, F]
        available_task = ~(self.task_full_feature[..., -2] == 0)
        available_task = available_task.squeeze(0)
        # cpu, mem, duration, instance
        self.task_feature = self.task_full_feature[:,available_task, :-1]
        task_size = self.task_feature.size(1)

        ## machine update
        [m.finished_task_move(self.time) for m in self.machines]
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

        ## reward
        self.step_state.reward = -1#self.total_energy_consumptipn
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
            # cpu, mem, energy, cpu, mem, disk, duration, inster
            self.step_state.machine_feature = features[:, [0,1,5,7,8,9,10]].clone()

        elif self.model_name == 'random':
            pass