import time
import torch
import numpy as np
import random
import copy
import gc
import pickle
# base
from tqdm import tqdm
from codes.job import Job
from codes.machine import MachineConfig

# env
from independent_job.env_Q import Env
from independent_job.problem import get_jobs
# model
from independent_job.matrix.alg import MatrixAlgorithm_Q
from independent_job.matrix.agent import BGCQ
from independent_job.buffer import ReplayBuffer

# log
import wandb

class trainerQ():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.model_name == 'matrix':
            self.cfg.buffer = ReplayBuffer(100000, 32)
            self.agent = BGCQ(cfg)
            self.algorithm = lambda agent : MatrixAlgorithm_Q(agent)
            self.name = f"{self.cfg.model_name}-{cfg.model_params['TMHA']}-{cfg.model_params['MMHA']}"
            
        elif cfg.model_name == 'fit':
            # self.agent = Fit(cfg)
            # self.algorithm = lambda cfg : FitAlgorithm(cfg)
            self.name = f"{self.cfg.model_name}"

    def setup(self):
        cpus, mems, pfs, pss, pes, mips = self.cfg.m_resource_config
        self.cfg.machine_configs = [MachineConfig(cpu_capacity=cpu,
                                    memory_capacity=mem_disk,
                                    disk_capacity=mem_disk,
                                    pf=pf, ps=ps, pe=pe,
                                            mips = mips) for cpu, mem_disk, pf, ps, pe, mips in zip(cpus,\
                mems, pfs, pss, pes, mips)]
        self.cfg.max_energy = sum([m.ps + (m.pf - m.ps) * (1.0 ** m.pe) for m in self.cfg.machine_configs])
        self.valid_task = [get_jobs(self.cfg) for _ in range(5)]
        self.best_valid_energy = self.cfg.max_energy * self.cfg.terminal_time
        wandb.init(project='cloud')
        wandb.run.name = self.name
        wandb.run.save()

        wandb.config.update(self.cfg.to_dict())
        
        
    def fit(self):
        self.setup()

        # self.cfg.task_configs = get_jobs(self.cfg)
        with tqdm(range(self.cfg.epoch), unit="Run") as runing_bar:
            for i in runing_bar:
                self.agent.scheduler.step()
                loss, clock, energy = self.training(i)
                valid_clock, valid_energy = self.valiing()

                runing_bar.set_postfix(train_loss=loss,
                                   valid_clock=valid_clock,
                                   valid_energy=valid_energy,)

                wandb.log({"Training loss": loss,
                           "Training clock": clock,
                           "Training energy": energy,
                           "valid_clock": valid_clock,
                           "valid_energy": valid_energy})
                
            finish_save_model_name = 'final' + self.cfg.model_params['save_path']
            self.cfg.agent.model_save(finish_save_model_name)
                
            with open('my_dict.pkl', 'wb') as f:
                pickle.dump(self.agent.train_V, f)
                
            
    def roll_out(self):
        clock_list, energy_list = [], []

        algorithm = self.algorithm(self.agent)
        sim = Env(self.cfg)
        sim.setup()
        sim.episode(algorithm)
        eg = sim.total_energy_consumptipn
            
        clock_list.append(sim.time)
        energy_list.append(eg)

        if self.cfg.buffer.size > 32 * 10:
            for _ in range(10):
                loss = self.agent.optimize_model()

        return loss, np.mean(clock_list), np.mean(energy_list)
    
    def training(self, epoch):
        losses, clocks, energys = [], [], []
        torch.cuda.empty_cache()
        self.agent.model.train()

        self.cfg.task_configs = get_jobs(self.cfg)
        loss, clock, energy = self.roll_out()

        losses.append(loss)
        clocks.append(clock)
        energys.append(energy)
        return np.mean(losses), np.mean(clocks), np.mean(energys)

    def valiing(self):
        clocks, energys = [], []
        self.agent.model.eval()

        for job in self.valid_task:
            self.cfg.task_configs = copy.deepcopy(job)
            algorithm = self.algorithm(self.agent)
            sim = Env(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            clocks.append(sim.time)
            energys.append(eg)
            
        if np.mean(energys) <= self.best_valid_energy:
            self.best_valid_energy = np.mean(energys)
            self.agent.model_save()
            print("model save")

        return np.mean(clocks), np.mean(energys)
    

import torch
import numpy as np
import random
import os

from independent_job.config import matrix_config
from independent_job.config import fit_config

if __name__ == '__main__':
    # seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) 

    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    epoch = 500
    jobs_len = 5
    model_name = 'matrix'
    if model_name == 'matrix':
        cfg = matrix_config()
        cfg.model_name = model_name
        cfg.epoch = epoch
        cfg.jobs_len = jobs_len
        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device

        # encoder type
        cfg.model_params['TMHA'] = 'depth'
        cfg.model_params['MMHA'] = 'depth'
        cfg.model_params['policy_loss_weight'] = 1.0
        cfg.model_params['G_t_loss_weight'] = 0.06
        cfg.model_params['entropy_loss_weight'] = 0.000005
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_{}Q.pth'.format(
                                                                        cfg.model_name,
                                                                        cfg.epoch,
                                                                        cfg.jobs_len,
                                                                        cfg.model_params['TMHA'],
                                                                        cfg.model_params['MMHA'],
                                                                        SEED)
    # elif model_name == 'fit':
    #     cfg = fit_config()
    #     cfg.model_name = model_name
        
    #     # epoch
    #     cfg.epoch = epoch
        
    #     # job len
    #     cfg.jobs_len = jobs_len

    #     cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    #     cfg.model_params['device'] = cfg.device
    #     cfg.model_params['policy_loss_weight'] = 1.0
    #     # model_name/epoch/train_len/valid_len/job_len//seed
    #     cfg.model_params['save_path'] = '{}_{}_{}.pth'.format(
    #                                                             cfg.model_name,
    #                                                             cfg.epoch,
    #                                                             cfg.jobs_len,
    #                                                             SEED)    
    
    triner = trainerQ(cfg)
    triner.fit()