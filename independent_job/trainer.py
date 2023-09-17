# get_jobs
import time
import torch
import numpy as np
import random
import copy
import pickle
import gc
# base
from tqdm import tqdm
from codes.job import Job
from codes.machine import MachineConfig

# env
from independent_job.env import Env, EnvTest
from independent_job.problem import get_jobs
# model
from independent_job.matrix.alg import MatrixAlgorithm
from independent_job.matrix.agent import BGC, BGCC, BGCmean, BGCN

from independent_job.fit.alg import FitAlgorithm
from independent_job.fit.model import Fit

# log
import wandb

class trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.model_name == 'matrix':
            self.agent = BGC(cfg)
            self.algorithm = lambda agent : MatrixAlgorithm(agent)
            self.name = f"{self.cfg.model_name}-{cfg.model_params['TMHA']}-{cfg.model_params['MMHA']}"
            
        elif cfg.model_name == 'fit':
            self.agent = Fit(cfg)
            self.algorithm = lambda cfg : FitAlgorithm(cfg)
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
        self.best_valid_make_span = self.cfg.terminal_time
        wandb.init(project='cloud')
        wandb.run.name = self.name 
        wandb.run.save()

        wandb.config.update(self.cfg.to_dict())
        
        
    def fit(self):
        self.setup()

        with tqdm(range(self.cfg.epoch), unit="Run") as runing_bar:
            for i in runing_bar:
                self.agent.scheduler.step()
                loss, clock, energy, make_span = self.training(i)
                
                if i % 1 == 0:
                    valid_clock, valid_energy, valid_make_span = self.valiing()

                    runing_bar.set_postfix(train_loss=loss,
                                    valid_clock=valid_clock,
                                    valid_energy=valid_energy,
                                    valid_make_span=valid_make_span,)

                    wandb.log({"Training loss": loss,
                            "Training clock": clock,
                            "Training energy": energy,
                            "valid_clock": valid_clock,
                            "valid_energy": valid_energy,
                            'valid_make_span':valid_make_span,})
            finish_save_model_name = 'final' + self.cfg.model_params['save_path']
            self.agent.model_save(finish_save_model_name)
                
            
    def roll_out(self, epoch):
        clock_list, energy_list, make_span = [], [], []
        for _ in range(12):
            random.shuffle(self.cfg.machine_configs)
            algorithm = self.algorithm(self.agent)
            sim = Env(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            # self.agent.trajectory(-eg / (sim.max_energy * sim.terminal_time)) 
            self.agent.trajectory(-sim.total_make_span) 
            
            clock_list.append(sim.time)
            make_span.append(sim.total_make_span)
            energy_list.append(eg)
            
        entropy_weight = 0#np.clip(1 - (epoch / (self.cfg.epoch / 2)), a_min=0, a_max=None)
        print(f"entropy_weight : {entropy_weight}")
        loss = self.agent.optimize_model(entropy_weight)

        return loss, np.mean(clock_list), np.mean(energy_list), np.mean(make_span)
    
    def training(self, epoch):
        losses, clocks, energys, make_spans = [], [], [], []
        torch.cuda.empty_cache()
        self.agent.model.train()

        self.cfg.task_configs = get_jobs(self.cfg)
        loss, clock, energy, make_span = self.roll_out(epoch)

        losses.append(loss)
        clocks.append(clock)
        energys.append(energy)
        make_spans.append(make_span)
        
        return np.mean(losses), np.mean(clocks), np.mean(energys), np.mean(make_spans)

    def valiing(self):
        clocks, energys, make_spans = [], [], []
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
            make_spans.append(sim.total_make_span)

        # if np.mean(energys) <= self.best_valid_energy:
        #     self.best_valid_energy = np.mean(energys)
        #     self.agent.model_save()
        #     print("model save")
        if np.mean(make_spans) <= self.best_valid_make_span:
            self.best_valid_make_span = np.mean(make_spans)
            self.agent.model_save()
            print("model save")
        return np.mean(clocks), np.mean(energys), np.mean(make_spans)
    
class trainerTest():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.model_name == 'matrix':
            self.agent = BGC(cfg)
            self.algorithm = lambda agent : MatrixAlgorithm(agent)
            self.name = f"{self.cfg.model_name}-{cfg.model_params['TMHA']}-{cfg.model_params['MMHA']}"
            
        elif cfg.model_name == 'fit':
            self.agent = Fit(cfg)
            self.algorithm = lambda cfg : FitAlgorithm(cfg)
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
        
        with open('train_scale.pickle', 'rb') as file:
            self.train_jobs = pickle.load(file)
        
        self.valid_task = [np.random.choice(self.train_jobs, size=self.cfg.jobs_len, \
                                            replace=False) for _ in range(5)]

        self.best_valid_energy = self.cfg.max_energy * self.cfg.terminal_time
        self.best_valid_make_span = self.cfg.terminal_time
        wandb.init(project='cloud')
        wandb.run.name = self.name 
        wandb.run.save()

        wandb.config.update(self.cfg.to_dict())
        
        
    def fit(self):
        self.setup()

        with tqdm(range(self.cfg.epoch), unit="Run") as runing_bar:
            for i in runing_bar:
                if self.agent.scheduler:
                    self.agent.scheduler.step()
                loss, clock, energy, make_span = self.training(i)
                
                if i % 1 == 0:
                    valid_clock, valid_energy, valid_make_span = self.valiing()

                    runing_bar.set_postfix(train_loss=loss,
                                    valid_clock=valid_clock,
                                    valid_energy=valid_energy,
                                    valid_make_span=valid_make_span,)

                    wandb.log({"Training loss": loss,
                            "Training clock": clock,
                            "Training energy": energy,
                            "valid_clock": valid_clock,
                            "valid_energy": valid_energy,
                            'valid_make_span':valid_make_span,})
            finish_save_model_name = 'final' + self.cfg.model_params['save_path']
            self.agent.model_save(finish_save_model_name)
                
            
    def roll_out(self, epoch):
        clock_list, energy_list, make_span = [], [], []
        for _ in range(12):
            # random.shuffle(self.cfg.machine_configs)
            self.cfg.task_configs = copy.deepcopy(self.train_job)
            algorithm = self.algorithm(self.agent)
            sim = EnvTest(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            self.agent.trajectory(-eg / (sim.max_energy * sim.terminal_time)) 
            # self.agent.trajectory(-sim.total_make_span) 
            
            clock_list.append(sim.time)
            make_span.append(sim.total_make_span)
            energy_list.append(eg)
            
        entropy_weight = self.cfg.model_params['entropy_loss_weight'] * \
            np.clip(1 - (epoch / (100)), a_min=0, a_max=None)
        print(f"entropy_weight : {entropy_weight}")
        loss = self.agent.optimize_model(entropy_weight)

        return loss, np.mean(clock_list), np.mean(energy_list), np.mean(make_span)
    
    def training(self, epoch):
        losses, clocks, energys, make_spans = [], [], [], []
        torch.cuda.empty_cache()
        self.agent.model.train()

        self.train_job = np.random.choice(self.train_jobs, size=self.cfg.jobs_len, \
                                            replace=False)
        loss, clock, energy, make_span = self.roll_out(epoch)

        losses.append(loss)
        clocks.append(clock)
        energys.append(energy)
        make_spans.append(make_span)
        
        return np.mean(losses), np.mean(clocks), np.mean(energys), np.mean(make_spans)

    def valiing(self):
        clocks, energys, make_spans = [], [], []
        self.agent.model.eval()
        for job in self.valid_task:
            self.cfg.task_configs = copy.deepcopy(job)
            algorithm = self.algorithm(self.agent)
            sim = EnvTest(self.cfg)
            sim.setup()
            sim.episode(algorithm)
            eg = sim.total_energy_consumptipn
            clocks.append(sim.time)
            energys.append(eg)
            make_spans.append(sim.total_make_span)

        # if np.mean(energys) <= self.best_valid_energy:
        #     self.best_valid_energy = np.mean(energys)
        #     self.agent.model_save()
        #     print("model save")
        if np.mean(make_spans) <= self.best_valid_make_span:
            self.best_valid_make_span = np.mean(make_spans)
            self.agent.model_save()
            print("model save")
        return np.mean(clocks), np.mean(energys), np.mean(make_spans)
    
class trainerC():
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.model_name == 'matrix':
            self.agent = BGCC(cfg)
            self.algorithm = lambda agent : MatrixAlgorithm(agent)
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
            self.agent.model_save(finish_save_model_name)
                
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
        loss = self.agent.optimize_model(-eg / (sim.max_energy * sim.terminal_time))

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