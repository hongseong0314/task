import numpy as np
from codes.job import Job

def get_jobs(cfg):
    submit_time = np.random.randint(low=0, high=50, size=cfg.jobs_len)
    submit_time.sort()
    # cfg.submit_time = submit_time
    return [Job(submit_time=t, tasks=get_tasks(cfg)) for t in submit_time]

def get_tasks(cfg):
    task_num = np.random.randint(low=cfg.min_task, high=cfg.max_task)
    cpu = np.random.uniform(low=cfg.mim_cpu,high=cfg.max_cpu,size=task_num)
    mem = np.random.uniform(low=cfg.min_memory,high=cfg.max_memory,size=task_num)
    duration = np.random.uniform(low=cfg.min_duration,high=cfg.max_duration,size=task_num)
    instance_num = np.random.randint(low=cfg.min_instance_num,high=cfg.max_instance_num,size=task_num)
    # cfg.task_num.append(task_num)
    return np.c_[cpu, mem, duration, instance_num].tolist()