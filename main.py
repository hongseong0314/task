import torch
import numpy as np
import random
import os
from independent_job.trainer import trainer, trainerTest

from independent_job.config import depth_config, mix_config
from independent_job.config import fit_config

if __name__ == '__main__':
    # seed
    SEED = 33
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED) 

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    epoch = 500
    jobs_len = 5
    model_name = 'matrix'

    if model_name == 'matrix':
        # encoder type
        encoder = 'depth'
        if encoder == 'depth':
            cfg = depth_config()
            cfg.model_params['TMHA'] = 'depth'
            cfg.model_params['MMHA'] = 'depth'
        else:
            cfg = mix_config() 
            cfg.model_params['TMHA'] = 'mix'
            cfg.model_params['MMHA'] = 'mix'

        cfg.encoder = encoder
        cfg.model_name = model_name
        cfg.epoch = epoch
        cfg.jobs_len = jobs_len
        cfg.seed = SEED
        cfg.object = 'eng'
        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_{}_{}step3e3.pth'.format(
                                                                        cfg.object,
                                                                        cfg.model_name,
                                                                        cfg.epoch,
                                                                        cfg.jobs_len,
                                                                        cfg.model_params['TMHA'],
                                                                        cfg.model_params['MMHA'],
                                                                        SEED)
    elif model_name == 'fit':
        cfg = fit_config()
        cfg.model_name = model_name
        cfg.seed = SEED
        cfg.object = 'eng'
        # epoch
        cfg.epoch = epoch
        
        # job len
        cfg.jobs_len = jobs_len

        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device

        # model_name/epoch/train_len/valid_len/job_len//seed
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_TEST.pth'.format(
                                                                cfg.object,
                                                                cfg.model_name,
                                                                cfg.epoch,
                                                                cfg.jobs_len,
                                                                SEED)    
    
    triner = trainerTest(cfg)
    triner.fit()