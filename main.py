import torch
import numpy as np
import random
import os
from independent_job.trainer import trainer, trainerC

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
        cfg.model_params['save_path'] = '{}_{}_{}_{}_{}_{}critic.pth'.format(
                                                                        cfg.model_name,
                                                                        cfg.epoch,
                                                                        cfg.jobs_len,
                                                                        cfg.model_params['TMHA'],
                                                                        cfg.model_params['MMHA'],
                                                                        SEED)
    elif model_name == 'fit':
        cfg = fit_config()
        cfg.model_name = model_name
        
        # epoch
        cfg.epoch = epoch
        
        # job len
        cfg.jobs_len = jobs_len

        cfg.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        cfg.model_params['device'] = cfg.device
        cfg.model_params['policy_loss_weight'] = 1.0
        # model_name/epoch/train_len/valid_len/job_len//seed
        cfg.model_params['save_path'] = '{}_{}_{}.pth'.format(
                                                                cfg.model_name,
                                                                cfg.epoch,
                                                                cfg.jobs_len,
                                                                SEED)    
    
    triner = trainerC(cfg)
    triner.fit()