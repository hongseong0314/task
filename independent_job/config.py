from ml_collections import config_dict

def base_config():
    cfg = config_dict.ConfigDict()
    
    # machine
    cfg.machines_number = 5
    cfg.nM = 2
    cfg.m_resource_config = [[16, 32, 32, 16, 32],
                            [0.5, 1.5, 2, 1.0, 0.5],
                            [675.7838, 643.8629,258.2628,332.1814,119.0417],
                            [193.4651, 193.8555,66.8607,101.3687,45.3834],
                            [0.9569,0.7257,1.5767,0.7119,1.5324],
                            [1.5,1.3,0.5,1.0,0.8]]
    # task
    cfg.jobs_len = 10
    cfg.nT = 4
    cfg.min_task, cfg.max_task = 1, 10
    cfg.mim_cpu, cfg.max_cpu = 0.4, 2.0
    cfg.min_memory, cfg.max_memory = 0.002, 0.008
    cfg.min_duration, cfg.max_duration = 1.0, 48.0
    cfg.min_instance_num, cfg.max_instance_num = 1, 29
    cfg.task_num = []

    # train
    cfg.epoch = 100
    cfg.terminal_time = 350
    return cfg

def depth_config():
    cfg = base_config()
    cfg.model_params = {
                        'embedding_dim': 32,
                        'sqrt_embedding_dim': 32**(1/2),
                        'encoder_layer_num': 3,
                        'qkv_dim': 8,
                        'sqrt_qkv_dim': 8**(1/2),
                        'head_num': 4,
                        'logit_clipping': 10,
                        'ff_hidden_dim': 64,

                        'nT':cfg.nT,
                        'nM':cfg.nM,
                        'depth_hidden_dim':6,
                        'depth__init':(1/3)**(1/2),
                        'FC_init':(1/6)**(1/2),

                        'policy_loss_weight':1.0,
                        'entropy_loss_weight':0.01,

                        'save_path' : None,
                        'load_path' : None,
                        'skip':False,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-3,
                            # 'weight_decay': 1e-5
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }
    return cfg

def mix_config():
    cfg = base_config()
    cfg.model_params = {
                        'embedding_dim': 32,
                        'sqrt_embedding_dim': 32**(1/2),
                        'encoder_layer_num': 2,
                        'qkv_dim': 8,
                        'sqrt_qkv_dim': 8**(1/2),
                        'head_num': 4,
                        'logit_clipping': 10,
                        'ff_hidden_dim': 32,    

                        'nT':cfg.nT,
                        'nM':cfg.nM,


                        'policy_loss_weight':1.0,
                        'entropy_loss_weight':0.00,

                        'ms_hidden_dim': 6,
                        'ms_layer1_init': (1/3)**(1/2),
                        'ms_layer2_init': (1/6)**(1/2),

                        'save_path' : None,
                        'load_path' : None,
                        'skip':False,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-3,
                            # 'weight_decay': 1e-5
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }
    return cfg

def fit_config():
    cfg = base_config()
    cfg.model_params = {
                        'nT':cfg.nT,
                        'nM':cfg.nM,
                        'policy_loss_weight':1.0,
                        'entropy_loss_weight':0,
                        'save_path' : None,
                        'load_path' : None,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-3,
                            # 'weight_decay': 1e-5
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }

    return cfg
