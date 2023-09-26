import torch

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR
from independent_job.matrix.model import CloudMatrixModel, CloudMatrixModel_one_pose
from independent_job.matrix.utill import CosineAnnealingWarmUpRestarts

class BGCD():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel_one_pose(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)
        # self.scheduler = None#CosineAnnealingWarmUpRestarts(self.optimizer, \
                                                     #  T_0=50, T_mult=1, eta_max=0.01,  T_up=25, gamma=0.8)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.995 ** epoch)
    
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa, self.entropie = [], []

        self.logpa_sum, self.G_ts, self.entropies = [], [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze() 
        
        self.logpa_sum.append(logpa.sum())
        self.entropies.append(entropie.mean())
        self.G_ts.append(G_t)

        self.logpa, self.entropie = [], []

    def decision(self, machine_feature, task_feature, D_TM, ninf_mask):
        machine_feature = machine_feature.to(self.device)
        task_feature = task_feature.to(self.device)
        D_TM = D_TM.to(self.device)

        if self.skip:
            skip_mask = torch.zeros(size=(1, self.machine_num, 1))
            ninf_mask = torch.cat((skip_mask, ninf_mask), dim=2)
            ninf_mask = ninf_mask.to(self.device)
        else:
            ninf_mask = ninf_mask.to(self.device)

        if self.model.training:
            probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]

            prob_size = torch.tensor(probs.size(1), device=self.device)
            entropie = torch.clip(dist.entropy() / prob_size.log(), min=0, max=1)

            self.logpa.append(logpa) ; self.entropie.append(entropie)

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self, entropy_loss_weight):
        G_T = torch.tensor(self.G_ts, dtype=torch.float32).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()
        entropies = torch.stack(self.entropies).squeeze()

        advantage_t = G_T - G_T.mean()

        # loss
        policy_loss = (-advantage_t.detach() * logpas).mean()
        entropie_loss = -entropies.mean()
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + entropy_loss_weight * entropie_loss)
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        self.logpa_sum, self.G_ts, self.entropies = [], [], []
        return loss_mean.detach().cpu().numpy()

class BGCM():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = MultiStepLR(self.optimizer, **cfg.optimizer_params['scheduler'])

        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa, self.entropie = [], []

        self.logpa_sum, self.G_ts, self.entropies = [], [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze() 
        
        self.logpa_sum.append(logpa.sum())
        self.entropies.append(entropie.mean())
        self.G_ts.append(G_t)

        self.logpa, self.entropie = [], []

    def decision(self, machine_feature, task_feature, D_TM, ninf_mask):
        machine_feature = machine_feature.to(self.device)
        task_feature = task_feature.to(self.device)
        D_TM = D_TM.to(self.device)

        if self.skip:
            skip_mask = torch.zeros(size=(1, self.machine_num, 1))
            ninf_mask = torch.cat((skip_mask, ninf_mask), dim=2)
            ninf_mask = ninf_mask.to(self.device)
        else:
            ninf_mask = ninf_mask.to(self.device)

        if self.model.training:
            probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            entropie = dist.entropy()

            self.logpa.append(logpa) ; self.entropie.append(entropie)

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self, entropy_loss_weight):
        G_T = torch.tensor(self.G_ts, dtype=torch.float32).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        advantage_t = G_T - G_T.mean()

        # loss
        policy_loss = (-advantage_t.detach() * logpas).mean()
        entropie_loss = -entropies.mean()
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + entropy_loss_weight * entropie_loss)
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        self.logpa_sum, self.G_ts, self.entropies = [], [], []
        return loss_mean.detach().cpu().numpy()
