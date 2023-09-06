import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

class Fit(object):
    def __init__(self,cfg):
        super().__init__()
        self.device = cfg.model_params['device']
        self.model = Qnet(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa = []
        self.logpa_sum, self.G_ts = [], []

    def trajectory(self, G_t):
        logpa = torch.stack(self.logpa).squeeze()

        self.logpa_sum.append(logpa.sum())
        self.G_ts.append(G_t)
        self.logpa = []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def decision(self, feature):
        feature = feature.to(self.device)

        if self.model.training:
            logits = \
                    self.model(feature)
            # [B, M*T]
            dist = torch.distributions.Categorical(logits=logits)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            self.logpa.append(logpa)
            return task_selected.item()
        else:
            with torch.no_grad():
               logits = \
                        self.model(feature)
            task_selected = logits.argmax(dim=1)
            logpa = None
            return task_selected.item()

    def optimize_model(self):
        G_T = torch.tensor(self.G_ts).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()

        advantage_t = G_T - G_T.mean()

        policy_loss = (-advantage_t.detach() * logpas).mean()

        self.model.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # self.model_save()

        self.logpa_sum, self.G_ts = [], []
        return policy_loss.detach().cpu().numpy()

class Qnet(torch.nn.Module):
    def __init__(self, **model_params):
        super(Qnet, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1 , 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 9),
            torch.nn.Tanh(),
            torch.nn.Linear(9, 18),
            torch.nn.Tanh(),
            torch.nn.Linear(18, 9),
            torch.nn.Tanh(),
        )
        self.FC = torch.nn.Linear(9, 1)

        # self.feature_extract = torch.nn.Sequential(
        #     torch.nn.Linear(model_params['nT'] + model_params['nM'] , 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.Tanh(),
        # )
        # self.FC = torch.nn.Linear(16, 1)
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.FC(x)
        return x.squeeze(-1).unsqueeze(0)