import numpy as np
import torch
import torch.nn.functional as F
import copy

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

class Fit(object):
    def __init__(self,cfg):
        super().__init__()
        self.device = cfg.model_params['device']
        if cfg.model_params['size'] == 'big':
            self.model = QnetBig(**cfg.model_params).to(self.device)
        elif cfg.model_params['size'] == 'orign':
            self.model = Qnet(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = None #LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
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

    def optimize_model(self, entropy_weight):
        G_T = torch.tensor(self.G_ts, dtype=torch.float32).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()

        advantage_t = G_T - G_T.mean()

        policy_loss = (-advantage_t.detach() * logpas).mean()

        self.model.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # self.model_save()

        self.logpa_sum, self.G_ts = [], []
        return policy_loss.detach().cpu().numpy()

class Fit2(object):
    def __init__(self,cfg):
        super().__init__()
        self.gamma = 1
        self.reward_to_go = True
        self.device = cfg.model_params['device']
        if cfg.model_params['size'] == 'big':
            self.model = QnetBig(**cfg.model_params).to(self.device)
        elif cfg.model_params['size'] == 'orign':
            self.model = Qnet(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = None #LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.batch_featurs, self.batch_actions, self.batch_rewards = [], [], [] 
        self.action = None
    def trajectory(self, G_t):
        pass

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
            # self.action = task_selected.item()
            # self.feature = feature.detach(),clone()
            # print(self.action)
            # [B,]
            # logpa = dist.log_prob(task_selected)
            # [B,]
            # self.logpa.append(logpa)
            return task_selected.item()
        else:
            with torch.no_grad():
               logits = \
                        self.model(feature)
            task_selected = logits.argmax(dim=1)
            logpa = None
            return task_selected.item()
    
    def _G_t(self):
        # G_t 
        all_returns = []
        for rewards in self.batch_rewards:
            T = len(rewards)
            discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
            returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
            all_returns.append(returns)
        return all_returns

    def _compute_advantage(self, all_returns):
        adv_n = copy.deepcopy(all_returns)
        max_length = max([len(adv) for adv in adv_n])

        # pad
        for i in range(len(adv_n)):
            adv_n[i] = np.append(adv_n[i], np.zeros(max_length - len(adv_n[i])))

        adv_n = np.array(adv_n)
        adv_n = adv_n - adv_n.mean(axis=0)

        # origin 
        advs = [adv_n[i][:all_returns[i].shape[0]] for i in range(len(adv_n))]
        return advs

    def _loss(self, state, action, G_t):
        logits = self.model(state.to(self.device))
        # print(f"s size {state.shape} prob size {logits.shape}, {action}")
        # categorical
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(torch.tensor(action).to(self.device))

        # logp = -F.nll_loss(F.softmax(logits), torch.tensor([action]).to(self.device))
        # # self.logps.append(-logp.detach().cpu().numpy())
        # self.avg.append(G_t)
        
        return -logp * torch.tensor(G_t).to(self.device)

    def optimize_model(self, entropy_weight):
        all_returns = self._G_t()
        adv_n = self._compute_advantage(all_returns)

        if True:  
            adv_s = []
            for advantages in adv_n:
                for advantage in advantages:
                    adv_s.append(advantage)
            adv_s = np.array(adv_s)
            mean = adv_s.mean()
            std = adv_s.std()
        adv_n__ = [(advantages - mean) / (std + np.finfo(np.float32).eps) for advantages in adv_n]
        adv_n = adv_n__

        # trajectorys   
        loss_values = []
        advantages__ = []
        print(len(self.batch_featurs), len(self.batch_actions), len(adv_n))
        for states, actions, adv in zip(self.batch_featurs, self.batch_actions, adv_n):
            self.logps, self.avg = [], []
            loss_by_trajectory = []
            cnt = 1
            # trajectory
            for s, a, r in zip(states, actions, adv):
                if s is None or a is None:
                    continue
                loss = self._loss(s, a, r)
                loss_by_trajectory.append(loss)
                loss_values.append(loss.detach().cpu().numpy())
                advantages__.append(r)
                
                if cnt % 1000 == 0:
                    self.optimizer.zero_grad()
                    policy_gradient = torch.stack(loss_by_trajectory).mean()
                    policy_gradient.backward()
                    self.optimizer.step()
                    loss_by_trajectory = []
                    print(policy_gradient.item())
                cnt += 1
            if len(loss_by_trajectory) > 0:
                self.optimizer.zero_grad()
                policy_gradient = torch.stack(loss_by_trajectory).mean()
                policy_gradient.backward()
                self.optimizer.step()
            # print(len(loss_by_trajectory))
        self.batch_featurs, self.batch_actions, self.batch_rewards = [], [], [] 
        return np.mean(loss_values), 0, 0
    
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
        # self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.FC(x)
        return x.squeeze(-1).unsqueeze(0)
    
class QnetBig(torch.nn.Module):
    def __init__(self, **model_params):
        super(QnetBig, self).__init__()

        ### parm small
        # self.feature_extract = torch.nn.Sequential(
        #     torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1 , 16),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(16, 22),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(22, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 16),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(16, 16),
        #     torch.nn.Tanh(),
        # )
        # self.FC = torch.nn.Linear(16, 1)

        # self.feature_extract = torch.nn.Sequential(
        #     torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1 , 9),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(9, 18),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(18, 120),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(120, 210),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(210, 18),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(18, 9),
        #     torch.nn.Tanh(),
        # )
        # self.FC = torch.nn.Linear(9, 1)

        ### parm base
        # self.feature_extract = torch.nn.Sequential(
        #     torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1 , 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 64),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(64, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(128, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 32),
        #     torch.nn.Tanh(),
        # )
        # self.FC = torch.nn.Linear(32, 1)
        
        ### parm big
        self.feature_extract = torch.nn.Sequential(
            torch.nn.Linear(model_params['nT'] + model_params['nM'] + 1 , 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 164),
            torch.nn.Tanh(),
            torch.nn.Linear(164, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
        )
        self.FC = torch.nn.Linear(128, 1)
        # self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.FC(x)
        return x.squeeze(-1).unsqueeze(0)