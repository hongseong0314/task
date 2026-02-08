import torch
import numpy as np
import copy
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR
from independent_job.matrix.model import CloudMatrixModel, CloudMatrixModel_one_pose, CloudMatrixModel2
from independent_job.matrix.utill import CosineAnnealingWarmUpRestarts
from torch.utils.checkpoint import checkpoint

class BGCDSACA():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel2(**cfg.model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25, eta_min=1e-5)
        self.scheduler = MultiStepLR(self.optimizer, **cfg.optimizer_params['scheduler'])
        # self.scheduler = None#CosineAnnealingWarmUpRestarts(self.optimizer, \
                                                     #  T_0=50, T_mult=1, eta_max=0.01,  T_up=25, gamma=0.8)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.995 ** epoch)

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

        self.entropie, self.G_ts, self.entropies = [], [], []
        
        self.machine_features, self.task_features, \
        self.D_TM, self.masks, self.actions = [], [], [], [], []

        self.b_machine_features, self.b_task_features, \
        self.b_D_TM, self.b_masks, self.b_actions = [], [], [], [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        self.b_machine_features.append(self.machine_features)
        self.b_task_features.append(self.task_features)
        self.b_D_TM.append(self.D_TM)
        self.b_masks.append(self.masks)
        self.b_actions.append(self.actions)

        entropie = torch.stack(self.entropie).squeeze()
        self.entropies.append(entropie.mean())
        self.G_ts.append(G_t)

        
        self.machine_features, self.task_features, \
        self.D_TM, self.masks, self.actions = [], [], [], [], []
        self.entropie = []

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
            with torch.no_grad():
               probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            # print(f"probs : {probs}")
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            prob_size = torch.tensor(probs.size(1), device=self.device)
            entropie = torch.clip(dist.entropy() / prob_size.log(), min=0, max=1)
            
            self.machine_features.append(machine_feature.detach())
            self.task_features.append(task_feature.detach())
            self.D_TM.append(D_TM.detach())
            self.masks.append(ninf_mask.detach())
            self.actions.append(task_selected.detach())
            self.entropie.append(entropie.detach())

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self, entropy_loss_weight):
        G_T = torch.tensor(self.G_ts, dtype=torch.float32).to(self.device)
        entropies = torch.stack(self.entropies).squeeze()

        # Advantage Compute
        advantage_t = G_T * 10 + entropy_loss_weight * entropies
        advantage_t = advantage_t - advantage_t.mean()
        # print(f"advantage_t {advantage_t} entropies : {entropies}")
        trj_loss = 0
        for mfs, tfs, dtms, masks, aes, adv in zip(self.b_machine_features,
                                                    self.b_task_features,
                                                    self.b_D_TM,
                                                    self.b_masks,
                                                    self.b_actions,
                                                   advantage_t):
            logpas, log_sums_sum = [], []
            for mf, tf, dtm, mask, ae in zip(mfs, tfs, dtms, masks, aes):
                mf = mf.to(self.device)
                tf = tf.to(self.device)
                dtm = dtm.to(self.device)
                mask = mask.to(self.device)
                ae = ae.to(self.device)
                with torch.set_grad_enabled(True):
                    probs = self.model(mf, tf, dtm, mask)
                    dist = torch.distributions.Categorical(probs)
                    logpa = dist.log_prob(ae)

                    prob_size = torch.tensor(probs.size(1), device=self.device)
                    entropie = torch.clip(dist.entropy() / prob_size.log(), min=0, max=1)

                logpas.append(logpa)
                log_sums_sum.append(entropie)
            logpas = torch.stack(logpas).squeeze().sum()
            log_sum = torch.stack(log_sums_sum).squeeze().mean()
            policy_loss = -(logpas * adv + entropy_loss_weight * log_sum) / 12
            # print(f"logpas : {logpas}, advantage_t {adv} log_sum {log_sum} policy_loss {policy_loss}")
            policy_loss.backward()
            trj_loss += policy_loss.detach().cpu()
        # policy_loss = -(advantage_t.detach() * logpas + 0.05 * alpha * entropies).mean()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       float('inf'))
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.entropie, self.G_ts, self.entropies = [], [], []

        self.b_machine_features, self.b_task_features, \
        self.b_D_TM, self.b_masks, self.b_actions = [], [], [], [], []
        return trj_loss, entropy_loss_weight, 0
    
class BGCDSAC():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel2(**cfg.model_params)
        if cfg.mutl_gpu:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25, eta_min=1e-5)
        self.scheduler = MultiStepLR(self.optimizer, **cfg.optimizer_params['scheduler'])
        # self.scheduler = None#CosineAnnealingWarmUpRestarts(self.optimizer, \
                                                     #  T_0=50, T_mult=1, eta_max=0.01,  T_up=25, gamma=0.8)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.995 ** epoch)
    
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

        self.logpa, self.log_sums, self.entropie = [], [], []

        self.logpa_sum, self.G_ts, self.entropies = [], [], []

        self.log_sums_sum = []

        entropy_lr = 0.01
        self.logalpha = torch.tensor(0, requires_grad=True, dtype=torch.float32, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.logalpha], lr=entropy_lr)

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        if isinstance(self.model, torch.nn.DataParallel): 
                torch.save(self.model.module.state_dict(), save_path) 
        else:
            torch.save(self.model.state_dict(), save_path) 
        # torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze() 
        log_sum = torch.stack(self.log_sums).squeeze() 

        self.logpa_sum.append(logpa.sum())
        self.entropies.append(entropie.mean())
        self.log_sums_sum.append(log_sum.mean())
        self.G_ts.append(G_t)

        self.logpa, self.log_sums, self.entropie = [], [], []

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
            # machine_feature.requires_grad = True
            # task_feature.requires_grad = True
            # D_TM.requires_grad = True
            # ninf_mask.requires_grad = True
            # # print(machine_feature)
            # probs = checkpoint(self.model, 
            #                     machine_feature, task_feature, D_TM, ninf_mask)
            probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            prob_size = torch.tensor(probs.size(1), device=self.device)
            # task_num = task_feature.size(1)
            # machine_mean_probs = probs.detach().reshape(-1, self.machine_num, task_num).mean(dim=-1)
            # machine_log = torch.nn.Softmax(dim=1)(machine_mean_probs).log().sum(dim=1)
            entropie = torch.clip(dist.entropy() / prob_size.log(), min=0, max=1)
            target_entropy = -1 / prob_size.log()
            # log_sum = (probs + 1e-6).log().sum(dim=1)
            # print(log_sum, prob_size, -log_sum - prob_size)
            self.logpa.append(logpa) ; self.entropie.append((entropie).detach())
            self.log_sums.append(entropie)

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
        log_sums_sum = torch.stack(self.log_sums_sum).squeeze()

        # loss
        # target_alpha = (entropies + self.target_entropy)
        # print(f"entropies : {entropies}")
        alpha_loss = (self.logalpha * entropies).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.logalpha.exp().detach() 

         # advantage_t = G_T - G_T.mean()
        advantage_t = G_T * 10 + entropy_loss_weight * entropies
        advantage_t = advantage_t - advantage_t.mean()

        policy_loss = -(logpas * advantage_t + entropy_loss_weight * log_sums_sum).mean()
        # policy_loss = -(advantage_t.detach() * logpas + 0.05 * alpha * entropies).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       float('inf'))   
        self.optimizer.step()
        # print(f"alpha : {alpha}")
        # print(f"alpha : {alpha} \
        # R()*P : {advantage_t} \
        # entorpy : {alpha * entropies} \
        # alpah * sum : {alpha * log_sums_sum}\
        # log sum : {logpas}")

        # print(f"total : {policy_loss}")
        self.logpa_sum, self.G_ts, self.entropies, self.log_sums_sum = [], [], [], []
        return policy_loss.detach().cpu().numpy()#, entropy_loss_weight, log_sums_sum.mean().detach().cpu().numpy()

class BGCDJ():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel_one_pose(**cfg.model_params).to(self.device)
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

        self.gamma = 1
        self.batch_machine_feature, self.batch_task_feature = [], []
        self.batch_D_TM, self.batch_ninf_mask = [], []
        self.batch_actions, self.batch_rewards = [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        pass

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

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

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

    def _loss(self, machine_feature, task_feature, D_TM, ninf_mask, action, G_t):
        machine_feature = machine_feature.to(self.device)
        task_feature = task_feature.to(self.device)
        D_TM = D_TM.to(self.device)
        ninf_mask = ninf_mask.to(self.device)

        probs = self.model(machine_feature, task_feature, D_TM, ninf_mask)
        # print(f"s size {state.shape} prob size {logits.shape}, {action}")
        # categorical
        dist = torch.distributions.Categorical(probs=probs)
        logp = dist.log_prob(torch.tensor(action).to(self.device))

        # logp = -F.nll_loss(F.softmax(logits), torch.tensor([action]).to(self.device))
        # # self.logps.append(-logp.detach().cpu().numpy())
        # self.avg.append(G_t)
        return -logp * torch.tensor(G_t).to(self.device)

    def optimize_model(self, entropy_loss_weight):
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
        print(len(self.batch_machine_feature), len(self.batch_actions), len(adv_n))
        for ms, ts, ds, masks, actions, adv in zip(self.batch_machine_feature, self.batch_task_feature,\
                                        self.batch_D_TM, self.batch_ninf_mask, \
                                        self.batch_actions, adv_n):
            self.logps, self.avg = [], []
            loss_by_trajectory = []
            cnt = 1
            # trajectory
            for m, t, d, mask, a, r in zip(ms, ts, ds, masks, actions, adv):
                if m is None or a is None:
                    continue
                loss = self._loss(m, t, d, mask, a, r)
                loss_by_trajectory.append(loss)
                loss_values.append(loss.detach().cpu().numpy())
                advantages__.append(r)

                if cnt % 500 == 0:
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
        self.batch_machine_feature, self.batch_task_feature = [], []
        self.batch_D_TM, self.batch_ninf_mask = [], []
        self.batch_actions, self.batch_rewards = [], []
        return np.mean(loss_values)
    
class BGCD():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel_one_pose(**cfg.model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **cfg.optimizer_params['optimizer'])
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       float('inf'))   
        self.optimizer.step()

        self.logpa_sum, self.G_ts, self.entropies = [], [], []
        return loss_mean.detach().cpu().numpy()

class BGCM():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel(**cfg.model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **cfg.optimizer_params['optimizer'])
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
        return loss_mean.detach().cpu().numpy(), 0, 0
