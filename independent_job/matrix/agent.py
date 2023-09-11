import torch

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from independent_job.matrix.model import CloudMatrixModel, CloudMatrixModelposition, \
                                        CloudMatrixModel_one, CloudMatrixModel_one_pose

class BGC():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModelposition(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.G_t_loss_weight = cfg.model_params['G_t_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa, self.G_t_pred, self.entropie = [], [], []

        self.logpa_sum, self.G_ts, self.entropies = [], [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        # G_t_pred = torch.stack(self.G_t_pred).squeeze()
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze() 
        
        self.logpa_sum.append(logpa.sum())
        self.entropies.append(entropie.sum())
        self.G_ts.append(G_t)
        # print(self.logpa_sum, self.entropies, self.G_ts)

        self.logpa, self.G_t_pred, self.entropie = [], [], []

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
            probs, G_t_pred = \
                    self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            entropie = dist.entropy()

            self.logpa.append(logpa) ; self.entropie.append(entropie)
            self.G_t_pred.append(G_t_pred)

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs, _ = \
                        self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self):
        G_T = torch.tensor(self.G_ts).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        # G_t_pred = torch.stack(self.G_t_pred).squeeze()

        advantage_t = G_T - G_T.mean()

        # loss
        # critic_loss = (G_T - G_t_pred).pow(2).mul(0.5).mean()
        policy_loss = (-advantage_t.detach() * logpas).mean()
        entropie_loss = -entropies.mean()
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + self.entropy_loss_weight * entropie_loss)
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # self.model_save()
        print(f"policy : {self.policy_loss_weight * policy_loss.detach().cpu().numpy()}")
        print(f"entropy : {self.entropy_loss_weight * entropie_loss.detach().cpu().numpy()}")
        # print(f"critic : {self.G_t_loss_weight * critic_loss.detach().cpu().numpy()}")

        self.logpa_sum, self.G_ts, self.entropies = [], [], []
        return loss_mean.detach().cpu().numpy()

class BGCmean():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel_one(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.G_t_loss_weight = cfg.model_params['G_t_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa, self.G_t_pred, self.entropie = [], [], []

        self.logpa_sum, self.G_ts, self.entropies = [], [], []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

    def trajectory(self, G_t):
        # G_t_pred = torch.stack(self.G_t_pred).squeeze()
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze() 
        
        self.logpa_sum.append(logpa.sum())
        self.entropies.append(entropie.mean())
        self.G_ts.append(G_t)
        # print(self.logpa_sum, self.entropies, self.G_ts)

        self.logpa, self.G_t_pred, self.entropie = [], [], []

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
            probs, G_t_pred = \
                    self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            entropie = dist.entropy()

            self.logpa.append(logpa) ; self.entropie.append(entropie)
            self.G_t_pred.append(G_t_pred)

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs, _ = \
                        self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self):
        G_T = torch.tensor(self.G_ts).to(self.device)
        logpas = torch.stack(self.logpa_sum).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        # G_t_pred = torch.stack(self.G_t_pred).squeeze()

        advantage_t = G_T - G_T.mean()

        # loss
        # critic_loss = (G_T - G_t_pred).pow(2).mul(0.5).mean()
        policy_loss = (-advantage_t.detach() * logpas).mean()
        entropie_loss = -entropies.mean()
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + self.entropy_loss_weight * entropie_loss)
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # self.model_save()
        print(f"policy : {self.policy_loss_weight * policy_loss.detach().cpu().numpy()}")
        print(f"entropy : {self.entropy_loss_weight * entropie_loss.detach().cpu().numpy()}")
        # print(f"critic : {self.G_t_loss_weight * critic_loss.detach().cpu().numpy()}")

        self.logpa_sum, self.G_ts, self.entropies = [], [], []
        return loss_mean.detach().cpu().numpy()
    
class BGCQ():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.gamma = 0.99
        self.model = CloudMatrixModel(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.buffer = cfg.buffer

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.G_t_loss_weight = cfg.model_params['G_t_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.train_V = {}
        self.train_V['G_t_pred'] = []

    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

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
            probs, _ = \
                    self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs, _ = \
                        self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self):
        machine_feature, \
            task_feature, \
            D_TM, \
            ninf_mask, \
            reward, \
            action, \
            done, \
            machine_feature_s_prime, \
            task_feature_s_prime, \
            D_TM_s_prime, \
            ninf_mask_s_prime, \
            reward_s_prime, \
            action_s_prime, \
            done_s_prime = self.buffer.sample()

        machine_feature = machine_feature.to(self.device)
        task_feature = task_feature.to(self.device)
        D_TM = D_TM.to(self.device)
        ninf_mask = ninf_mask.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        
        machine_feature_s_prime = machine_feature_s_prime.to(self.device)
        task_feature_s_prime = task_feature_s_prime.to(self.device)
        D_TM_s_prime = D_TM_s_prime.to(self.device)
        ninf_mask_s_prime = ninf_mask_s_prime.to(self.device)

        probs, G_t_pred = \
                self.model(machine_feature, task_feature, D_TM, ninf_mask)
        dist = torch.distributions.Categorical(probs)
        task_selected = dist.sample()
        logpa = dist.log_prob(task_selected)
        entropies = dist.entropy()


        self.train_V['G_t_pred'].append(G_t_pred.tolist())


        _, G_t_next = \
                self.model(machine_feature_s_prime, task_feature_s_prime,
                                    D_TM_s_prime, ninf_mask_s_prime)

        target_q_sa = (reward + (self.gamma * G_t_next.squeeze() * (1 - done.int()))).detach()

        policy_loss = -(G_t_pred.squeeze().detach() * logpa).mean()
        entropie_loss = -entropies.mean()
        critic_loss = (G_t_pred.squeeze() - target_q_sa).pow(2).mul(0.5).mean()
        
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + self.entropy_loss_weight * entropie_loss \
                     + self.G_t_loss_weight * critic_loss)
        
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # self.model_save()
        print(f"policy : {self.policy_loss_weight * policy_loss.detach().cpu().numpy()}")
        print(f"entropy : {self.entropy_loss_weight * entropie_loss.detach().cpu().numpy()}")
        print(f"critic : {self.G_t_loss_weight * critic_loss.detach().cpu().numpy()}")
        return loss_mean.detach().cpu().item()

class BGCC():
    def __init__(self, cfg):
        self.device = cfg.model_params['device']
        self.model = CloudMatrixModel(**cfg.model_params).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **cfg.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **cfg.optimizer_params['scheduler'])
        self.save_path = cfg.model_params['save_path']
        self.load_path = cfg.model_params['load_path']
        self.skip = cfg.model_params['skip']
        self.machine_num = cfg.machines_number

        self.policy_loss_weight = cfg.model_params['policy_loss_weight']
        self.G_t_loss_weight = cfg.model_params['G_t_loss_weight']
        self.entropy_loss_weight = cfg.model_params['entropy_loss_weight'] 

        if self.load_path:
            print(f"load weight : {self.load_path}")
            self.model.load_state_dict(torch.load(cfg.model_params['load_path'],
                                                  map_location=self.device))
            self.model.eval()

        self.logpa, self.G_t_pred, self.entropie = [], [], []
    
        self.train_V = {}
        self.train_V['G_t'] = []
        self.train_V['G_t_pred'] = []
        
        
    def model_save(self, save_path=None):
        if save_path == None:
            save_path = self.save_path
        torch.save(self.model.state_dict(), save_path)

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
            probs, G_t_pred = \
                    self.model(machine_feature, task_feature, D_TM, ninf_mask)
            # [B, M*T]
            dist = torch.distributions.Categorical(probs)
            task_selected = dist.sample()
            # [B,]
            logpa = dist.log_prob(task_selected)
            # [B,]
            entropie = dist.entropy()

            self.logpa.append(logpa) ; self.entropie.append(entropie)
            self.G_t_pred.append(G_t_pred)

            return task_selected.detach().cpu().item()

        else:
            with torch.no_grad():
               probs, _ = \
                        self.model(machine_feature, task_feature, D_TM, ninf_mask)
            task_selected = probs.argmax(dim=1)
            return task_selected.detach().cpu().item()

    def optimize_model(self, G_T):
        G_t_pred = torch.stack(self.G_t_pred).squeeze()
        logpa = torch.stack(self.logpa).squeeze()
        entropie = torch.stack(self.entropie).squeeze()
        
        self.train_V['G_t_pred'].append(G_t_pred.tolist())
        self.train_V['G_t'].append(G_T)
        

        # loss
        critic_loss = (G_T - G_t_pred).pow(2).mul(0.5).mean()
        policy_loss = (-G_t_pred.detach() * logpa).mean()
        entropie_loss = -entropie.mean()
        loss_mean = (self.policy_loss_weight * policy_loss \
                     + self.G_t_loss_weight * critic_loss \
                     + self.entropy_loss_weight * entropie_loss)

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()


        self.model_save()
        print(f"policy : {self.policy_loss_weight * policy_loss.detach().cpu().numpy()}")
        print(f"entropy : {self.entropy_loss_weight * entropie_loss.detach().cpu().numpy()}")
        print(f"critic : {self.G_t_loss_weight * critic_loss.detach().cpu().numpy()}")

        self.logpa, self.G_t_pred, self.entropie = [], [], []
        return loss_mean.detach().cpu().numpy()