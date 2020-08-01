import os.path as osp
import numpy as np
import random
import os
from sys import argv
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from gym.spaces import Discrete
from gym import spaces
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

class Net(torch.nn.Module):
    def __init__(self, max_layer=10, dataset='Cora'):
        self.hidden = []
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        for i in range(max_layer - 2):
            self.hidden.append(GCNConv(16, 16, cached=True))
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)


    def forward(self, action, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for i in range(action-2):
            x = F.relu(self.hidden[i](x, edge_index))
            x = F.dropout(x, training=self.training)
        self.embedding = self.conv2(x, edge_index)
        return F.log_softmax(self.embedding, dim=1)

class gcn_env(object):
    def __init__(self, dataset='Cora', lr=0.01, weight_decay=5e-4, max_layer=10, batch_size=128, policy=""):
        device = 'cpu'
        dataset = dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
        data = dataset[0]

        adj = to_dense_adj(data.edge_index).numpy()[0]
        norm = np.array([np.sum(row) for row in adj])
        self.adj = (adj/norm).T
        self.init_k_hop(max_layer)

        self.model, self.data = Net(max_layer, dataset).to(device), data.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        train_mask = self.data.train_mask.to('cpu').numpy()
        self.train_indexes = np.where(train_mask==True)[0]
        self.batch_size = len(self.train_indexes) - 1
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(max_layer)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.max_layer = max_layer

        # For Experiment #
        self.random = False
        self.gcn = False # GCN Baseline
        self.enable_skh = True # only when GCN is false will be useful
        self.enable_dlayer = True
        self.baseline_experience = 50
        
        # buffers for updating
        #self.buffers = {i: [] for i in range(max_layer)}
        self.buffers = defaultdict(list)
        self.past_performance = [0]

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def init_k_hop(self, max_hop):
        sp_adj = csr_matrix(self.adj)
        dd = sp_adj
        self.adjs = [dd]
        for i in range(max_hop):
            dd *= sp_adj
            self.adjs.append(dd)

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max) 

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.model.train()
        self.optimizer.zero_grad()
        if self.random == True:
            action = random.randint(1,5)
        # train one step
        index = self.train_indexes[self.i]
        pred = self.model(action, self.data)[index]
        pred = pred.unsqueeze(0)
        y = self.data.y[index]
        y = y.unsqueeze(0)
        F.nll_loss(pred, y).backward()
        self.optimizer.step()
        
        # get reward from validation set
        val_acc = self.eval_batch()

        # get next state
        self.i += 1
        self.i = self.i % len(self.train_indexes)
        next_index = self.train_indexes[self.i]
        #next_state = self.data.x[next_index].to('cpu').numpy()
        next_state = self.data.x[next_index].numpy()
        if self.i == 0:
            done = True
        else:
            done = False
        return next_state, val_acc, done, "debug"

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.data.x[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        self.model.train()
        self.optimizer.zero_grad()
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        done = False
        for act, idx in zip(actions, index):
            if self.gcn == True or self.enable_dlayer == False:
                act = self.max_layer
            self.buffers[act].append(idx)
            if len(self.buffers[act]) >= self.batch_size:
                self.train(act, self.buffers[act])
                self.buffers[act] = []
                done = True
        if self.gcn == True or self.enable_skh == False:
            ### Random ###
            self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
            start = self.i
            end = (self.i + self.batch_size) % len(self.train_indexes)
            index = self.train_indexes[start:end]
        else:
            index = self.stochastic_k_hop(actions, index) 
        next_state = self.data.x[index].to('cpu').numpy()
        #next_state = self.data.x[index].numpy()
        val_acc_dict = self.eval_batch()
        val_acc = [val_acc_dict[a] for a in actions]
        test_acc = self.test_batch()
        baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
        self.past_performance.extend(val_acc)
        reward = [100 * (each - baseline) for each in val_acc] # FIXME: Reward Engineering
        r = np.mean(np.array(reward))
        return next_state, reward, [done]*self.batch_size, (val_acc, r)

    #def eval_step2(self, actions):
    #    self.model.eval()
    #    start = self.i
    #    end = (self.i + self.batch_size) % len(self.train_indexes)
    #    index = self.train_indexes[start:end]
    #    done = False
    #    for act, idx in zip(actions, index):
    #        if self.gcn == True or self.enable_dlayer == False:
    #            act = self.max_layer
    #        self.buffers[act].append(idx)
    #        if len(self.buffers[act]) >= self.batch_size:
    #            self.train(act, self.buffers[act])
    #            self.buffers[act] = []
    #            done = True

    #    if self.gcn == True or self.enable_skh == False:
    #        ### Random ###
    #        self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
    #        start = self.i
    #        end = (self.i + self.batch_size) % len(self.train_indexes)
    #        index = self.train_indexes[start:end]
    #    else:
    #        index = self.stochastic_k_hop(actions, index) 
    #    next_state = self.data.x[index].to('cpu').numpy()
    #    val_acc_dict = self.eval_batch() # val_acc is a dict, key --> actions, value --> accuracy.
    #    val_acc = [val_acc_dict[a] for a in actions]
    #    #test_acc = self.test_batch()
    #    baseline = np.mean(np.array(self.past_performance))
    #    self.past_performance.extend(val_acc)
    #    #reward = [val_acc - baseline] * self.batch_size
    #    reward = [each - baseline for each in val_acc]
    #    return next_state, reward, [done]*self.batch_size, (val_acc, test_acc)

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            prob = self.adjs[act].getrow(idx).toarray().flatten()
            cand = np.array([i for i in range(len(prob))])
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, action, indexes):
        self.model.train()
        pred = self.model(action, self.data)[indexes]
        y = self.data.y[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()
        
    def eval_batch(self):
        self.model.eval()
        batch_dict = {}
        val_index = np.where(self.data.val_mask.to('cpu').numpy()==True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(val_index)) 
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(val_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(val_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        #acc = 0.0
        acc = {a: 0.0 for a in range(self.max_layer)}
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.data) 
            pred = logits[idx].max(1)[1]
            #acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
            acc[a] = pred.eq(self.data.y[idx]).sum().item() / len(idx)
        #acc = acc / len(batch_dict.keys())
        return acc

    def test_batch(self):
        self.model.eval()
        batch_dict = {}
        test_index = np.where(self.data.test_mask.to('cpu').numpy()==True)[0]
        val_states = self.data.x[test_index].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(test_index)) 
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(test_index), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(test_index, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        acc = 0.0
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.data) 
            pred = logits[idx].max(1)[1]
            acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
        acc = acc / len(batch_dict.keys())
        return acc

    def check(self):
        self.model.eval()
        train_index = np.where(self.data.train_mask.to('cpu').numpy()==True)[0]
        tr_states = self.data.x[train_index].to('cpu').numpy()
        tr_acts = self.policy.eval_step(tr_states)

        val_index = np.where(self.data.val_mask.to('cpu').numpy()==True)[0]
        val_states = self.data.x[val_index].to('cpu').numpy()
        val_acts = self.policy.eval_step(val_states)

        test_index = np.where(self.data.test_mask.to('cpu').numpy()==True)[0]
        test_states = self.data.x[test_index].to('cpu').numpy()
        test_acts = self.policy.eval_step(test_states)

        return (train_index, tr_states, tr_acts), (val_index, val_states, val_acts), (test_index, test_states, test_acts)

