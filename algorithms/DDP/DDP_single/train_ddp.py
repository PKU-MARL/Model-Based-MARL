
from torch import nn

#----------------------------------------------------------------------
from dataset import Dataset
from algorithms.utils import Config, LogClient, LogServer, mem_report
# from algorithms.modelx import GraphConvolutionalModel
import sys  # 导入sys模块
sys.setrecursionlimit(20000)  # 将默认的递归深度修改为3000
from re import S
from algorithms.envs.flow import networks
from math import log
import numpy as np
import ipdb as pdb
import itertools
from gym.spaces import Box, Discrete
import random

import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

#----------------------------------------------------------------------


import torch.distributed  as dist
from torch.nn.parallel import DistributedDataParallel
import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# from algorithms.models import GraphConvolutionalModel
from torch.optim import Adam
import numpy as np

# def get_args():
#     parser = argparse.ArgumentParser(description="Train program use DDP")
#
#     # ddp运行所需参数
#     parser.add_argument("--local_rank", type=int ,default=-1, help="local_rank for ddp")
#     # ddp训练过程中意外终止代码可能导致进程驻留，about参数帮助查看相关进程，并杀死
#     parser.add_argument("--about", type=str, default="ddp", help="description for ddp process")
#     # 观察程序运行过程中显存的使用情况，显存空余较大情况，可提高batch_size
#     parser.add_argument("--batch_size", type=int, default=16, help="batch size")
#     # 数据集加载过程中多线程参数，参数越大，数据加载越快，可根据cpu核心数调整
#     parser.add_argument("--workers", type=int, default=8, help="num woekers")
#     parser.add_argument("--save_path", type=str, default="./checkpoints", help="model save path")
#     parser.add_argument("--epochs", type=int, default=1000, help="mutil train model")
#
#     args = parser.parser_args()
#     return args



class GraphConvolutionalModel(nn.Module):
    class EdgeNetwork(nn.Module):
        def __init__(self, i, j, sizes, activation=nn.ReLU, output_activation=nn.Identity):
            super().__init__()
            self.i = i
            self.j = j
            layers = []
            for t in range(len(sizes) - 1):
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
            self.net = nn.Sequential(*layers)
        
        def forward(self, s:torch.Tensor):
            """
            Input: [batch_size, n_agent, node_embed_dim] # raw input
            Output: [batch_size, edge_embed_dim]
            """
            s1 = s.select(dim=1, index=self.i)
            s2 = s.select(dim=1, index=self.j)
            s = torch.cat([s1, s2], dim=-1)
            return self.net(s)
    
    class NodeNetwork(nn.Module):
        def __init__(self, sizes, n_embedding=0, action_dim=0, activation=nn.ReLU, output_activation=nn.ReLU):
            super().__init__()
            layers = []
            for t in range(len(sizes) - 1):
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
            self.net = nn.Sequential(*layers)
            
            
            if n_embedding != 0:
                self.action_embedding_fn = nn.Embedding(action_dim, n_embedding)
                self.action_embedding = lambda x: self.action_embedding_fn(x.squeeze(-1))
            else:
                self.action_embedding = nn.Identity()

        def forward(self, h_ls, a):
            """
            Input: 
                h_ls: list of tensors with sizes of [batch_size, edge_embed_dim]
                a: [batch_size, action_dim]
            Output: 
                h: [batch_size, node_embed_dim]
            """
            embedding = 0
            for h in h_ls:
                embedding += h
            a = self.action_embedding(a)
            while a.ndim < embedding.ndim:
                a = a.unsqueeze(-1)
            embedding = torch.cat([embedding, a], dim=-1)
            return self.net(embedding)

    class NodeWiseEmbedding(nn.Module):
        def __init__(self, n_agent, input_dim, output_dim, output_activation):
            super().__init__()
            self.nets = nn.ModuleList()
            self.n_agent = n_agent
            for _ in range(n_agent):
                self.nets.append(nn.Sequential(*[nn.Linear(input_dim, output_dim), output_activation()]))
        
        def forward(self, h):
            # input dim = 3, output the same
            items = []
            for i in range(self.n_agent):
                items.append(self.nets[i](h.select(dim=1, index=i)))
            items = torch.stack(items, dim=1)
            return items

    def __init__(self, adj, state_dim, action_dim, n_agent):
        super().__init__()
        # self.logger = logger.child("p")
        self.adj = adj > 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agent = n_agent
        self.n_conv = 1
        self.n_embedding = 0
        self.residual = True
        self.edge_embed_dim = 12
        self.edge_hidden_size =  [16, 16]
        self.node_embed_dim = 8
        self.node_hidden_size =  [16, 16]
        self.reward_coeff = 1

        self.node_nets = self._init_node_nets()
        self.edge_nets = self._init_edge_nets()
        self.node_embedding, self.state_head, self.reward_head, self.done_head = self._init_node_embedding()
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCELoss(reduction='none')


        # ddp部分
        # self.args = args
        # self.device = torch.device("cuda", args.local_rank)
        # self.observation_dim = observation_dim
        # self.action_dim = action_dim
        # self.n_agent = n_agent

        # self.model = GraphConvolutionalModel(adj, observation_dim, action_dim, n_agent).to(self.device)
        # self.model.to(self.device)
        # self.optimizer_p = Adam(self.model.parameters(), lr=0.001)

        # #此处的model是数据集加载部分产生数据的model，请勿混淆
        # #数据集加载代码，所需参数根据自身需要修改
        # self.train_data = Dataset(model, samples_num=2000, batch_size=256)
        # self.train_data = data
        # sample = DistributedSampler(self.train_data)
        # self.train_dataloader = DataLoader(self.train_data, num_workers=args.workers, batch_size=args.batch_size, sampler=sample, shuffle=False)



    
    def predict(self, s, a):
        """
            Input: 
                s: [batch_size, n_agent, state_dim]
                a: [batch_size, n_agent, action_dim]
            Output: [batch_size, n_agent, state_dim] # same as input state
        """
        with torch.no_grad():
            r1, s1, d1 = self.forward(s, a)
            done = torch.clamp(d1, 0., 1.)
            done = torch.cat([1 - done, done], dim=-1)
            done = Categorical(done).sample() > 0  # [b]
            return r1, s1, done
    
    def forward(self, s, a, r, s1, d, length = 1):
        # print('a=',s.shape)
       
        """
        Input shape: [batch_size, T, n_agent, dim]
        """
        pred_s, pred_r, pred_d = [], [], []
        s0 = s.select(dim=1, index=0)
        length = min(length, s.shape[1])
        for t in range(length):
            r_, s_, d_ = self.f(s0, a.select(dim=1, index=t))
            pred_r.append(r_)
            pred_s.append(s_)
            pred_d.append(d_)
            s0 = s_
        reward_pred = torch.stack(pred_r, dim=1)
        state_pred = torch.stack(pred_s, dim=1)
        done_pred = torch.stack(pred_d, dim=1)

        state_loss = self.MSE(state_pred, s1).mean()  
        s1_view = s1.view(-1, s1.shape[-1])
        state_var = self.MSE(s1_view, s1_view.mean(dim=0, keepdim=True).expand(*s1_view.shape))
        rel_state_loss = state_loss / (state_var.mean() + 1e-7)
        # self.logger.log(state_loss=state_loss, state_var=state_var.mean(), rel_state_loss=rel_state_loss)
        loss = state_loss

        reward_loss = self.MSE(reward_pred, r)
        loss += self.reward_coeff * reward_loss.mean()
        r_view = r.view(-1, r.shape[-1])
        reward_var = self.MSE(r_view, r_view.mean(dim=0, keepdim=True).expand(*r_view.shape)).mean()
        rel_reward_loss = reward_loss.mean() / (reward_var.mean() + 1e-7)

        # self.logger.log(reward_loss=reward_loss,
        #                 reward_var=reward_var,
        #                 reward=r,
        #                 reward_norm=torch.norm(r),
        #                 rel_reward_loss=rel_reward_loss)

        d = d.float()
        done_loss = self.BCE(done_pred, d)
        loss = loss + done_loss.mean()
        done = done_pred > 0
        done_true_positive = (done * d).mean()
        d = d.mean()
        # self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

        return (loss, rel_state_loss)
    
    def f(self, s, a):
        """
            Input: [batch_size, n_agent, state_dim]
            Output: [batch_size, n_agent, state_dim]
        """
        embedding = self.node_embedding(s) # dim = 3
        for _ in range(self.n_conv):
            edge_info_of_nodes = [[] for __ in range(self.n_agent)]
            for edge_net in self.edge_nets:
                edge_info = edge_net(embedding) # dim = 2
                edge_info_of_nodes[edge_net.i].append(edge_info)
                edge_info_of_nodes[edge_net.j].append(edge_info)
            node_preds = []
            for i in range(self.n_agent):
                node_net = self.node_nets[i]
                node_pred = node_net(edge_info_of_nodes[i], a.select(dim=1, index=i)) # dim = 2
                node_preds.append(node_pred)
            embedding = torch.stack(node_preds, dim=1) # dim = 3
        state_pred = self.state_head(embedding)
        if self.residual:
            state_pred += s
        reward_pred = self.reward_head(embedding)
        done_pred = self.done_head(embedding)
        return reward_pred, state_pred, done_pred

    def _init_node_nets(self):
        node_nets = nn.ModuleList()
        action_dim = self.n_embedding if self.n_embedding > 0 else self.action_dim
        sizes = [self.edge_embed_dim + action_dim] + self.node_hidden_size + [self.node_embed_dim]
        for i in range(self.n_agent):
            node_nets.append(GraphConvolutionalModel.NodeNetwork(sizes=sizes, n_embedding=self.n_embedding, action_dim=self.action_dim))
        return node_nets

    def _init_edge_nets(self):
        edge_nets = nn.ModuleList()
        sizes = [self.node_embed_dim * 2] + self.edge_hidden_size + [self.edge_embed_dim]
        for i in range(self.n_agent):
            for j in range(i + 1, self.n_agent):
                if self.adj[i][j]:
                    edge_nets.append(GraphConvolutionalModel.EdgeNetwork(i, j, sizes))
        return edge_nets

    def _init_node_embedding(self):
        node_embedding = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.state_dim, self.node_embed_dim, output_activation=nn.ReLU)
        state_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, self.state_dim, output_activation=nn.Identity)
        reward_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Identity)
        done_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Sigmoid)
        return node_embedding, state_head, reward_head, done_head













class Model(nn.Module):
    def __init__(self,adj,observation_dim,action_dim,n_agent,data,args):
        super().__init__()
        # ddp部分
        self.adj = adj
        self.args = args
        self.device = torch.device("cuda", args.local_rank)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_agent = n_agent

        self.model = GraphConvolutionalModel(adj, observation_dim, action_dim, n_agent).to(self.device)
        self.model.to(self.device)
        self.optimizer_p = Adam(self.model.parameters(), lr=0.001)

        # 此处的model是数据集加载部分产生数据的model，请勿混淆
        # 下列代码为数据集加载代码，所需参数根据自身需要修改
        # self.train_data = Dataset(model, samples_num=2000, batch_size=256)
        self.train_data = data
        sample = DistributedSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, num_workers=args.workers, batch_size=args.batch_size, sampler=sample, shuffle=False)



    def updateModel(self, length=1):
        min_train_loss = 10000
        
        for epoch in range(self.args.epochs):
        #ss,actions,rs, sls, ds是数据，是tensor/list的形式，ss, actions, rs, s1s, ds = [...], [...], [...], [...],[...]
        #train模型
            if self.args.local_rank != -1:
                self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)

                train_loss = 0
                self.model.train()
                

            for step, train_batch in enumerate(self.train_dataloader):
                ss, actions, rs, s1s, ds = train_batch
                
                ss = ss[0]
                actions = actions[0]
                rs = rs[0]
                s1s = s1s[0]
                ds = ds[0]

                loss, rel_state_error = self.model(ss, actions, rs, s1s, ds, length) # [n_traj, T, n_agent, dim]
                self.optimizer_p.zero_grad()
                loss.sum().backward()
                self.optimizer_p.step()
                train_loss += loss.item() * len(train_batch)

            train_loss /= len(self.train_dataloader)
            print("Epoch: %d, train_loss: %.6f" % (epoch, train_loss))
            if self.args.local_rank == 0:
                if train_loss < min_train_loss:
                    save_file = '%s/DG_rotate_epoch_%04d_loss_%.6f.pth' % (self.args.save_path, epoch, train_loss)
                    torch.save(self.model.state_dict(), save_file)
                    min_train_loss = train_loss


        # return rel_state_error.item()



if __name__ == "__main__":
    


    
    
    def get_args():
        parser = argparse.ArgumentParser(description="Train program use DDP")

        # ddp运行所需参数
        parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for ddp")
        # ddp训练过程中意外终止代码可能导致进程驻留，about参数帮助查看相关进程，并杀死
        parser.add_argument("--about", type=str, default="ddp", help="description for ddp process")
        # 观察程序运行过程中显存的使用情况，显存空余较大情况，提高batch_size
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        # 数据集加载过程中多线程参数，参数越大，数据加载越快，可根据cpu核心数调整
        parser.add_argument("--workers", type=int, default=8, help="num woekers")
        parser.add_argument("--save_path", type=str, default="./checkpoints", help="model save path")
        parser.add_argument("--epochs", type=int, default=10, help="mutil train model")

        args = parser.parse_args()
        return args


    args = get_args()
    args.device = "cuda:%s" % args.local_rank
    
    torch.cuda.set_device(args.device)
    
    #需要传入的数据 length,logger,adj,observation_dim,action_dim,n_agent,p_args,data
    #-----------------------------------------------------------------------   
    data = Dataset(1, samples_num=200, batch_size=256,length=4)
    adj = torch.as_tensor([[1]*28]*28, device='cuda', dtype=torch.float)
    observation_dim = 10
    action_dim = 1
    n_agent = 28

#---------------------------------------------------------------------------------------
    
    

    dist.init_process_group(backend="nccl", init_method="env://")

    # model = GraphConvolutionalModel(adj, 10, action_dim, n_agent, args)

    model = Model(adj,observation_dim,action_dim,n_agent,data,args)   
    
    
    #model.to('cuda')
    model.updateModel(4)

    if args.local_rank == 0:
        dist.destroy_process_group()

    torch.cuda.empty_cache()

    