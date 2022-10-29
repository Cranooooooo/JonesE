import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class RouE(Model):
    def __init__(self, config):
        super(RouE, self).__init__(config)
        self.emb_Rou = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Thet = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.rel_Rou = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_Thet = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_Phi = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb_Rou.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi.weight.data)
        nn.init.xavier_uniform_(self.emb_Thet.weight.data)
        nn.init.xavier_uniform_(self.rel_Rou.weight.data)
        nn.init.xavier_uniform_(self.rel_Phi.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet.weight.data)

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou):
        # Transform
        ho_Phi = h_Phi + r_Phi
        ho_Thet = h_Thet + r_Thet
        ho_Rou = h_Rou + r_Rou

        # Calculate
        Rou_score = 1./(torch.abs(torch.log(ho_Rou/t_Rou))+1)
        Thet_score = torch.cos(t_Thet - ho_Thet)
        Phi_score = torch.cos(t_Phi - ho_Phi)
        score_r = (Rou_score+Thet_score+Phi_score)/3. - 0.5
        return -torch.sum(score_r, -1)
    

    def loss(self, score, regul, regul2, regul3):
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul
                +3*self.config.lmbda * regul2
                +3*self.config.lmbda * regul3)

    def prepare_enti_rela(self):
        h_Phi = self.emb_Phi(self.batch_h)
        h_Thet = self.emb_Thet(self.batch_h)
        h_Rou = torch.sigmoid(self.emb_Rou(self.batch_h)) + torch.relu(self.emb_Rou(self.batch_h))

        t_Phi = self.emb_Phi(self.batch_t)
        t_Thet = self.emb_Thet(self.batch_t)
        t_Rou = torch.sigmoid(self.emb_Rou(self.batch_t)) + torch.relu(self.emb_Rou(self.batch_t))

        r_Phi = self.emb_Phi(self.batch_r)
        r_Thet = self.emb_Thet(self.batch_r)
        r_Rou = torch.sigmoid(self.rel_Rou(self.batch_r)) + torch.relu(self.rel_Rou(self.batch_r))

        return h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou

    def forward(self):
        h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou = self.prepare_enti_rela()

        score = self._calc(h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou)

        regul = (torch.mean(torch.abs(h_Rou) ** 2)
                 + torch.mean(torch.abs(t_Rou) ** 2)
                 + torch.mean(torch.abs(t_Rou) ** 2))

        regul2 = (torch.mean(torch.relu(torch.abs(h_Thet)-2*torch.pi) ** 2) # <2pi
                 + torch.mean(torch.relu(-1 * h_Thet) ** 2) # >0
                 + torch.mean(torch.relu(torch.abs(t_Thet)-2*torch.pi) ** 2)
                 + torch.mean(torch.relu(-1 * t_Thet) ** 2)
                 + torch.mean(torch.relu(torch.abs(r_Thet)-2*torch.pi) ** 2)
                 + torch.mean(torch.relu(-1 * r_Thet) ** 2))

        regul3 = (torch.mean(torch.relu(torch.abs(h_Phi)-torch.pi) ** 2)
                 + torch.mean(torch.relu(-1*h_Phi) ** 2)
                 + torch.mean(torch.relu(torch.abs(t_Phi)-torch.pi) ** 2)
                 + torch.mean(torch.relu(-1*t_Phi) ** 2)
                 + torch.mean(torch.relu(torch.abs(r_Phi)-torch.pi) ** 2))

        return self.loss(score, regul, regul2, regul3)

    def predict(self):
        h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou = self.prepare_enti_rela()

        score = self._calc(h_Phi, h_Thet, h_Rou, t_Phi, t_Thet, t_Rou, r_Phi, r_Thet, r_Rou )
        return score.cpu().data.numpy()

