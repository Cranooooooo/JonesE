import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class JonesE_Inter_Dual_asym(Model):
    def __init__(self, config):
        super(JonesE_Inter_Dual_asym, self).__init__(config)
        self.emb_E_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_E_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_Eta = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.criterion = nn.Softplus()
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.score_trans = torch.nn.Linear(1,1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb_E_x.weight.data)
        nn.init.xavier_uniform_(self.emb_E_y.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_x.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_y.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet.weight.data)
        nn.init.xavier_uniform_(self.score_trans.weight.data)

    # Build Optical Interference calculation
    def _interfer(self, h_wf_list, t_wf_list, Eta, Thet):
        # 8 Interference Terms
        interfer = torch.sin(Thet)**2*h_wf_list[0]*t_wf_list[0]*torch.cos(t_wf_list[2]-h_wf_list[2]-Eta)
        interfer += torch.sin(Thet)**2*h_wf_list[1]*t_wf_list[1]*torch.cos(t_wf_list[3]-h_wf_list[3]+Eta)
        interfer += torch.cos(Thet)**2*h_wf_list[0]*t_wf_list[0]*torch.cos(t_wf_list[2]-h_wf_list[2]+Eta)
        interfer += torch.cos(Thet)**2*h_wf_list[1]*t_wf_list[1]*torch.cos(t_wf_list[3]-h_wf_list[3]-Eta)

        interfer += torch.sin(Thet)*torch.cos(Thet)*h_wf_list[1]*t_wf_list[0]*torch.cos(t_wf_list[1]-h_wf_list[2]+Eta)
        interfer += -torch.sin(Thet)*torch.cos(Thet)*h_wf_list[1]*t_wf_list[0]*torch.cos(t_wf_list[1]-h_wf_list[2]-Eta)
        interfer += torch.sin(Thet)*torch.cos(Thet)*h_wf_list[0]*t_wf_list[1]*torch.cos(t_wf_list[3]-h_wf_list[2]+Eta)
        interfer += -torch.sin(Thet)*torch.cos(Thet)*h_wf_list[0]*t_wf_list[1]*torch.cos(t_wf_list[3]-h_wf_list[2]-Eta)

        return interfer

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet):
        # Amp Normalization
        E_x_h_ = E_x_h/(E_x_h**2+E_y_h**2)**0.5
        E_y_h_ = E_y_h/(E_x_h**2+E_y_h**2)**0.5
        E_x_t_ = E_x_t/(E_x_t**2+E_y_t**2)**0.5
        E_y_t_ = E_y_t/(E_x_t**2+E_y_t**2)**0.5

        # Optical Interference calculation
        interfer = self._interfer([E_x_h_, E_y_h_, Phi_x_h, Phi_y_h],
                                  [E_x_t_, E_y_t_, Phi_x_t, Phi_y_t], Eta, Thet) # Bigger Score --> More similar

        return self.score_trans(torch.sum(interfer, -1).unsqueeze(-1)).squeeze(-1)
        # return -torch.sum(score_r, -1)


    def loss(self, score, regul, regul2):
        sap_node = int(self.batch_y.shape[0] * np.random.rand())
        # print(score[sap_node], self.batch_y[sap_node])
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul + self.config.lmbda * regul2)

    def prepare_enti_rela(self):
        E_x_h = torch.sigmoid(self.emb_E_x(self.batch_h))+torch.relu(self.emb_E_x(self.batch_h))
        E_y_h = torch.sigmoid(self.emb_E_y(self.batch_h))+torch.relu(self.emb_E_y(self.batch_h))
        # Phi_x_h = torch.tanh(self.emb_Phi_x(self.batch_h))*torch.pi
        # Phi_y_h = torch.tanh(self.emb_Phi_y(self.batch_h))*torch.pi
        Phi_x_h = self.emb_Phi_x(self.batch_h)
        Phi_y_h = self.emb_Phi_y(self.batch_h)

        E_x_t = torch.sigmoid(self.emb_E_x(self.batch_t))+torch.relu(self.emb_E_x(self.batch_t))
        E_y_t = torch.sigmoid(self.emb_E_y(self.batch_t))+torch.relu(self.emb_E_y(self.batch_t))
        # Phi_x_t = torch.tanh(self.emb_Phi_x(self.batch_t))*torch.pi
        # Phi_y_t = torch.tanh(self.emb_Phi_y(self.batch_t))*torch.pi
        Phi_x_t = self.emb_Phi_x(self.batch_t)
        Phi_y_t = self.emb_Phi_y(self.batch_t)

        Eta = self.rel_Eta(self.batch_r)
        Thet = self.rel_Thet(self.batch_r)

        return E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet

    def forward(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet)

        regul = (torch.mean(E_x_h ** 2)
                 + torch.mean(E_y_h ** 2)
                 + torch.mean(E_x_t ** 2)
                 + torch.mean(E_y_t ** 2))
                 # + torch.mean(Phi_x_h ** 2)
                 # + torch.mean(Phi_y_h ** 2)
                 # + torch.mean(Phi_x_t ** 2)
                 # + torch.mean(Phi_y_t ** 2))
        #
        #
        # regul2 = (torch.mean(torch.relu(torch.abs(Eta_1)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Thet_1)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Eta_2)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Thet_2)-torch.pi) ** 2)
        #           )

        return self.loss(score, regul=regul, regul2=0)

    def predict(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, Eta, Thet)
        return score.cpu().data.numpy()
