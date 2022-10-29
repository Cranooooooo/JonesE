import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState



class BoxE(Model):
    def __init__(self, config):
        super(BoxE, self).__init__(config)
        self.emb_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_w = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_h = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.rel_x_shift = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_y_shift = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_w_zoom = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_h_zoom = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.score_weight = torch.nn.Embedding(self.config.relTotal, 3)

        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)

        # self.score_trans = torch.nn.Linear(3, 1)
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb_x.weight.data)
        nn.init.xavier_uniform_(self.emb_w.weight.data)
        nn.init.xavier_uniform_(self.emb_y.weight.data)
        nn.init.xavier_uniform_(self.emb_h.weight.data)

        nn.init.xavier_uniform_(self.rel_x_shift.weight.data)
        nn.init.xavier_uniform_(self.rel_y_shift.weight.data)
        nn.init.xavier_uniform_(self.rel_w_zoom.weight.data)
        nn.init.xavier_uniform_(self.rel_h_zoom.weight.data)

    # Build BOX IOU
    def _IOU(self, h_x, h_y, h_w, h_h, t_x, t_y, t_w, t_h):

        n = nn.ReLU()
        endx = torch.max(h_x + h_w, t_x + t_w)
        startx = torch.min(h_x, t_x)
        width = n(h_w + t_w - (endx - startx))

        endy = torch.max(h_y + h_h, t_y + t_h)
        starty = torch.min(h_y, t_y)
        height = n(h_h + t_h - (endy - starty))

        Area = width * height
        Area_h = h_w * h_h
        Area_t = t_w * t_h

        ratio_1 = Area / (Area_h+Area_t-Area)
        ratio_2 = Area / Area_h
        ratio_3 = Area / Area_t

        return ratio_1.unsqueeze(dim=-1), ratio_2.unsqueeze(dim=-1), ratio_3.unsqueeze(dim=-1)

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight):

        # Head Transfer with relation
        Eh_x = Eh_x + Rel_x_shift
        Eh_y = Eh_y + Rel_y_shift
        Eh_w = Eh_w * Rel_w_zoom
        Eh_h = Eh_h * Rel_h_zoom

        # Score
        ratio_1, ratio_2, ratio_3 = self._IOU(Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h)
        score_r = torch.sum(torch.cat((ratio_1, ratio_2, ratio_3), dim=-1)*score_weight, dim=-1)

        return torch.sum(score_r, -1)


    def loss(self, score, regul1, regul2):
        sap_node = int(self.batch_y.shape[0] * np.random.rand())
        print(score[sap_node], self.batch_y[sap_node])
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul1 + self.config.lmbda * regul2)

    def prepare_enti_rela(self):

        '''
        self.emb_x1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_x2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_y1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_y2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.rel_x_shift = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_y_shift = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_x_zoom = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_y_zoom = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        :return:
        '''

        # m = nn.Softplus()
        Eh_x = torch.relu(self.emb_x(self.batch_h))
        Eh_y = torch.relu(self.emb_y(self.batch_h))
        Eh_w = torch.exp(self.emb_w(self.batch_h))
        Eh_h = torch.exp(self.emb_h(self.batch_h))

        Et_x = torch.relu(self.emb_x(self.batch_t))
        Et_y = torch.relu(self.emb_y(self.batch_t))
        Et_w = torch.exp(self.emb_w(self.batch_t))
        Et_h = torch.exp(self.emb_h(self.batch_t))

        Rel_x_shift = self.rel_x_shift(self.batch_r)
        Rel_y_shift = self.rel_y_shift(self.batch_r)
        Rel_w_zoom = torch.exp(self.rel_w_zoom(self.batch_r))
        Rel_h_zoom = torch.exp(self.rel_h_zoom(self.batch_r))

        score_weight = self.score_weight(self.batch_r).unsqueeze(dim=1)

        return Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight

    def forward(self):
        Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight = self.prepare_enti_rela()
        score = self._calc(Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight)

        regul1 = torch.mean(Eh_x**2 + Eh_y**2 + Eh_w**2 + Eh_h**2 + Et_x**2 + Et_y**2 + Et_w**2 + Et_h**2)
        regul2 = torch.mean(Rel_x_shift**2 + Rel_y_shift**2 + Rel_w_zoom**2 + Rel_h_zoom**2)

        return self.loss(score, regul1=regul1, regul2=regul2)

    def predict(self):
        Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight = self.prepare_enti_rela()
        score = self._calc(Eh_x, Eh_y, Eh_w, Eh_h, Et_x, Et_y, Et_w, Et_h, Rel_x_shift, Rel_y_shift, Rel_w_zoom, Rel_h_zoom, score_weight)
        return score.cpu().data.numpy()
