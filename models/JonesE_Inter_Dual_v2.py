import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class JonesE_Inter_Dual(Model):
    def __init__(self, config):
        super(JonesE_Inter_Dual, self).__init__(config)
        self.emb_E_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_E_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.rel_Eta_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Eta_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.rel_Eta_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Eta_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
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
        nn.init.xavier_uniform_(self.rel_Eta_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_3.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_3.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_4.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_4.weight.data)
        nn.init.xavier_uniform_(self.score_trans.weight.data)

    # Build Optical Interference calculation
    def _interfer(self, h_wf_list, t_wf_list):
        self.use_interfer = False
        total_interfer_intense = 0.
        for h_wf_idx in range(len(h_wf_list)):
            h_wf = h_wf_list[h_wf_idx]
            for t_wf_idx in range(len(t_wf_list)):
                t_wf = t_wf_list[t_wf_idx]
                if self.use_interfer:  # Use Interference
                    x_interf = h_wf[0] ** 2 + t_wf[0] ** 2 + 2 * h_wf[0] * t_wf[0] * torch.cos(h_wf[2] - t_wf[2])
                    y_interf = h_wf[1] ** 2 + t_wf[1] ** 2 + 2 * h_wf[1] * t_wf[1] * torch.cos(h_wf[3] - t_wf[3])
                    total_interfer_intense += x_interf
                    total_interfer_intense += y_interf

                else:  # Use Inner product
                    head_x_amp_norm = h_wf[0].unsqueeze(-1)
                    tail_x_amp_norm = t_wf[0].unsqueeze(-1)
                    x_amp_in_prod = torch.sum((head_x_amp_norm - tail_x_amp_norm) ** 2, dim=2).squeeze(-1)*h_wf[0]*t_wf[0]

                    head_y_amp_norm = h_wf[1].unsqueeze(-1)
                    tail_y_amp_norm = t_wf[1].unsqueeze(-1)
                    y_amp_in_prod = torch.sum((head_y_amp_norm - tail_y_amp_norm) ** 2, dim=2).squeeze(-1)*h_wf[1]*t_wf[1]

                    head_x_phase_norm = h_wf[2].unsqueeze(-1)
                    tail_x_phase_norm = t_wf[2].unsqueeze(-1)
                    x_phase_in_prod = torch.sum((head_x_phase_norm - tail_x_phase_norm) ** 2, dim=2).squeeze(-1)*h_wf[0]*t_wf[0]

                    head_y_phase_norm = h_wf[3].unsqueeze(-1)
                    tail_y_phase_norm = t_wf[3].unsqueeze(-1)
                    y_phase_in_prod = torch.sum((head_y_phase_norm - tail_y_phase_norm) ** 2, dim=2).squeeze(-1)*h_wf[1]*t_wf[1]

                    total_interfer_intense += (x_amp_in_prod + y_amp_in_prod+
                                               x_phase_in_prod + y_phase_in_prod) #1./(1.+amp_in_prod*+phase_in_prod)
        return total_interfer_intense# / len(h_wf_list)

    # Build Jone Matrix trans process
    def _jonesMat_trans(self, in_wf_list, J_mat):
        # A wave fron t will split into 4 wave fronts
        out_wf_list = []

        # Calculate each wave front
        cos_Thet = torch.cos(J_mat[0]) # torch.sigmoid(J_mat[0]) #
        sin_Thet = torch.sin(J_mat[0]) # (1.-cos_Thet**2)**0.5 #

        for wf in in_wf_list:
            out_wave_1 = [wf[0] * (cos_Thet ** 2), wf[1] * (cos_Thet ** 2), # Ampli
                          wf[2]-J_mat[1], wf[3]+J_mat[1]] # Phase
            out_wf_list.append(out_wave_1)

            out_wave_2 = [wf[0] * (sin_Thet ** 2), wf[1] * (sin_Thet ** 2), # Ampli
                          wf[2]+J_mat[1], wf[3]-J_mat[1]] # Phase
            out_wf_list.append(out_wave_2)

            out_wave_3 = [wf[1] * (sin_Thet*cos_Thet), wf[0] * (sin_Thet*cos_Thet), # Ampli
                          wf[3]-J_mat[1], wf[2]-J_mat[1]] # Phase
            out_wf_list.append(out_wave_3)

            out_wave_4 = [wf[1] * (sin_Thet*cos_Thet), wf[0] * (sin_Thet*cos_Thet), # Ampli
                          wf[3]+J_mat[1]+torch.pi, wf[2]+J_mat[1]+torch.pi] # Phase
            out_wf_list.append(out_wave_4)

        # Wave front number assertion
        assert len(out_wf_list) == 4*len(in_wf_list)
        return out_wf_list


    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,
              Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4):

        # Amp Normalization
        E_x_h_ = E_x_h/(E_x_h**2+E_y_h**2)**0.5
        E_y_h_ = E_y_h/(E_x_h**2+E_y_h**2)**0.5
        E_x_t_ = E_x_t/(E_x_t**2+E_y_t**2)**0.5
        E_y_t_ = E_y_t/(E_x_t**2+E_y_t**2)**0.5

        # Head/Tail wave front
        h_wf_list = [[E_x_h_, E_y_h_, Phi_x_h, Phi_y_h]]
        t_wf_list = [[E_x_t_, E_y_t_, Phi_x_t, Phi_y_t]]
        J_mat_1 = [Thet_1, Eta_1]
        # J_mat_2 = [Thet_2, Eta_2]

        J_mat_3 = [Thet_3, Eta_3]
        # J_mat_4 = [Thet_4, Eta_4]

        # Wave Split after Jones Mat
        h_wf_out_list = self._jonesMat_trans(h_wf_list, J_mat_1)
        # h_wf_out_list = self._jonesMat_trans(h_wf_out_list, J_mat_2)

        t_wf_out_list = self._jonesMat_trans(t_wf_list, J_mat_3)
        # t_wf_out_list = self._jonesMat_trans(t_wf_out_list, J_mat_4)

        # Optical Interference calculation
        score_r = self._interfer(h_wf_out_list, t_wf_out_list) # Bigger Score --> More similar

        # return self.score_trans(-torch.sum(score_r, -1).unsqueeze(-1)).squeeze(-1)
        return -torch.sum(score_r, -1)


    def loss(self, score, regul, regul2):
        sap_node = int(self.batch_y.shape[0] * np.random.rand())
        print(score[sap_node], self.batch_y[sap_node])
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

        # Eta_1 = torch.tanh(self.rel_Eta_1(self.batch_r))*torch.pi
        # Thet_1 = torch.tanh(self.rel_Thet_1(self.batch_r))*torch.pi
        # Eta_2 = torch.tanh(self.rel_Eta_1(self.batch_r))*torch.pi
        # Thet_2 = torch.tanh(self.rel_Thet_1(self.batch_r))*torch.pi
        Eta_1 = self.rel_Eta_1(self.batch_r)
        Thet_1 = self.rel_Thet_1(self.batch_r)
        Eta_2 = self.rel_Eta_2(self.batch_r)
        Thet_2 = self.rel_Thet_2(self.batch_r)

        Eta_3 = self.rel_Eta_3(self.batch_r)
        Thet_3 = self.rel_Thet_3(self.batch_r)
        Eta_4 = self.rel_Eta_4(self.batch_r)
        Thet_4 = self.rel_Thet_4(self.batch_r)

        return E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
               Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4

    def forward(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4)

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
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3, Eta_4, Thet_4)
        return score.cpu().data.numpy()
