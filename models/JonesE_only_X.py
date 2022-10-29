import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

'''
Each entity is represented as a unit amp wave front.
Thus, (E_x)**2+(E_y)**2 = 1,
Thus, we only need a ksi angle, [0, pi/2]
Ex = cos(ksi); Ey = sin(ksi)

-------
Instead of using Jones Matrix,
we need an amplifier, to adjust Ex and Ey.
we need a phase modifier, to adjust phi_x and phi_y.

-------
But, if these two are independent,
we only need Ex or EY.
Thus we still need Jones-Mat, which not only adjust phi_x and phi_y,
but also reschedule the energy allocation between X and Y-axis

'''


class JonesE_only_X(Model):
    def __init__(self, config):
        super(JonesE_only_X, self).__init__(config)
        self.emb_Ex = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Ey = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.emb_Phi_x = nn.Embedding(self.config.entTotal, self.config.hidden_size, max_norm=torch.pi*2)
        self.emb_Phi_y = nn.Embedding(self.config.entTotal, self.config.hidden_size, max_norm=torch.pi*2)

        self.rel_ampf_h = nn.Embedding(self.config.relTotal, 2)
        self.rel_Eta_h = nn.Embedding(self.config.relTotal, self.config.hidden_size, max_norm=torch.pi*2)
        self.rel_Thet_h = nn.Embedding(self.config.relTotal, self.config.hidden_size, max_norm=torch.pi*2)

        self.rel_ampf_t = nn.Embedding(self.config.relTotal, 2)
        self.rel_Eta_t = nn.Embedding(self.config.relTotal, self.config.hidden_size, max_norm=torch.pi*2)
        self.rel_Thet_t = nn.Embedding(self.config.relTotal, self.config.hidden_size, max_norm=torch.pi*2)

        self.criterion = nn.Softplus()
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)

        self.use_interfer = False
        if not self.use_interfer: self.score_trans = torch.nn.Linear(1, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb_Ex.weight.data)
        nn.init.xavier_uniform_(self.emb_Ey.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_x.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_y.weight.data)

        # nn.init.xavier_uniform_(self.rel_ampf_h.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_h.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_h.weight.data)

        # nn.init.xavier_uniform_(self.rel_ampf_t.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_t.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_t.weight.data)

        if not self.use_interfer: nn.init.xavier_uniform_(self.score_trans.weight.data)

    # Build Optical Interference calculation
    def _interfer(self, h_wf_list, t_wf_list):
        total_interfer_intense = 0.
        for wf_idx in range(len(h_wf_list)):
            h_wf = h_wf_list[wf_idx]
            t_wf = t_wf_list[wf_idx]
            if self.use_interfer:  # Use Interference
                x_interf = 2 * h_wf[0] * t_wf[0] * torch.cos(h_wf[2] - t_wf[2])
                y_interf = 2 * h_wf[1] * t_wf[1] * torch.cos(h_wf[3] - t_wf[3])
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

                # Using "-", because the more similar the higher the score, which is contrast to the calculation here
                total_interfer_intense -= (x_amp_in_prod + y_amp_in_prod + x_phase_in_prod + y_phase_in_prod)

        return total_interfer_intense# / len(h_wf_list)

    # Build Jone Matrix trans process
    def _jonesMat_trans(self, in_wf_list, J_mat):
        # A wave fron t will split into 4 wave fronts
        out_wf_list = []

        # Calculate each wave front
        cos_Thet = torch.cos(J_mat[0])
        sin_Thet = torch.sin(J_mat[0])

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
    def _calc(self, E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
              ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t):

        # Head/Tail wave front
        E_x_h = E_x_h * ampf_h[:, 0].unsqueeze(-1); E_y_h = E_y_h * ampf_h[:, 1].unsqueeze(-1)
        E_x_t = E_x_t * ampf_t[:, 0].unsqueeze(-1); E_y_t = E_y_t * ampf_t[:, 1].unsqueeze(-1)
        h_wf_list = [[E_x_h, E_y_h, Phi_x_h, Phi_y_h]]
        t_wf_list = [[E_x_t, E_y_t, Phi_x_t, Phi_y_t]]
        J_mat_h = [Thet_h, Eta_h]
        J_mat_t = [Thet_t, Eta_t]

        # Wave Split after Jones Mat
        h_wf_out_list = self._jonesMat_trans(h_wf_list, J_mat_h)
        t_wf_out_list = self._jonesMat_trans(t_wf_list, J_mat_t)

        # Optical Interference calculation
        score_r = self._interfer(h_wf_out_list, t_wf_out_list) # Bigger Score --> More similar
        if not self.use_interfer:
            return self.score_trans(torch.sum(score_r, -1).unsqueeze(-1)).squeeze(-1)
        return torch.sum(score_r, -1) #-torch.sum(score_r, -1)


    def loss(self, score, regul, regul2):
        sap_node = int(self.batch_y.shape[0] * np.random.rand())
        print(score[sap_node], self.batch_y[sap_node])
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul + self.config.lmbda * regul2)

    def prepare_enti_rela(self):
        E_x_h = torch.relu(self.emb_Ex(self.batch_h)) + torch.sigmoid(self.emb_Ex(self.batch_h))
        E_y_h = torch.relu(self.emb_Ey(self.batch_h)) + torch.sigmoid(self.emb_Ey(self.batch_h))
        Phi_x_h = torch.relu(self.emb_Phi_x(self.batch_h))-torch.pi
        Phi_y_h = torch.relu(self.emb_Phi_y(self.batch_h))-torch.pi

        E_x_t = torch.relu(self.emb_Ex(self.batch_t)) + torch.sigmoid(self.emb_Ex(self.batch_t))
        E_y_t = torch.relu(self.emb_Ey(self.batch_t)) + torch.sigmoid(self.emb_Ey(self.batch_t))
        Phi_x_t = torch.relu(self.emb_Phi_x(self.batch_t))-torch.pi
        Phi_y_t = torch.relu(self.emb_Phi_y(self.batch_t))-torch.pi

        m=nn.Softplus()
        ampf_h = m(self.rel_ampf_h(self.batch_r))
        Eta_h = torch.relu(self.rel_Eta_h(self.batch_r))-torch.pi
        Thet_h = torch.relu(self.rel_Thet_h(self.batch_r))-torch.pi

        ampf_t = m(self.rel_ampf_t(self.batch_r))
        Eta_t = torch.relu(self.rel_Eta_t(self.batch_r))-torch.pi
        Thet_t = torch.relu(self.rel_Thet_t(self.batch_r))-torch.pi

        return E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
               ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t

    def forward(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
        ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
                           ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t)

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
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
        ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
                           ampf_h, Eta_h, Thet_h, ampf_t, Eta_t, Thet_t)
        return score.cpu().data.numpy()
