import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class JonesE_Inter(Model):
    def __init__(self, config):
        super(JonesE_Inter, self).__init__(config)
        self.emb_E_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_E_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_x = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_y = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.rel_Eta_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Eta_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        if False:
            E_x, E_y, Phi_x, Phi_y = self.quaternion_init(self.config.entTotal, self.config.hidden_size, type='entity')
            E_x, E_y = torch.from_numpy(E_x), torch.from_numpy(E_x)
            Phi_x, Phi_y = torch.from_numpy(Phi_x), torch.from_numpy(Phi_y)

            # Embedding
            self.emb_E_x.weight.data = E_x.type_as(self.emb_E_x.weight.data)
            self.emb_E_y.weight.data = E_y.type_as(self.emb_E_y.weight.data)
            self.emb_Phi_x.weight.data = Phi_x.type_as(self.emb_Phi_x.weight.data)
            self.emb_Phi_y.weight.data = Phi_y.type_as(self.emb_Phi_y.weight.data)

            # Jones Matrix 1
            Eta_1, Thet_1 = self.quaternion_init(self.config.entTotal, self.config.hidden_size, type='relation')
            Eta_1, Thet_1 = torch.from_numpy(Eta_1), torch.from_numpy(Thet_1)
            self.rel_Eta_1.weight.data = Eta_1.type_as(self.rel_Eta_1.weight.data)
            self.rel_Thet_1.weight.data = Thet_1.type_as(self.rel_Thet_1.weight.data)

            # Jones Matrix 2
            Eta_2, Thet_2 = self.quaternion_init(self.config.entTotal, self.config.hidden_size, type='relation')
            Eta_2, Thet_2 = torch.from_numpy(Eta_2), torch.from_numpy(Thet_2)
            self.rel_Eta_2.weight.data = Eta_2.type_as(self.rel_Eta_2.weight.data)
            self.rel_Thet_2.weight.data = Thet_2.type_as(self.rel_Thet_2.weight.data)

            nn.init.xavier_uniform_(self.rel_w.weight.data)

        else:
            nn.init.xavier_uniform_(self.emb_E_x.weight.data)
            nn.init.xavier_uniform_(self.emb_E_y.weight.data)
            nn.init.xavier_uniform_(self.emb_Phi_x.weight.data)
            nn.init.xavier_uniform_(self.emb_Phi_y.weight.data)
            nn.init.xavier_uniform_(self.rel_Eta_1.weight.data)
            nn.init.xavier_uniform_(self.rel_Thet_1.weight.data)
            nn.init.xavier_uniform_(self.rel_Eta_2.weight.data)
            nn.init.xavier_uniform_(self.rel_Thet_2.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)



    # Build Optical Interference calculation
    def _interfer(self, wf_list, tail_wf, print_switch):
        self.use_interfer = True
        total_interfer_intense = 0.
        for wf in wf_list:
            # Whether Print the information
            if print_switch:
                sap_node = int(10 * np.random.rand())
                sap_dim = int(self.config.hidden_size * np.random.rand())
                print('X amp : {} / {}'.format(wf[0][sap_node][sap_dim], tail_wf[0][sap_node][sap_dim]))
                print('Y amp : {} / {}'.format(wf[1][sap_node][sap_dim], tail_wf[1][sap_node][sap_dim]))
                print('X Phase: {} / {}'.format(wf[2][sap_node][sap_dim], tail_wf[2][sap_node][sap_dim]))
                print('Y Phase: {} / {}'.format(wf[3][sap_node][sap_dim], tail_wf[3][sap_node][sap_dim]))

            if self.use_interfer:  # Use Interference
                x_interf = wf[0] ** 2 + tail_wf[0] ** 2 + 2 * wf[0] * tail_wf[0] * torch.cos(wf[2] - tail_wf[2])
                y_interf = wf[1] ** 2 + tail_wf[1] ** 2 + 2 * wf[1] * tail_wf[1] * torch.cos(wf[3] - tail_wf[3])
                total_interfer_intense += x_interf
                total_interfer_intense += y_interf

            else:  # Use Inner product
                head_amp_norm = F.normalize(torch.cat([wf[0].unsqueeze(-1), wf[1].unsqueeze(-1)], dim=2), p=2, dim=2)
                tail_amp_norm = F.normalize(torch.cat([tail_wf[0].unsqueeze(-1), tail_wf[1].unsqueeze(-1)], dim=2), p=2, dim=2)
                amp_in_prod = torch.sum(head_amp_norm* tail_amp_norm, dim=2)

                head_phase_norm = F.normalize(torch.cat([wf[2].unsqueeze(-1), wf[3].unsqueeze(-1)], dim=2), p=2, dim=2)
                tail_phase_norm = F.normalize(torch.cat([tail_wf[2].unsqueeze(-1), tail_wf[3].unsqueeze(-1)], dim=2), p=2, dim=2)
                phase_in_prod = torch.sum(head_phase_norm * tail_phase_norm, dim=2).squeeze(-1)

                total_interfer_intense += (amp_in_prod + phase_in_prod) / 2
        return total_interfer_intense / len(wf_list)

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
    def _calc(self, E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,
              Eta_1, Thet_1, Eta_2, Thet_2, print_switch=False):

        # Amp Normalization
        E_x_h = E_x_h/(E_x_h**2+E_y_h**2)**0.5
        E_y_h = E_y_h/(E_x_h**2+E_y_h**2)**0.5
        E_x_t = E_x_t/(E_x_t**2+E_y_t**2)**0.5
        E_y_t = E_y_t/(E_x_t**2+E_y_t**2)**0.5

        # Head/Tail wave front
        h_wf_list = [[E_x_h, E_y_h, Phi_x_h, Phi_y_h]]
        t_wf = [E_x_t, E_y_t, Phi_x_t, Phi_y_t]
        J_mat_1 = [Thet_1, Eta_1]
        J_mat_2 = [Thet_2, Eta_2]

        # Wave Split after Jones Mat
        h_wf_out_list = self._jonesMat_trans(h_wf_list, J_mat_1)
        h_wf_out_list = self._jonesMat_trans(h_wf_out_list, J_mat_2)

        # Optical Interference calculation
        score_r = self._interfer(h_wf_out_list, t_wf, print_switch)
        score_r = score_r-0.5 # Since Max score is 1, Min score is 0

        # print('Max/Min Score is : {}'.format(round(float(torch.max(score_r)), 4)),
        #                                      round(float(torch.min(score_r)), 4))

        return -torch.sum(score_r, -1)

    

    def loss(self, score, regul, regul2):
        # sap_node = int(self.batch_y.shape[0] * np.random.rand())
        # print(score[sap_node], self.batch_y[sap_node])
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        # print('Score loss is {}'.format(round(float(score_loss), 4)))
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

        return E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
               Eta_1, Thet_1, Eta_2, Thet_2

    def forward(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2)

        regul = (torch.mean(torch.abs(E_x_h) ** 2)
                 + torch.mean(torch.abs(E_y_h) ** 2)
                 + torch.mean(torch.abs(E_x_t) ** 2)
                 + torch.mean(torch.abs(E_y_t) ** 2)
                 + torch.mean(torch.relu(torch.abs(Phi_x_h)-0.5*torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Phi_y_h)-0.5*torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Phi_x_t)-0.5*torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Phi_y_t)-0.5*torch.pi) ** 2))


        regul2 = (torch.mean(torch.relu(torch.abs(Eta_1)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Thet_1)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Eta_2)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Thet_2)-torch.pi) ** 2)
                  )

        return self.loss(score, regul=regul, regul2=regul2)

    def predict(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2, print_switch=False)
        return score.cpu().data.numpy()
