import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class JonesE_Inter_l1(Model):
    def __init__(self, config):
        super(JonesE_Inter_l1, self).__init__(config)
        self.emb_E_x_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_E_y_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_x_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_y_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)

        self.emb_E_x_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_E_y_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_x_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_Phi_y_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)


        self.rel_Eta_1_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_1_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Eta_2_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_2_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.rel_Eta_1_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_1_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Eta_2_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_Thet_2_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)


        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.emb_E_x_1.weight.data)
        nn.init.xavier_uniform_(self.emb_E_y_1.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_x_1.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_y_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_1_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_1_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_2_1.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_2_1.weight.data)

        nn.init.xavier_uniform_(self.emb_E_x_2.weight.data)
        nn.init.xavier_uniform_(self.emb_E_y_2.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_x_2.weight.data)
        nn.init.xavier_uniform_(self.emb_Phi_y_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_1_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_1_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Eta_2_2.weight.data)
        nn.init.xavier_uniform_(self.rel_Thet_2_2.weight.data)

    # Build Optical Interference calculation
    def _interfer(self, wf_list, tail_wf, print_switch):
        self.use_interfer = False
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
                head_amp_norm = F.normalize(torch.cat([wf[0], wf[1]], dim=1), p=2, dim=1)
                tail_amp_norm = F.normalize(torch.cat([tail_wf[0], tail_wf[1]], dim=1), p=2, dim=1)
                amp_in_prod = torch.sum(head_amp_norm*tail_amp_norm, dim=1)

                head_phase_norm = F.normalize(torch.cat([wf[2], wf[3]], dim=1), p=2, dim=1)
                tail_phase_norm = F.normalize(torch.cat([tail_wf[2], tail_wf[3]], dim=1), p=2, dim=1)
                phase_in_prod = torch.sum(head_phase_norm*tail_phase_norm, dim=1)

                total_interfer_intense += (amp_in_prod + phase_in_prod) / 2
        return total_interfer_intense / len(wf_list)

    # Build Jone Matrix trans process
    def _jonesMat_trans(self, in_wf_list, J_mat):
        # A wave fron t will split into 4 wave fronts
        out_wf_list = []

        # Calculate each wave front
        if J_mat[-1] == 'Eta':
            for wf in in_wf_list:
                out_wave_1 = [wf[0], wf[1], wf[2] - J_mat[0], wf[3] + J_mat[0]]  # Phase
                out_wf_list.append(out_wave_1)

        else:
            cos_Thet = torch.cos(J_mat[0])
            sin_Thet = torch.sin(J_mat[0])
            for wf in in_wf_list:
                out_wave_1 = [wf[0] * (cos_Thet ** 2), wf[1] * (cos_Thet ** 2), wf[2], wf[3]]  # Phase
                out_wf_list.append(out_wave_1)

                out_wave_2 = [wf[0] * (sin_Thet ** 2), wf[1] * (sin_Thet ** 2), wf[2], wf[3]]  # Phase
                out_wf_list.append(out_wave_2)

                out_wave_3 = [wf[1] * (sin_Thet * cos_Thet), wf[0] * (sin_Thet * cos_Thet), wf[2], wf[3]]  # Phase
                out_wf_list.append(out_wave_3)

                out_wave_4 = [wf[1] * (sin_Thet * cos_Thet), wf[0] * (sin_Thet * cos_Thet),  wf[3] + torch.pi, wf[2] + torch.pi]  # Phase
                out_wf_list.append(out_wave_4)

        return out_wf_list

    # Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, h_wf, t_wf, JMat_list, print_switch=False):

        # Amp Normalization
        E_x_h, E_y_h, Phi_x_h, Phi_y_h = h_wf
        E_x_t, E_y_t, Phi_x_t, Phi_y_t = t_wf
        Eta_1, Thet_1, Eta_2, Thet_2 = JMat_list

        # x_amp_ratio = E_x_h / E_x_t;
        # x_amp_score = 1. + torch.abs(torch.log(x_amp_ratio))
        # y_amp_ratio = E_y_h / E_y_t;
        # y_amp_score = 1. + torch.abs(torch.log(y_amp_ratio))
        E_x_h = E_x_h / (E_x_h ** 2 + E_y_h ** 2) ** 0.5
        E_y_h = E_y_h / (E_x_h ** 2 + E_y_h ** 2) ** 0.5
        E_x_t = E_x_t / (E_x_t ** 2 + E_y_t ** 2) ** 0.5
        E_y_t = E_y_t / (E_x_t ** 2 + E_y_t ** 2) ** 0.5

        # Head/Tail wave front
        h_wf_list = [[E_x_h, E_y_h, Phi_x_h, Phi_y_h]]
        t_wf = [E_x_t, E_y_t, Phi_x_t, Phi_y_t]
        J_mat_1 = [Eta_1, 'Eta']
        J_mat_2 = [Thet_1, 'Thet']
        J_mat_3 = [Eta_2, 'Eta']
        J_mat_4 = [Thet_2, 'Thet']

        # Wave Split after Jones Mat
        h_wf_out_list = self._jonesMat_trans(h_wf_list, J_mat_1)
        h_wf_out_list = self._jonesMat_trans(h_wf_out_list, J_mat_2)
        h_wf_out_list = self._jonesMat_trans(h_wf_out_list, J_mat_3)
        h_wf_out_list = self._jonesMat_trans(h_wf_out_list, J_mat_4)

        # Optical Interference calculation
        score_r = self._interfer(h_wf_out_list, t_wf, print_switch)
        # score_r = score_r / (x_amp_score * y_amp_score)
        score_r = score_r - 0.5  # Since Max score is 1, Min score is 0

        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        # sap_node = int(self.batch_y.shape[0] * np.random.rand())
        # print(score[sap_node], self.batch_y[sap_node])
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        # print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul + self.config.lmbda * regul2)

    def prepare_enti_rela(self):
        # Head Entity, Wave-1 and Wave-2
        E_x_h_1 = torch.sigmoid(self.emb_E_x_1(self.batch_h)) + torch.relu(self.emb_E_x_1(self.batch_h))
        E_y_h_1 = torch.sigmoid(self.emb_E_y_1(self.batch_h)) + torch.relu(self.emb_E_y_1(self.batch_h))
        Phi_x_h_1 = self.emb_Phi_x_1(self.batch_h)
        Phi_y_h_1 = self.emb_Phi_y_1(self.batch_h)

        E_x_h_2 = torch.sigmoid(self.emb_E_x_2(self.batch_h)) + torch.relu(self.emb_E_x_2(self.batch_h))
        E_y_h_2 = torch.sigmoid(self.emb_E_y_2(self.batch_h)) + torch.relu(self.emb_E_y_2(self.batch_h))
        Phi_x_h_2 = self.emb_Phi_x_2(self.batch_h)
        Phi_y_h_2 = self.emb_Phi_y_2(self.batch_h)


        # Tail Entity, Wave-1 and Wave-2
        E_x_t_1 = torch.sigmoid(self.emb_E_x_1(self.batch_t)) + torch.relu(self.emb_E_x_1(self.batch_t))
        E_y_t_1 = torch.sigmoid(self.emb_E_y_1(self.batch_t)) + torch.relu(self.emb_E_y_1(self.batch_t))
        Phi_x_t_1 = self.emb_Phi_x_1(self.batch_t)
        Phi_y_t_1 = self.emb_Phi_y_1(self.batch_t)

        E_x_t_2 = torch.sigmoid(self.emb_E_x_2(self.batch_t)) + torch.relu(self.emb_E_x_2(self.batch_t))
        E_y_t_2 = torch.sigmoid(self.emb_E_y_2(self.batch_t)) + torch.relu(self.emb_E_y_2(self.batch_t))
        Phi_x_t_2 = self.emb_Phi_x_2(self.batch_t)
        Phi_y_t_2 = self.emb_Phi_y_2(self.batch_t)

        # Relation Matrix for Wave-1 and Wave-2
        Eta_1_1 = self.rel_Eta_1_1(self.batch_r)
        Thet_1_1 = self.rel_Thet_1_1(self.batch_r)
        Eta_2_1 = self.rel_Eta_2_1(self.batch_r)
        Thet_2_1 = self.rel_Thet_2_1(self.batch_r)

        Eta_1_2 = self.rel_Eta_1_2(self.batch_r)
        Thet_1_2 = self.rel_Thet_1_2(self.batch_r)
        Eta_2_2 = self.rel_Eta_2_2(self.batch_r)
        Thet_2_2 = self.rel_Thet_2_2(self.batch_r)

        return [E_x_h_1, E_y_h_1, Phi_x_h_1, Phi_y_h_1], \
               [E_x_h_2, E_y_h_2, Phi_x_h_2, Phi_y_h_2], \
               [E_x_t_1, E_y_t_1, Phi_x_t_1, Phi_y_t_1], \
               [E_x_t_2, E_y_t_2, Phi_x_t_2, Phi_y_t_2], \
               [Eta_1_1, Thet_1_1, Eta_2_1, Thet_2_1], \
               [Eta_1_2, Thet_1_2, Eta_2_2, Thet_2_2]


    def forward(self):
        h_wf_1, h_wf_2, t_wf_1, t_wf_2, JMat_list_1, JMat_list_2 = self.prepare_enti_rela()

        # Calculate Interference score
        score = self._calc(h_wf_1, t_wf_1, JMat_list_1)
        score += self._calc(h_wf_2, t_wf_2, JMat_list_2)

        # Calculate Amp similarity


        # regul = (torch.mean(torch.abs(E_x_h) ** 2)
        #          + torch.mean(torch.abs(E_y_h) ** 2)
        #          + torch.mean(torch.abs(E_x_t) ** 2)
        #          + torch.mean(torch.abs(E_y_t) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Phi_x_h)-0.5*torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Phi_y_h)-0.5*torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Phi_x_t)-0.5*torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Phi_y_t)-0.5*torch.pi) ** 2))

        # regul2 = (torch.mean(torch.relu(torch.abs(Eta_1)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Thet_1)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Eta_2)-torch.pi) ** 2)
        #          + torch.mean(torch.relu(torch.abs(Thet_2)-torch.pi) ** 2)
        #           )

        return self.loss(score, regul=0, regul2=0.)

    def predict(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
        Eta_1, Thet_1, Eta_2, Thet_2 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
                           Eta_1, Thet_1, Eta_2, Thet_2, print_switch=False)
        return score.cpu().data.numpy()
