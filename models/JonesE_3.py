import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState


class JonesE_3(Model):
    def __init__(self, config):
        super(JonesE_3, self).__init__(config)
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

            # Jones Matrix 3
            Eta_3, Thet_3 = self.quaternion_init(self.config.entTotal, self.config.hidden_size, type='relation')
            Eta_3, Thet_3 = torch.from_numpy(Eta_3), torch.from_numpy(Thet_3)
            self.rel_Eta_3.weight.data = Eta_3.type_as(self.rel_Eta_3.weight.data)
            self.rel_Thet_3.weight.data = Thet_3.type_as(self.rel_Thet_3.weight.data)

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
            nn.init.xavier_uniform_(self.rel_Eta_3.weight.data)
            nn.init.xavier_uniform_(self.rel_Thet_3.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)

    # Build Jone Vector
    def _jonesVec(self, E_x, E_y, Phi_x, Phi_y):
        return (E_x*torch.cos(Phi_x), E_y*torch.cos(Phi_y))

    # Build Jone Matrix
    def _jonesMat(self, Eta, Thet):
        cos_Thet = torch.cos(Thet)
        sin_Thet = torch.sin(Thet)
        cos_Eta = torch.cos(Eta)

        jonesMat11 = cos_Thet**2 + cos_Eta*sin_Thet**2
        jonesMat12 = (1. - cos_Eta) * cos_Thet * sin_Thet
        jonesMat21 = (1. - cos_Eta) * cos_Thet * sin_Thet
        jonesMat22 = sin_Thet**2 + cos_Eta*cos_Thet**2

        return (jonesMat11, jonesMat12, jonesMat21, jonesMat22)

    #Calculate the inner product of the head entity and the relationship Hamiltonian product and the tail entity
    def _calc(self, E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,
              Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3):
        # Build Jones Vector
        jonesVec_h = self._jonesVec(E_x_h, E_y_h, Phi_x_h, Phi_y_h)
        jonesVec_t = self._jonesVec(E_x_t, E_y_t, Phi_x_t, Phi_y_t)

        # Build Jones Matrix
        jonesMat1 = self._jonesMat(Eta_1, Thet_1)
        jonesMat2 = self._jonesMat(Eta_2, Thet_2)
        jonesMat3 = self._jonesMat(Eta_3, Thet_3)

        # Jones Matrix Propagation
        jonesVec_ho = [jonesMat1[0] * jonesVec_h[0] + jonesMat1[1] * jonesVec_h[1],
                       jonesMat1[2] * jonesVec_h[0] + jonesMat1[3] * jonesVec_h[1]]
        jonesVec_ho = [jonesMat2[0] * jonesVec_ho[0] + jonesMat2[1] * jonesVec_ho[1],
                       jonesMat2[2] * jonesVec_ho[0] + jonesMat2[3] * jonesVec_ho[1]]
        jonesVec_ho = [jonesMat3[0] * jonesVec_ho[0] + jonesMat3[1] * jonesVec_ho[1],
                       jonesMat3[2] * jonesVec_ho[0] + jonesMat3[3] * jonesVec_ho[1]]

        # Orthogonal Polarization Calculation
        score_r = jonesVec_ho[0]*jonesVec_t[0]/(E_x_h*E_x_t)+\
                  jonesVec_ho[1]*jonesVec_t[1]/(E_y_h*E_y_t)

        # print('Max/Min Score is : {}'.format(round(float(torch.max(score_r)), 4)),
        #                                      round(float(torch.min(score_r)), 4),
        #       end=' --- ')

        # print('Mean Score is : {}'.format(round(float(torch.mean(score_r)), 4)), end=' --- ')
        return -torch.sum(score_r, -1)

    

    def loss(self, score, regul, regul2):
        score_loss = torch.mean(self.criterion(score * self.batch_y))
        print('Score loss is {}'.format(round(float(score_loss), 4)))
        return (score_loss + self.config.lmbda * regul + self.config.lmbda * regul2)

    def prepare_enti_rela(self):
        E_x_h = torch.sigmoid(self.emb_E_x(self.batch_h))+torch.relu(self.emb_E_x(self.batch_h))
        E_y_h = torch.sigmoid(self.emb_E_y(self.batch_h))+torch.relu(self.emb_E_y(self.batch_h))
        Phi_x_h = torch.tanh(self.emb_Phi_x(self.batch_h))*torch.pi
        Phi_y_h = torch.tanh(self.emb_Phi_y(self.batch_h))*torch.pi
        # Phi_x_h = self.emb_Phi_x(self.batch_h)
        # Phi_y_h = self.emb_Phi_y(self.batch_h)

        E_x_t = torch.sigmoid(self.emb_E_x(self.batch_t))+torch.relu(self.emb_E_x(self.batch_t))
        E_y_t = torch.sigmoid(self.emb_E_y(self.batch_t))+torch.relu(self.emb_E_y(self.batch_t))
        Phi_x_t = torch.tanh(self.emb_Phi_x(self.batch_t))*torch.pi
        Phi_y_t = torch.tanh(self.emb_Phi_y(self.batch_t))*torch.pi
        # Phi_x_t = self.emb_Phi_x(self.batch_t)
        # Phi_y_t = self.emb_Phi_y(self.batch_t)

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

        return E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t, \
               Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3

    def forward(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3)

        regul = (torch.mean(torch.abs(E_x_h) ** 2)
                 + torch.mean(torch.abs(E_y_h) ** 2)
                 + torch.mean(torch.abs(E_x_t) ** 2)
                 + torch.mean(torch.abs(E_y_t) ** 2))
                 # + torch.mean(torch.relu(torch.abs(Phi_x_h)-torch.pi) ** 2)
                 # + torch.mean(torch.relu(torch.abs(Phi_y_h)-torch.pi) ** 2)
                 # + torch.mean(torch.relu(torch.abs(Phi_x_t)-torch.pi) ** 2)
                 # + torch.mean(torch.relu(torch.abs(Phi_y_t)-torch.pi) ** 2)


        regul2 = (torch.mean(torch.relu(torch.abs(Eta_1)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Thet_1)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Eta_2)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Thet_2)-torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Eta_3) - torch.pi) ** 2)
                 + torch.mean(torch.relu(torch.abs(Thet_3) - torch.pi) ** 2)
                  )

        return self.loss(score, regul, regul2)

    def predict(self):
        E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
        Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3 = self.prepare_enti_rela()

        score = self._calc(E_x_h, E_y_h, Phi_x_h, Phi_y_h, E_x_t, E_y_t, Phi_x_t, Phi_y_t,\
                           Eta_1, Thet_1, Eta_2, Thet_2, Eta_3, Thet_3)
        return score.cpu().data.numpy()

    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(2020)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)


        # Calculate the three parts about t
        kernel_shape1 = (in_features, out_features)
        number_of_weights1 = np.prod(kernel_shape1)
        t_i = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_j = np.random.uniform(0.0, 1.0, number_of_weights1)
        t_k = np.random.uniform(0.0, 1.0, number_of_weights1)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights1):
            norm1 = np.sqrt(t_i[i] ** 2 + t_j[i] ** 2 + t_k[i] ** 2) + 0.0001
            t_i[i] /= norm1
            t_j[i] /= norm1
            t_k[i] /= norm1
        t_i = t_i.reshape(kernel_shape1)
        t_j = t_j.reshape(kernel_shape1)
        t_k = t_k.reshape(kernel_shape1)
        tmp_t = rng.uniform(low=-s, high=s, size=kernel_shape1)


        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        phase1 = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape1)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        wt_i = tmp_t * t_i * np.sin(phase1)
        wt_j = tmp_t * t_j * np.sin(phase1)
        wt_k = tmp_t * t_k * np.sin(phase1)

        i_0=weight_r
        i_1=weight_i
        i_2=weight_j
        i_3=weight_k
        i_4=(-wt_i*weight_i-wt_j*weight_j-wt_k*weight_k)/2
        i_5=(wt_i*weight_r+wt_j*weight_k-wt_k*weight_j)/2
        i_6=(-wt_i*weight_k+wt_j*weight_r+wt_k*weight_i)/2
        i_7=(wt_i*weight_j-wt_j*weight_i+wt_k*weight_r)/2


        return (i_0,i_1,i_2,i_3,i_4,i_5,i_6,i_7)
