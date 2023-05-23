import time

start_time = time.time()
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import sys
import math
from random import shuffle
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(threshold=torch.inf)
parser = argparse.ArgumentParser(description='Model Variables')
parser.add_argument('Vgs', type=float, help='Value of Vgs. Used during inference')
parser.add_argument('t_ox', type=float, help='Enter the tox value in nm. Used during inference')
parser.add_argument('N_A', type=float, help='Enter the N_A value as a coefficient to 1e24. Used during inference')
parser.add_argument('train_samples', type=int, help='Number of Training Samples. This is used during inference, when we need to sample for y. This number is the number of datapoints of y, uniformly sampled.')
parser.add_argument('batch_size', type=int, help='Batch Size during training. \n During inference, this has a crucial role to play.\n If batch size = -1, then Inversion Charge characteristics, $V_T$ characteristics and $\eta_0$ characteristics are calculated \n' +
                                                 'If batch size = 0, then predictions for a given Vgs is calculated. And also it plots the surface potential characteristics for Vgs in [-3Vt, 3Vt]. Saves a csv file of the predictions. It also makes surface potential predictions with different device parameters.\n' +
                                                 'If batch size = 2, then saves a combine plot for $V_G$ in [-3Vt, 3Vt] and surface potential as well\n' +
                                                 'If batch size other than above, then a document is created in which we can scrutinizingly observe the profile predictions.')
parser.add_argument('lr', type=float, help='learning rate of the adam optimizer')
parser.add_argument('do_training', type=int, help='Set 1 to perform Training, else setting this to zero will do the task of inference.')
parser.add_argument('train_continue', type=int, help='Set 1 to train from the previously stored model else 0. This is for the case in which the preloaded model has to start training again')
parser.add_argument('save_model', type=int, help='save the model. Set 1 to save else 0. This is used just to check the loss and the models wont get updated during training')
parser.add_argument('save_model_tag', type=str,
                    help='tag with which to save the model or saved model tag for inference. A folder will be created with this name and all the files and models will be stored in this. ' +
                         'Also, this is the name that will be used, during inference.')
parser.add_argument('training_data_reference', type=str, help='tag for dataset to choose for training')
parser.add_argument('test_data_reference', type=str, help='tag for dataset to choose for training. Ensure this to be same as training_data_reference')
parser.add_argument('update_text_file', type=int, help='Update the text file. Ensure this to be zero. This is when in case mistakenly you type a save_model_tag ' +
                                                       'to be the one which already exists. In case if you wish to update, then keep 1', default=0)
parse = parser.parse_args()

sigmoid_coeff = 40
n_hid = 3
scale_factor = 1
mid_neurons = 50 #
na_value = parse.N_A
tox_value = parse.t_ox
parse.t_ox = torch.tensor(parse.t_ox * 1e-9, requires_grad=False).to(device).reshape((-1, 1))
parse.N_A = torch.tensor(parse.N_A * 1e24, requires_grad=False).to(device).reshape((-1, 1))

if parse.do_training and not parse.train_continue:
    print(f"Device being used is '{device}'")
    code_file = open('./A_ODE_all_tox_NA.py')
    code = ""
    code = code.join(code_file.readlines())
    code_file.close()
    #Following is a short description of the model. When running, this will be saved as a text file
    model_notes = str(f"1) Model name is {parse.save_model_tag} and batch size is {parse.batch_size}\n"
                 +"2) Model has second derivative loss\n"
                 +f"3) Model has ({mid_neurons}, {n_hid}) , with Tanh() activation function\n"
                 +f"4) Model is trained with learning rate {parse.lr} and has 0.96 for 1000 epochs\n"
                 +f"5) Model has the training dataset as {parse.training_data_reference} and has the test dataset as {parse.test_data_reference}\n"
               +"6) tsi was taken as 50nm\n"
                +"7) N_A was taken as 1e24\n"
                 +"8) Model was trained with data dependent statistics for standardizing the inputs\n"
                  +"9) Model has L1 loss only.\n"
                      +"10) Model optimizer has no weight decay\n"
                       +"11) Model with variable tox\n"
                      + "12) Model has variable N_A\n"
                      + "13) Model has t_ox in 0.8e-9 to 1.5e-9\n"
                        +"14) Model has N_A in 1e23 to 1e24\n")

    if os.path.realpath(__file__).split('/')[-1] == f'{parse.save_model_tag}_code.py':
        sys.path.append('../')
    from make_note import make_notes_and_code
    make_notes_and_code(parse.save_model_tag, model_notes, code, parse.update_text_file, parse.train_continue) #This makes a textfile of above written text and also copies the current code for future reference and inference. This also makes a folder for current model 
else:
    if os.path.realpath(__file__).split('/')[-1] == f'{parse.save_model_tag}_code.py':
        sys.path.append('../')
    os.chdir(f'./{parse.save_model_tag}')

from stats_to_word import create_document
from Scripts.solve_bvp_any_fun import main
from integrate import integrate, calc_eta0, calc_multiple_eta0
from plot_many_sp import plot_many_sp
from predict_from_model import predict
from vt_vs_tox import vt_vs_tox


t_ox = 1e-9  # 1e-1
t_si = 200e-9  # 4e1
Vgs = parse.Vgs
do_training = parse.do_training
n_epochs = 150000
train_samples = parse.train_samples
test_samples = train_samples
lr = parse.lr
psi_t = 26e-3
batch_size = parse.batch_size
k = 1e8

N_A = torch.tensor(1e24, requires_grad=False)
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0 * 11.9
epsilon_sio2 = epsilon_0 * 3.9
delta_psi_MS = 0.21
n_i = torch.tensor(1e16, requires_grad=False)
psi_F = psi_t * torch.log(N_A / n_i).to(device)
q = 1.6e-19
Cox = epsilon_sio2 / t_ox
A = q * N_A / epsilon_si
C = (n_i / N_A) ** 2
a = (Cox / epsilon_si)

psi_P = psi_t*np.log(parse.N_A.item()/n_i.item())
Cox = epsilon_sio2 / parse.t_ox.item()
gamma = np.sqrt(2*q*epsilon_si*parse.N_A.item())/Cox
v_t = 2*psi_P + gamma*np.sqrt(2*psi_P)
Cox_ = Cox

print(
    f"tsi = {t_si}, and t_ox = {parse.t_ox}, and psi_t = {psi_t}, and A = {A}, and Vgs =  {Vgs} and psi_F  {psi_F} and a*tsi = {a * t_si} and v_t = {v_t}")

del A, a, N_A, psi_F, Cox, C


class act_fun(nn.Module):
    def __init__(self):
        super().__init__()
        # self.a = nn.parameter.Parameter(data=torch.ones((1, 1), dtype=torch.float64, requires_grad=True), requires_grad=True)
        # self.b = nn.parameter.Parameter(data=torch.ones((1, 1), dtype=torch.float64, requires_grad=True), requires_grad=True)
        # self.c = nn.parameter.Parameter(data=torch.ones((1, 1), dtype=torch.float64, requires_grad=True), requires_grad=True)
        # self.d = nn.parameter.Parameter(data=torch.ones((1, 1), dtype=torch.float64, requires_grad=True), requires_grad=True)

    def forward(self, x):
        act1 = nn.Tanh()
        # act1 = nn.Sigmoid()
        # return x*act1.forward(x)
        return act1.forward(x)
        # return torch.log(torch.abs(x))
        # value = (torch.exp(self.a*x) - torch.exp(-self.b*x))/(torch.exp(self.c*x) + torch.exp(-self.d*x))
        # return value
        # return nn.Sigmoid()(x)
#         # return torch.log(torch.abs(x))
#         # return x**2
#         # return torch.sin(x)


# class NN1(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden, n_layers):
#         super().__init__()
#         self.fcs = nn.Sequential(*[
#             nn.Linear(n_input, n_hidden//4, dtype=torch.float64),
#             nn.Tanh(),
#         ])
#         self.fch = nn.Sequential(nn.Linear(n_hidden//4, n_hidden//2, dtype=torch.float64), nn.Tanh(),
#                                  nn.Linear(n_hidden//2, n_hidden, dtype=torch.float64), nn.Tanh(),
#                                  nn.Linear(n_hidden, n_hidden//2, dtype=torch.float64), nn.Tanh(),
#                                  nn.Linear(n_hidden//2, n_hidden//4, dtype=torch.float64), nn.Tanh()) #Note the Architecture
#         self.fce = nn.Linear(n_hidden//4, n_output, dtype=torch.float64)
#
#     def forward(self, x):
#         x = self.fcs(x)
#         x = self.fch(x)
#         x = self.fce(x)
#         return x

class NN1(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden, dtype=torch.float64),
            # nn.BatchNorm1d(n_hidden, dtype=torch.float64),
            act_fun(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden, dtype=torch.float64),
                # nn.BatchNorm1d(n_hidden, dtype=torch.float64),
                act_fun(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output, dtype=torch.float64)
        # self.bn = nn.BatchNorm1d(n_output, dtype=torch.float64)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        # x = self.bn(x)
        return x

# class NN1(nn.Module):
#     def __init__(self, n_input, n_output, n_hidden, n_layers):
#         super().__init__()
#         self.fcs = nn.Sequential(*[
#             nn.Linear(n_input, n_hidden, dtype=torch.float64),
#             # nn.BatchNorm1d(n_hidden, dtype=torch.float64),
#             act_fun(),
#         ])
#         self.fch1 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         self.fch2 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch3 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch4 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch5 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch6 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch7 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         # self.fch8 = nn.Linear(n_hidden, n_hidden, dtype=torch.float64)
#         self.fce = nn.Linear(n_hidden, n_output, dtype=torch.float64)
#         # self.bn = nn.BatchNorm1d(n_output, dtype=torch.float64)
#
#     def forward(self, x):
#         x = self.fcs(x)
#         x = nn.Tanh()(self.fch1(x)) #+ x
#         x = nn.Tanh()(self.fch2(x)) #+ x
#         # x = nn.Tanh()(self.fch3(x)) + x
#         # x = nn.Tanh()(self.fch4(x)) + x
#         # x = nn.Tanh()(self.fch5(x)) + x
#         # x = nn.Tanh()(self.fch6(x)) + x
#         # x = nn.Tanh()(self.fch7(x)) + x
#         # x = nn.Tanh()(self.fch8(x)) + x
#         x = self.fce(x)
#         # x = self.bn(x)
#         return x


class NN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid_coeff = nn.parameter.Parameter(sigmoid_coeff*torch.ones((1, 1), dtype=torch.float64))
        # self.sigmoid_coeff1 = nn.parameter.Parameter(-sigmoid_coeff * torch.ones((1, 1), dtype=torch.float64))
        self.net_y1 = NN1(4, 1, mid_neurons, n_hid) # (20, 5)
        # self.param = torch.tensor(1.0, dtype=torch.float64, requires_grad=False)
        # self.net_y2 = NN1(2, 1, mid_neurons, n_hid)
        # self.net_y3 = NN1(2, 1, mid_neurons, n_hid)
        # self.net_y4 = NN1(2, 1, mid_neurons, n_hid)
        # self.net_y5 = NN1(2, 1, mid_neurons, n_hid)
        # self.net_y6 = NN1(2, 1, mid_neurons, n_hid).to(device)
        # self.net_y7 = NN1(2, 1, 20, n_hid).to(device)
        # self.net_y8 = NN1(2, 1, 20, n_hid).to(device)
        # self.net_y9 = NN1(2, 1, 20, n_hid).to(device)
        # self.net_y10 = NN1(2, 1,20, n_hid).to(device)
        # self.net_y11 = NN1(2, 1, 20, n_hid).to(device)
        # self.net_y12 = NN1(2, 1, 20, n_hid).to(device)
        # self.net_y13 = NN1(1, 1, 10, n_hid).to(device)
        # self.end = nn.Sequential(nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))
        # self.act_fun = nn.LeakyReLU(0.5)

    def forward(self, y, vgs, tox, na, ret_first_derivative=False, ret_second_derivative=False, ret_first_derivative_vgs=False, return_pred1_tsi=False, ret_second_derv_vgs=False):
        y_ = (y - mean1) / std1
        vgs_ = (vgs - mean2) / std2
        tox_ = (tox - mean3) / std3
        na_ = (na - mean4) / std4
        input = torch.cat((y_, vgs_, tox_, na_), axis=1)

        # input1 = torch.cat((self.net_y1.forward(input), vgs_), axis=1)
        # input2 = torch.cat((self.net_y2.forward(input1), vgs_), axis=1)
        # input3 = torch.cat((self.net_y3.forward(input2), vgs_), axis=1)
        # # input4 = torch.cat((self.net_y4.forward(input3), vgs_), axis=1)
        # x = self.net_y4.forward(input3)


        x =  self.net_y1.forward(input) #* (self.net_y2.forward(input)) * self.net_y3.forward(input).to(device) \
            #* self.net_y4.forward(input).to(device) * self.net_y5(input).to(device) * self.net_y6(input).to(device) #\
            # + ((self.net_y7.forward(input).to(device) * self.net_y8.forward(input).to(device) * self.net_y9.forward(input).to(device)  \
            # * self.net_y10.forward(input).to(device) * self.net_y11(input).to(device) * self.net_y12(input).to(device)))

        # calculating the model value at zero
        y1 = torch.zeros(y.shape, dtype=torch.float64, requires_grad=True).to(device)
        y1_ = (y1 - mean1) / std1

        bc1 = torch.cat((y1_, vgs_, tox_, na_), axis=1)

        # input1 = torch.cat((self.net_y1.forward(bc1), vgs_), axis=1)
        # input2 = torch.cat((self.net_y2.forward(input1), vgs_), axis=1)
        # input3 = torch.cat((self.net_y3.forward(input2), vgs_), axis=1)
        # input4 = torch.cat((self.net_y4.forward(input3), vgs_), axis=1)
        # x1 = self.net_y4.forward(input3)

        x1 =  self.net_y1.forward(bc1) #* (self.net_y2.forward(bc1)) * self.net_y3.forward(bc1).to(device) \
            #* self.net_y4.forward(bc1).to(device) * self.net_y5(bc1).to(device) * self.net_y6(bc1).to(device) #\
            # + ((self.net_y7.forward(bc1).to(device) * self.net_y8.forward(bc1).to(device) * self.net_y9.forward(bc1).to(device) \
            # * self.net_y10.forward(bc1).to(device) * self.net_y11(bc1).to(device) * self.net_y12(bc1).to(device)))

        x1_prime = torch.autograd.grad(x1, y1, torch.ones_like(x1), create_graph=True)[0]

        # calculating the model derivative value at t_si
        # if return_pred1_tsi:
        #     y1_ = t_si*torch.ones(y.shape, dtype=torch.float64, requires_grad=False).to(device)
        #     y1__ = (y1_ - mean1) / std1
        #     bc1_ = torch.cat((y1__, vgs_), axis=1).to(device)
        #
        #     x1_ =  (self.net_y1.forward(bc1_).to(device) + (self.net_y2.forward(bc1_)) + self.net_y3.forward(bc1_).to(device) \
        #         + self.net_y4.forward(bc1_).to(device) + self.net_y5(bc1_).to(device) + self.net_y6(bc1_).to(device)) #\
        #         # + self.net_y7.forward(bc1_).to(device) * self.net_y8.forward(bc1_).to(device) * self.net_y9.forward(bc1_).to(device) \
        #         # + self.net_y10.forward(bc1_).to(device) * self.net_y11(bc1_).to(device) * self.net_y12(bc1_).to(device)
        #
        #     B = ((x1 + scale_factor * a * t_si) / (1 + a * t_si)).to(device)
        #     return (-1/t_si)*(B + x1_)

        # print(f'Derv: Min: {torch.min(x1_prime)}, Max: {torch.max(x1_prime)}\n')
        # # The model:
        # print(f'Denm: Min: {torch.min((x1 * (1 + a * t_si) - t_si * x1_prime))}, Max: {torch.max((x1 * (1 + a * t_si) - t_si * x1_prime))}\n')
        # B = ((2*x1 + 4*a*t_si)/(2*a*t_si - 12*vgs + 2.0)).to(device) #15 HERE IS SCALE FACTOR!!!!!!!!!!!!!!!!!!!!
        # B = (2*t_si*x1_prime + 4*a*t_si)/(2 + 2*a*t_si - 12*vgs) - x1
        # B = (a*t_si)/((1 + a*t_si)*x1 - t_si*x1_prime)
        # B = (a*t_si - (1 + a*t_si)*x1 + t_si*x1_prime)/(1 + a*t_si)
        # B = (x1_prime + a - a * x1)/(a)
        C_ox = epsilon_sio2/tox
        a = C_ox/epsilon_si
        # B = (a*t_si*torch.exp(x1))/(self.param*torch.exp(-self.param) + t_si*(1 - torch.exp(-self.param))*x1_prime + a*t_si*(1 - torch.exp(-self.param)))

        B = (a*t_si*torch.exp(x1))/(1 + a*t_si + t_si*x1_prime)
        # B = (a*t_si)/((1 + a*t_si)*x1 - t_si*x1_prime)
        # B = (a * t_si) / (1 + a * t_si + x1)

        # B = ((x1_prime + a)*t_si/(1 + a*t_si) - x1)#((x1 + scale_factor * a * t_si) / (1 + a * t_si)).to(device)
        # pred = vgs * (self.sigmoid(12 *vgs * y/t_si))*((1 - y / t_si) * (B + (y/t_si) * x)).to(device)  #15 HERE IS SCALE FACTOR!!!!!!!!!!!!!!!!!!!!
        # pred = vgs * (self.sigmoid(12 * vgs * y / t_si)) * ((1 - y / t_si) * (B + x)).to(device)
        # pred = vgs*(1 - (y/t_si)**2)*(B + x)
        # pred = vgs * B * (1 - torch.exp((1 - y / t_si))) * torch.exp(-x)
        # pred = vgs*B*(1 - torch.exp(-self.param*(1 - y/t_si)))*torch.exp(-x)


        pred = vgs*B*(1 - y/t_si)*torch.exp(-x)
        # pred = vgs*(1-y/t_si)*B*x
        # pred = vgs * B * (1 - y / t_si) * torch.exp(-(y/t_si)*x)

        if ret_first_derivative:
            pred1 = torch.autograd.grad(pred, y, torch.ones_like(pred), create_graph=True)[0]
            return pred1
        if ret_second_derivative:
            pred1 = torch.autograd.grad(pred, y, torch.ones_like(pred), create_graph=True)[0]
            pred2 = torch.autograd.grad(pred1, y, torch.ones_like(pred1), create_graph=True)[0]
            return pred, pred2
        if ret_first_derivative_vgs:
            pred1 = torch.autograd.grad(pred, vgs, torch.ones_like(pred), create_graph=True)[0]
            return pred1
        if ret_second_derv_vgs:
            pred1 = torch.autograd.grad(pred, vgs, torch.ones_like(pred), create_graph=True)[0]
            pred2 = torch.autograd.grad(pred1, vgs, torch.ones_like(pred1), create_graph=True)[0]
            return pred2
        return pred


# def initialise_weights(model):
#     for layer in model.modules():
#         if isinstance(layer, nn.Linear):
#             mu = np.random.uniform(-0.02, 0.02, (1,))
#             sigma = np.random.uniform(0, 0.02, (1,))
#             nn.init.normal_(layer.weight.data, float(mu), float(sigma))


if __name__ == '__main__':

    data_f_test = pd.read_csv(f'../Data/data_y_100_100_58_vgs_100_50_100_{parse.test_data_reference}.csv')
    y_test = data_f_test['y'].values.astype(np.float64)
    vgs_test = data_f_test['Vgs'].values.astype(np.float64)
    psi_test = data_f_test['psi'].values.astype(np.float64)
    test = torch.tensor(y_test, dtype=torch.float64)
    train_target_test = torch.tensor(vgs_test, dtype=torch.float64)
    train_target1_test = torch.tensor(psi_test, dtype=torch.float64)
    del y_test, vgs_test, psi_test, data_f_test

    data_f = pd.read_csv(f'../Data/data_y_100_100_58_vgs_100_50_100_{parse.training_data_reference}.csv')
    y = data_f['y'].values.astype(np.float64)
    vgs = data_f['Vgs'].values.astype(np.float64)
    train = torch.tensor(y, dtype=torch.float64)
    train_target = torch.tensor(vgs, dtype=torch.float64)
    train_tensor = TensorDataset(train, train_target)
    mean1 = torch.mean(train).to(device)
    std1 = torch.std(train).to(device)
    mean2 = torch.mean(train_target).to(device)
    std2 = torch.std(train_target).to(device)
    torch.manual_seed(100)
    np.random.seed(100)
    del y, vgs, data_f

    t_oox = torch.linspace(0.8e-9, 1.5e-9, 29, dtype=torch.float64).to(device)
    mean3 = torch.mean(t_oox).to(device)
    std3 = torch.std(t_oox).to(device)

    N_AA = torch.linspace(1e23, 1e24, 41, dtype=torch.float64).to(device)
    mean4 = torch.mean(N_AA).to(device)
    std4 = torch.std(N_AA).to(device)

    if do_training:
        model = NN(2, 1, 20, 4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.96) #1000, 0.96
        train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
        if parse.train_continue: #Assuming we most probably wont consider retraining!!
            model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_39215.pth').state_dict())
            optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
            epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
            print(f"Loaded Saved Model")
            # start_epoch = np.load(f'{parse.save_model_tag}_training_loss.npy').size
        else:
            # initialise_weights(model)
            print(f"Started Training From Scratch")
            strat_epoch = 0
        loss = nn.L1Loss(reduction='sum')
        # loss = nn.MSELoss(reduction='sum')
        loss_gt = nn.MSELoss(reduction='sum')
        epoch_list = []
        loss_list = []
        loss_list_bc1 = []
        loss_list_bc2 = []
        batch_loss_list = []
        max_loss = np.inf
        temp_var = 0
        last_epoch_saved = 0
        len_of_loader = train_loader.__len__()
        print(f"These are the number of datapoints: {len_of_loader * parse.batch_size}")
        avg_epoch_time = 0
        nans_encountered = 0
        for epoch in range(n_epochs):
            start_time_epoch = time.time()
            epoch_scheduler.step()
            check_loss = 0
            model.train()
            if epoch == 0:
                model.requires_grad_(False)
                for index, (y, vgs) in enumerate(train_loader):
                    # pred = model.forward(y.reshape((-1, 1)), vgs.reshape((-1, 1))).to(device)
                    indices = np.round_(np.random.uniform(0, 28, size=(y.shape[0], 1)))
                    indices1 = np.round_(np.random.uniform(0, 40, size=(y.shape[0], 1)))
                    y = y.to(device)
                    vgs = vgs.to(device)
                    y.requires_grad = True
                    pred, pred2 = model.forward(y.reshape((-1, 1)), vgs.reshape((-1, 1)), t_oox[indices].reshape((-1, 1)), N_AA[indices1].reshape((-1, 1)), ret_second_derivative=True)
                    psi_F = psi_t * torch.log(N_AA[indices1] / n_i).to(device)
                    A = q * N_AA[indices1] / epsilon_si
                    y.requires_grad = False
                    with torch.no_grad():
                        # os.system('nvidia-smi')
                        # gt_loss = loss(pred, train_target1_test).item()
                        check_loss += (((torch.pow(((((loss((-(torch.exp(-pred/(scale_factor*psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(scale_factor*psi_t)) - 1))) * A, (pred2)))))), 1)))).item()
                        # check_loss += gt_loss
                model.requires_grad_(True)
            else:
                check_loss = np.sum(np.array(batch_loss_list))
                batch_loss_list.clear()
            loss_list.append(check_loss)
            epoch_list.append(epoch)
            # if not parse.train_continue:
            # else:
            #     np.save(f'{parse.save_model_tag}_training_loss.npy' ,np.array([np.load(f'{parse.save_model_tag}_training_loss.npy'), np.array(loss_list)]).reshape((-1, 1)))
            # print("Epoch: ", epoch)
            # print(
            #     f"The Total Training log10 loss is: {np.log10(check_loss)} and save log10 loss is {np.log10(max_loss)}, \nThe differential loss is {np.log10(check_loss)} and the gt loss is {None}")
            if (max_loss > check_loss) and parser.parse_args().save_model:
                max_loss = check_loss
                torch.save(model, f'Poisson_model_{parse.save_model_tag}_epoch_{epoch}.pth')
                torch.save(optimizer, f'optimizer_mul_{parse.save_model_tag}.pth')
                torch.save(epoch_scheduler, f'epoch_scheduler_{parse.save_model_tag}.pth')
                print("Epoch: ", epoch)
                print(f"Model saved with sum batch loss log10 as {np.log10(max_loss)}")
                last_epoch_saved = epoch
            elif math.isnan(check_loss):
                nans_encountered += 1
                model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_{last_epoch_saved}.pth').state_dict())
                optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
                epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9 #decreasing the learning rate to 90%
                loss_list.pop()
            print(f'Number of nans encountered: {nans_encountered}')
            np.save(f'{parse.save_model_tag}_training_loss.npy', np.array(loss_list))


            for index, (y, vgs) in enumerate(train_loader):
                y = y.reshape((-1, 1)).to(device)
                vgs = vgs.reshape((-1, 1)).to(device)
                y.requires_grad = True
                # vgs.requires_grad = True
                # t_oox.requires_grad = True
                # N_AA.requires_grad = True

                indices = np.random.choice(28, size=y.shape, replace=True)
                indices1 = np.random.choice(40, size=y.shape, replace=True)

                # indices = np.round_(np.random.uniform(0, 56, size=(y.shape[0], 1)))
                # indices1 = np.round_(np.random.uniform(0, 20, size=(y.shape[0], 1)))
                # pred = model.forward(y, vgs).to(device)
                # pred1 = model.forward(y, vgs, ret_first_derivative=True)
                pred, pred2 = model.forward(y, vgs, t_oox[indices], N_AA[indices1], ret_second_derivative=True)
                # y.requires_grad = False
                # pred1_tsi = model.forward(y, vgs, return_pred1_tsi=True)
                # reweight = torch.zeros_like(y, dtype=torch.float64, requires_grad=False)
                # reweight[vgs >= 0] = 1
                # reweight[vgs < 0] = 1e4

                # if torch.min(pred).item() >= -2 and torch.max(pred).item() <= 2: #Testing Condition
                # if epoch % 5 in [0, 1, 2]:
                # loss_ = (loss(torch.asinh((-(torch.exp(-pred/(scale_factor*psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(scale_factor*psi_t)) - 1))) * A), torch.asinh(pred2))).to(device)
                A = q * N_AA[indices1] / epsilon_si
                psi_F = psi_t * torch.log(N_AA[indices1] / n_i).to(device)
                # loss_ = ((((((((loss((-(torch.exp(-pred/(psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(psi_t)) - 1))) * A, (pred2)))))))))).to(device)
                loss_ = ((torch.log((((((loss((-(torch.exp(-pred/(psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(psi_t)) - 1))) * A, (pred2)))))))))).to(device)
                #loss_ = (((torch.pow(((((loss((-(torch.exp(-pred/(psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(psi_t)) - 1))) * A, (pred2)))))), 1/8)))).to(device)
                # else:
                #     loss_ = (((torch.pow(((((loss((-(
                #                 torch.exp(-pred / (scale_factor * psi_t)) - 1 - torch.exp(-2 * psi_F / psi_t) * (
                #                     torch.exp(pred / (scale_factor * psi_t)) - 1))), (pred2 / (A * scale_factor))))))),
                #                          1)))).to(device)
                # else:
                #     loss_ = ((((loss(-(torch.exp(-pred / psi_t) - 1 - (torch.exp((pred - 2 * psi_F) / psi_t)) + torch.exp(-2 * psi_F / psi_t)), pred2 / A))))).to(device)
                #loss_ = torch.pow(torch.sum(torch.abs((pred1 ** 2) - (pred1_tsi**2 + (2*A)*(psi_t*torch.exp((-pred)/psi_t) + (pred - psi_t) + (torch.exp((-2*psi_F)/psi_t))*(psi_t*torch.exp(pred/psi_t) - pred - psi_t))))), 0.125).to(device)
                # loss_f = torch.pow(torch.sum(reweight*(torch.abs(pred2/(2*A*torch.exp(-psi_F/psi_t)) - (torch.sinh((pred - psi_F)/psi_t) + torch.sinh(psi_F/psi_t))))), 0.125).to(device)
                # loss_t = ((loss(psi, pred).to(device))).to(device)
                # loss_ = loss_f #+ 1e12*loss_t
                loss_.backward()

                # for param in model.parameters():
                #     # print(param.grad)
                #     param.grad = 1/(param.grad + 1e-9)
                #     # print(param.grad)

                optimizer.step()
                optimizer.zero_grad()
                # net_loss = loss_

                # pred = model.forward(y, vgs).to(device)
                # pred1 = model.forward(y, vgs, ret_first_derivative=True)
                # pred2 = model.forward(y, vgs, ret_second_derivative=True)
                # pred1_tsi = model.forward(y, vgs, return_pred1_tsi=True)
                # loss_ =
                # loss_.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # loss1 = loss(psi, pred).to(device)
                # batch_loss_list.append(loss_.item()**12)
                batch_loss_list.append(np.exp(loss_.item()))
                # else:
                #     batch_loss_list.append((loss_.item()))
                # print(f"This is the batch number {index}, this is the log10 loss {10*np.log10(loss_.item())}, this is the range of y {torch.min(y).item(), torch.max(y).item()}, this is the range of vgs {torch.min(vgs).item(), torch.max(vgs).item()}")

            #     if ((epoch + 1) % 10 == 0 or epoch == 0) and index % 100 == 0:
            #         model.eval()
            #         # epoch_list.append(epoch)
            #         # loss_list.append(net_loss.item())
            #         tempv = Vgs * torch.ones((1, 1), dtype=torch.float64).to(device)
            #         tempy = torch.zeros((1, 1), requires_grad=True, dtype=torch.float64).to(device)
            #         pred_bc = model.forward(tempy, tempv).to(device)
            #         pred1_bc = model.forward(tempy, tempv, ret_first_derivative=True).to(device)
            #         initial_value = pred_bc.item()
            #         print(f"This is the initial value: {initial_value} and initial derivative: {pred1_bc.item()}")
            #         print(
            #             f"This is log10 train loss: {np.log10(net_loss.item())}, < {-temp_var + net_loss.item()}>, This is log10 save loss: {np.log10(max_loss)}, and the gt loss {None} and this is the tag: {parse.save_model_tag} and this is the dataset reference {parse.training_data_reference}")
            #         temp_var = net_loss.item()
            #         print(
            #             f"Total Time Elapsed from starting = {(time.time() - start_time) // 3600} hours {(time.time() - start_time) // 60 + 60 * ((time.time() - start_time) // 3600)} Minutes {(time.time() - start_time) - 60 * ((time.time() - start_time) // 60 - 60 * ((time.time() - start_time) // 3600))} Seconds")
            #         print()
            avg_epoch_time += time.time() - start_time_epoch
            print(f'The maximum batch loss is: {np.max(np.array(batch_loss_list)) :e} and the minimum batch loss is: {np.min(np.array(batch_loss_list)) :e} and the total loss is {np.exp(loss_.item()) :e}')
            if epoch % 25 == 0 or epoch < 25:
                print(f'This is the epoch number {epoch} and this is the log10 training loss {np.log10(np.sum(np.array(batch_loss_list)))}')
                print("Last Epoch saved", last_epoch_saved, f" Learning Rate: ")
                # print(f"This is the model: {model.param}\n")
                for p in optimizer.param_groups: print(p['lr'])
                print(f"Average Time required for an epoch is {avg_epoch_time if epoch < 25 else avg_epoch_time/25} Seconds")
                avg_epoch_time = 0
            # print()

            # check_loss = np.sum(np.array(batch_loss_list))
            # batch_loss_list.clear()
            # print(f"Epoch number: {epoch}")
            # print("Last Epoch saved", last_epoch_saved, f" Learning Rate: ")
            # for p in optimizer.param_groups: print(p['lr'])
            # # if epoch % 500 == 0:
            # #     for p in optimizer.param_groups: p['lr'] = parse.lr
            # # elif epoch % 50 == 0:
            # #     for p in optimizer.param_groups: p['lr'] = p['lr']*0.4
            # print(f"This is the log10 training loss: {np.log10(check_loss)}")

        plt.title("Training Plot for Ground Truth")
        plt.plot(np.array(epoch_list), np.array(loss_list))
        plt.legend(['First', 'BC1', 'BC2'])
        plt.savefig('Model_Training_Error.png')
        plt.close()
    background = '#D7E5E5'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['legend.title_fontsize'] = 10
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['axes.labelweight'] = 'heavy'
    mpl.rcParams['axes.linewidth'] = 2
    torch.set_printoptions(threshold=torch.inf, precision=20)
    check_epoch = parse.save_model
    np.random.seed(1000)
    model = torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_{check_epoch}.pth').to(device)
    print(f'These are all the arguments: {parse}')
    if parse.batch_size == -1:
        y = torch.ones((1, 1), dtype=torch.float64).to(device) * 200e-9  # in nm
        # vgs = parse.Vgs * torch.ones_like(y, dtype=torch.float64)
        y_prime = torch.linspace(0, t_si, 8000, dtype=torch.float64).reshape(-1, 1).to(device)
        volt_array = (torch.linspace(0, 3 * v_t, 20, dtype=torch.float64)).reshape((-1, 1)).to(device)
        volt_list = []
        int_val_list = []
        fem_val_list = []
        anly_list = []
        for vgs in tqdm(volt_array):
            volt_list.append(vgs.item() / v_t)  # plotting against vg/v_t
            fem_solution, _ = main(y_new=np.linspace(0, t_si, 8000, dtype=np.float64).reshape(-1, ),
                                   VGS=vgs.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                                   NA=parse.N_A.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                                   Tox=parse.t_ox.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                                   if_print=False)  #this part gives us fem solution
            predictions_model = model(y_prime, vgs * torch.ones_like(y_prime, dtype=torch.float64),
                                      parse.t_ox * torch.ones_like(y_prime, dtype=torch.float64),
                                      parse.N_A * torch.ones_like(y_prime, dtype=torch.float64)).detach().cpu().numpy()
            int_val_list.append(
                integrate(model, y, vgs, parse.t_ox, parse.N_A.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                          discrete=predictions_model.reshape((-1,)).astype(np.float64)))
            fem_val_list.append(
                integrate(model, y, vgs, parse.t_ox, parse.N_A.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                          discrete=fem_solution[0].astype(np.float64)))
            anly_list.append(-Cox_ * (vgs.item() - fem_solution[0][0] - gamma * (np.sqrt(fem_solution[0][0]))))
        df = pd.DataFrame({'Vgs': np.array(volt_list).reshape(-1, ), 'FEM charge': np.array(fem_val_list).reshape(-1, ),
                           'NN charge': np.array(int_val_list).reshape(-1, ),
                           'V_T': v_t * np.ones_like(np.array(fem_val_list)).reshape(-1, ),
                           'CSA': np.array(anly_list).reshape(-1, )})
        df.to_csv(f'Charge_predictions.csv') #saving all charge predictions to a .csv file
        plt.plot(volt_list, np.array(int_val_list).reshape(-1, 1))
        plt.plot(volt_list, np.array(fem_val_list).reshape(-1, 1))  #
        plt.plot(volt_list, (np.array(anly_list)).reshape(-1, 1))
        plt.legend(['From NN', 'From FEM', 'Anly'])
        plt.grid(True)
        plt.title('Q vs Vg/Vt')
        plt.xlabel('Vg/Vt')
        plt.ylabel('Q')
        plt.savefig('QvsVg_Vt.png')
        plt.close()

        plt.semilogy(volt_list, (np.abs(np.array(int_val_list))).reshape(-1, 1))
        plt.semilogy(volt_list, (np.abs(np.array(fem_val_list))).reshape(-1, 1))
        plt.semilogy(volt_list, np.abs(np.array(anly_list)).reshape(-1, 1))
        plt.legend(['From NN', 'From FEM'])
        plt.grid(True)
        plt.title('Q vs Vg/Vt')
        plt.xlabel('Vg/Vt')
        plt.ylabel('log10(|Q|)')
        plt.savefig('QvsVg_Vt_log10.png')
        plt.close()

        print(
            f'\nPredicted value of eta0 = {calc_eta0(np.log10(np.abs(np.array(int_val_list))).reshape(-1, 1), np.array(volt_list))} and Analytical value = {1 + (gamma / (2 * np.sqrt(2 * psi_P)))}')
        calc_multiple_eta0(model, main)
        vt_vs_tox(model)
        exit('\nExit at line 661 after ending the integration process and plotting for eta0 and vt for the desired. Plotted \n a) |Q| vs $V_G$ plot. \n b) $\eta_0$ vs $t_{ox}$ plot. \n c) $V_T$ vs $t_{ox}$ plot.')
    predp = True
    if parse.batch_size == 0:
        array = v_t * np.array([Vgs], dtype=np.float64)
    else:
        array = v_t.item() * np.array([-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3])

    color = np.random.uniform(0, 1, size=(2 * array.shape[0], 3))
    color = np.append(color, np.ones((color.shape[0], 1)), axis=1)
    color_list = list(color)
    y_test_list = []
    if os.path.exists(f'./{parse.save_model_tag}_outputs.docx'):
        os.remove(f'./{parse.save_model_tag}_outputs.docx')
    for index, Vgs in enumerate(array):
        Vgs = round(Vgs, 5)
        np.save(f'./{parse.save_model_tag}_vgs.npy', Vgs)
        psi0_required = True if (index + 1 == len(array) and parse.batch_size != 0) else False
        test_surface_vgs = torch.round(
            (torch.linspace(-3 * v_t, 3 * v_t, 300, dtype=torch.float64)).reshape((-1, 1)),
            decimals=5).to(device)

        psi0_required = False
        print(
            "Skip on demand of not solving for the surface potential... may get error for surface potential as its not being calculated for saving time and keeping psi0_required=False")

        if psi0_required:
            vg = test_surface_vgs[-1].detach().cpu().numpy()
        else:
            vg = None

        # if not psi0_required:
        #     required_array, psi0_list = main(psi0_required, psi0_samples=300, VGS=vg)
        # else:
        #     psi0_required = False
        required_array, psi0_list = main(psi0_required, psi0_samples=300, VGS=vg)

        train_plot = (np.linspace(0, t_si, test_samples, dtype=np.float64)).reshape((-1, 1))
        data_test_y = torch.linspace(0, t_si, test_samples, dtype=torch.float64).reshape((-1, 1)).to(device)
        data_test_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)
        temp2_y = torch.zeros((test_samples, 1), dtype=torch.float64, requires_grad=True).reshape((-1, 1)).to(device)
        temp2_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)

        vgs_width = (test_surface_vgs[1] - test_surface_vgs[0]).detach().cpu().numpy()
        test_surface_vgs.requires_grad = True

        test_surface_y = torch.zeros(test_surface_vgs.shape, dtype=torch.float64).to(device)

        surface_potential = model.forward(test_surface_y, test_surface_vgs,
                                          parse.t_ox * torch.ones_like(test_surface_y),
                                          parse.N_A * torch.ones_like(test_surface_y)).detach().cpu().numpy()

        surface_potential_derivative = model.forward(test_surface_y, test_surface_vgs,
                                                     parse.t_ox * torch.ones_like(test_surface_y),
                                                     parse.N_A * torch.ones_like(test_surface_y),
                                                     ret_first_derivative_vgs=True).detach().cpu().numpy()
        surface_potential_second_derivative = model.forward(test_surface_y, test_surface_vgs,
                                                            parse.t_ox * torch.ones_like(test_surface_y),
                                                            parse.N_A * torch.ones_like(test_surface_y),
                                                            ret_second_derv_vgs=True).detach().cpu().numpy()
        test_surface_vgs = test_surface_vgs.detach().cpu().numpy()

        if psi0_required:
            np.save(f'./{parse.save_model_tag}_vgs.npy', test_surface_vgs[-1] + vgs_width) #saving the voltage for the case that it can be harnessed by other scripts
            psi0_list.append(main(need_psi0=True)) #We get here the numerical surface potential
            surface_potential_derivative_true = list(np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0) / (vgs_width)) #We get here numerical surface potential discrete derivative
            np.save(f'./{parse.save_model_tag}_vgs.npy', test_surface_vgs[-1] + 2 * vgs_width)
            psi0_list.append(main(need_psi0=True))
            surface_potential_second_derivative_true = list(np.diff(np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0), axis=0) / vgs_width ** 2) #We get here numerical surface potetial discrete double derivative
            psi0_list.pop()
            psi0_list.pop()

        data_test_y.requires_grad = True
        prediction = model.forward(data_test_y, data_test_vgs, parse.t_ox * torch.ones_like(data_test_y),
                                   parse.N_A * torch.ones_like(data_test_y)).to(device)
        prediction_derivative = model.forward(data_test_y, data_test_vgs, parse.t_ox * torch.ones_like(data_test_y),
                                              parse.N_A * torch.ones_like(data_test_y),
                                              ret_first_derivative=True).to(device)
        _, prediction_second_derivative = model.forward(data_test_y, data_test_vgs,
                                                        parse.t_ox * torch.ones_like(data_test_y),
                                                        parse.N_A * torch.ones_like(data_test_y),
                                                        ret_second_derivative=True)

        prediction = prediction.detach().cpu().numpy()
        prediction_derivative = prediction_derivative.detach().cpu().numpy()
        prediction_second_derivative = prediction_second_derivative.detach().cpu().numpy()
        y_test = required_array[0].reshape((test_samples, 1))
        y_test_list.append(y_test)
        A = q * parse.N_A / epsilon_si
        psi_F = psi_t * torch.log(parse.N_A / n_i).to(device)
        y_test_second_derivative = -A.item() * (
                np.exp(-y_test / psi_t) - 1 - np.exp(-2 * psi_F.item() / psi_t) * (np.exp(y_test / psi_t) - 1))
        y_test_derivative = required_array[1].reshape((test_samples, 1))
        test_loss = np.abs(prediction - y_test)
        test_loss_derivative = np.abs(prediction_derivative - y_test_derivative)

        if psi0_required:
            surface_potential_error = np.abs(surface_potential - np.array(psi0_list, dtype=np.float64))

        if parse.batch_size == 0:
            df = pd.DataFrame({'Vgs/V_T': (np.ones_like(prediction) * (array / v_t)).reshape(-1, ),
                               'Vgs': (np.ones_like(prediction) * array).reshape(-1, ),
                               'V_T': (np.ones_like(prediction) * v_t).reshape(-1, ),
                               'Prediction_NN': (prediction).reshape(-1, ), 'FEM Solution': (y_test).reshape(-1, ),
                               'Absolute Error': np.abs((prediction).reshape(-1, ) - (y_test).reshape(-1, )),
                               'y': data_test_y.detach().cpu().numpy().reshape((-1,))})
            df.to_csv(f'vgs={parse.Vgs}_{parse.save_model_tag}.csv') #saving the potential profile with absolute error in an excel file.
            df = pd.DataFrame({'vgs': test_surface_vgs.reshape(-1, ), 'predicted surface potential': surface_potential.reshape(-1, )})
            df.to_csv(f'surface_potential_only_predictions_{parse.save_model_tag}.csv') #saving the surface potential predictions in a excel file.
            plot_many_sp(model, v_t) #plotting various predictions for surface potential with different values of t_ox and N_A. see from where this function is called.
            print(f'Saved the predictions for vgs = {v_t * parse.Vgs}, equivalent to vgs/v_t = {parse.Vgs} as !!vgs={parse.Vgs}_{parse.save_model_tag}.csv!!')
            print(f'Saved the surface potential predictions for vgs in -{parse.Vgs}vt to {parse.Vgs}vt, and saved as !!surface_potential_only_predictions{parse.save_model_tag}.csv!!')
            fem_predictions, _ = main(y_new=np.linspace(0, t_si, 3000, dtype=np.float64), VGS=parse.Vgs * v_t.item(),
                                      NA=parse.N_A.item(), Tox=parse.t_ox.item(), if_print=False)
            predict(model, None, parse.Vgs * v_t, parse.t_ox, parse.N_A,
                    save_name=f'normv={parse.Vgs}_NA={parse.N_A.item() :e}_tox={parse.t_ox.item() :e}',
                    fem_predictions=fem_predictions[0], v_t=v_t.item(),
                    if_use_y=torch.linspace(0, t_si, 3000, dtype=torch.float64).reshape(-1, ).to(device))  # saving all the predictions in an excel file. This is saving again the profile.
            exit('Exit at line 724 after saving the predictions for the desired voltages in numpy format')

        if parse.batch_size == 2 and parse.batch_size != 0:
            if predp:
                predp_list = []
                predp_list.append(prediction)
                predp = False
            else:
                predp_list.append(prediction)
            plt.plot(train_plot * 10 ** 9, y_test / psi_P, marker='o', markersize=5, markevery=15, mfc='none',
                     linewidth=0.0000001, c=color_list[index])
            if index == len(array) - 1:
                plt.xlabel("y (nm)")
                plt.ylabel("$\Psi$ / $\Phi_B$")
                plt.legend(['$V_G/V_T$=' + str(np.round_(i / v_t.item(), 5)) for i in array])
                for ind, i in enumerate(predp_list):
                    plt.plot(train_plot * 10 ** 9, i / psi_P, linewidth=2, mfc='none', c=color_list[ind])
                plt.tight_layout()
                plt.xlim(-1, 60)
                plt.savefig('zoom_in_Poisson_Model_Prediction_vs_actual_all.png', dpi=200) #zoomed in version of the potential profile \Psi (y) vs numerical solution
                plt.xlim(-1, 205)
                plt.text(140, -0.3, f'Epoch: {check_epoch}', fontsize=13)
                plt.text(75, 0.4, f'$N_A={na_value}$' + '$ x 10^{24}m^{-3}, $' + '$t_{ox}=$' + f'$ {tox_value}nm $', fontsize=13)
                plt.savefig('Poisson_Model_Prediction_vs_actual_all.png', dpi=200) # potential profile \Psi (y) vs numerical solution
                plt.close()
                print('Saved all the plots in a single plot')
                for ind, i in enumerate(predp_list):
                    plt.semilogy(train_plot * 10 ** 9, np.abs(i - y_test_list[ind]), c=color_list[ind], mfc='none')
                plt.xlabel('y (nm)')
                plt.ylabel('$|\psi_{Numerical} - \psi_{ML}|$')
                plt.xlim(-1)
                plt.ylim(1e-11, 1e0)
                plt.tight_layout()
                plt.savefig('Absolute_error_plot.png', dpi=200) #Plotting the absolute error plot in semilog scale
                plt.close()
            else:
                continue
        plt.plot(train_plot, prediction)
        plt.plot(train_plot, y_test)
        plt.title(f"Model prediction vs Actual for Vgs = {Vgs}")
        plt.legend(['Actual_prediction', 'Ground_Truth'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction_vs_actual.png") #for plotting for a particular V_G, prediction vs numerical
        plt.close()

        plt.plot(train_plot, prediction_derivative)
        plt.plot(train_plot, y_test_derivative)
        plt.legend(['Actual_prediction_derivative', 'Ground_Truth_derivative'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction_vs_actual_derivative.png") #for plotting for a particular V_G first derivative, prediction vs numerical
        plt.close()

        plt.plot(train_plot, prediction_second_derivative)
        plt.plot(train_plot, y_test_second_derivative)
        plt.legend(['Actual_prediction_second_derivative', 'Ground_Truth_second_derivative'])
        plt.savefig("Poisson_Model_Prediction_vs_actual_second_derivative.png") #for plotting for a particular V_G second derivative, prediction vs numerical
        plt.close()

        plt.plot(train_plot, y_test)
        plt.grid(True)
        plt.savefig("Poisson_Model_Ground_Truth.png")  #plotting just the actual solution of the numercial method
        plt.close()

        plt.plot(train_plot, prediction)
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction.png") #plotting just the predicted solution of ML model
        plt.close()

        plt.plot(train_plot, test_loss)
        plt.title(f'log10 Absolute_error_Plot_for_Vgs = {Vgs}') #plotting the loss curve for a particular V_G 
        plt.legend(['Loss Curve'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Loss.png")
        plt.close()
        create_document(parse.save_model_tag,
                        [f'./Poisson_Model_Prediction_vs_actual.png', f'./Poisson_Model_Loss.png']) #This adds all requested figure names to a document to scrutinizingly observe.

        plt.plot(train_plot, test_loss_derivative)
        plt.title('Absolute_error_Plot_derivative')
        plt.grid(True)
        plt.legend(['Loss Curve'])
        plt.savefig("Poisson_Model_Loss_derivative.png") #This is for the loss incurred in the first derivative as compared with the predicted solution
        plt.close()

        if index == len(array) - 1 and parse.batch_size != 0:
            plt.plot(test_surface_vgs / v_t.item(), np.array(surface_potential)/psi_F.item())
            plt.plot(test_surface_vgs / v_t.item(), np.array(psi0_list)/psi_F.item())
            plt.title("Surface Potential")
            plt.xlabel("$V_G$ / $V_T$ ")
            plt.ylabel("$\Psi_s$ / $\Psi_B$")
            plt.legend(["Predicted Surface Potential", "Actual Surface Potential"])
            plt.grid(True)
            plt.savefig("Surface_Potential.png") #Saving the normalised surface potential for prediction vs numerical solution
            plt.close()

            plt.plot(test_surface_vgs / v_t.item(), np.log10(surface_potential_error))
            plt.title("log10 Surface Potential absolute error")
            plt.xlabel('$V_{gs}/V_T$')
            plt.grid(True)
            plt.savefig("Surface_Potential_absolute_error.png") #saving surface potential error observed in terms of log10(error)
            plt.close()

            plt.plot(test_surface_vgs / v_t.item(), surface_potential_derivative)
            plt.plot(test_surface_vgs / v_t.item(), surface_potential_derivative_true)
            plt.title("Surface Potential Derivative")
            plt.xlabel("Vgs/Vt")
            plt.ylabel("$d\Psi(0)/dV_{gs}$")
            plt.legend(["Predicted Surface Potential Derivative", "Actual Surface Potential Derivative"])
            plt.grid(True)
            plt.savefig("Surface_Potential_derivative.png") #saving the ML model prediction and numerical solution prediction for surface potential first derivative
            plt.close()

            pd.DataFrame({'Vgs': np.array(test_surface_vgs).reshape(-1, ),
                          'Vg/v_t': (test_surface_vgs / v_t.item()).reshape(-1, ),
                          'SP_derv_ML': np.array(surface_potential_derivative).reshape(-1, ),
                          'SP_derv_FEM': np.array(surface_potential_derivative_true).reshape(-1, ),
                          'Psi0_FEM': np.array(psi0_list).reshape(-1, ),
                          'Psi0_NN': np.array(surface_potential).reshape(-1, )}).to_csv(f'SP_derv.csv') #saving the surface potential and its derivative values in a csv file.
            print('Saved the Predictions and actual values of surface potential derivatives in SP_derv.csv')

            plt.plot(test_surface_vgs, surface_potential_second_derivative)
            plt.plot(test_surface_vgs, surface_potential_second_derivative_true)
            plt.grid(True)
            plt.title("Surface Potential Second Derivative")
            plt.xlabel("Vgs")
            plt.ylabel("$d^{2}\Psi_s/{d{V^2}_{gs}}$")
            plt.legend(["Predicted Surface Potential Second Derivative", "Actual Surface Potential Second Derivative"])
            plt.savefig("Surface_Potential_second_derivative.png") #plotting surface potential second derivative
            plt.close()
            create_document(parse.save_model_tag,
                            [f'./Surface_Potential.png', f'./Surface_Potential_absolute_error.png',
                             f'Surface_Potential_derivative.png', f'Surface_Potential_second_derivative.png']) #putting the required stuffs in a document

    print(f"Time Required to Execute the Training is {time.time() - start_time} Seconds")