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
from random import shuffle


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(threshold=torch.inf)
parser = argparse.ArgumentParser(description='Model Variables')
parser.add_argument('Vgs', type=float, help='Value of Vgs')
parser.add_argument('t_ox', type=float, help='Enter the tox value in nm')
parser.add_argument('N_A', type=float, help='Enter the N_A value as a coefficient to 1e24')
parser.add_argument('train_samples', type=int, help='Number of Training Samples')
parser.add_argument('batch_size', type=int, help='Batch Size')
parser.add_argument('lr', type=float, help='learning rate of the model')
parser.add_argument('do_training', type=int, help='Set 1 to perform Training')
parser.add_argument('train_continue', type=int, help='Set 1 to train from the previously stored model else 0')
parser.add_argument('save_model', type=int, help='save the model. Set 1 to save else 0')
parser.add_argument('save_model_tag', type=str,
                    help='tag with which to save the model or saved model tag for inference')
parser.add_argument('training_data_reference', type=str, help='tag for dataset to choose for training')
parser.add_argument('test_data_reference', type=str, help='tag for dataset to choose for training')
parser.add_argument('update_text_file', type=int, help='Update the text file', default=0)
parse = parser.parse_args()

sigmoid_coeff = 40
n_hid = 3
scale_factor = 1
mid_neurons = 50 #
parse.t_ox = torch.tensor(parse.t_ox * 1e-9, requires_grad=False).to(device).reshape((-1, 1))
parse.N_A = torch.tensor(parse.N_A * 1e24, requires_grad=False).to(device).reshape((-1, 1))
l_1 = 1
l_2 = 1

if parse.do_training and not parse.train_continue:
    print(f"Device being used is '{device}'")
    code_file = open('./poisson_solver.py')
    code = ""
    code = code.join(code_file.readlines())
    code_file.close()
    model_notes = str(f"1) Model name is {parse.save_model_tag} and batch size is {parse.batch_size}\n"
                 +"2) Model has second derivative loss\n"
                 +f"3) Model has 1 model with ({mid_neurons}, {n_hid}) , with Tanh() activation function; The Model is new. Please refer the code, the model is joint. Has scaling factor as {scale_factor}. Model as multiplication of vgs ahead of it.\n"
                 +f"4) Model is trained with learning rate {parse.lr} and has 0.96 for 1000 epochs\n"
                 +f"5) Model has the training dataset as {parse.training_data_reference} and has the test dataset as {parse.test_data_reference}\n"
                  +"6) Model was not introduced with psi_p\n"
               +"7) tsi was taken as 200nm\n"
                +"8) N_A wasn't constant\n"
                 +"9) Model was trained with data dependent statistics\n"
                  +"10) Model doesnot have L1 loss with 3 x torch.sqrt before.\n"
                      +"11) Model optimizer has no weight decay\n"
                       +"12) Model with variable tox\n"
                      + "13) Model has variable N_A\n"
                      + "14) Model has t_ox and N_A not as graph nodes\n"
                        +"15) Model has N_A in 0.1e24 to 1e24\n"
                      +"16) Model has no variable exponential\n"
                       +f"17) PINN model, with lambda_1 = {l_1} and \lambda_2 = {l_2}")
    if os.path.realpath(__file__).split('/')[-1] == f'{parse.save_model_tag}_code.py':
        sys.path.append('../')
    from make_note import make_notes_and_code
    make_notes_and_code(parse.save_model_tag, model_notes, code, parse.update_text_file, parse.train_continue)
else:
    if os.path.realpath(__file__).split('/')[-1] == f'{parse.save_model_tag}_code.py':
        sys.path.append('../')
    os.chdir(f'./{parse.save_model_tag}')

from stats_to_word import create_document
from Scripts.solve_bvp_any_fun import main

t_ox = 1e-9  # 1e-1
t_si = 20e-8  # 4e1
Vgs = parse.Vgs
do_training = parse.do_training
# train_with_only_ground_truth = parse.train_w_gt
n_epochs = 300000
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
# B = epsilon_sio2 / (t_ox * epsilon_si)
C = (n_i / N_A) ** 2
# mean1 = t_si/2
# std1 = t_si/(2*np.sqrt(3.0))
# mean2 = 0
# std2 = 1
a = (Cox / epsilon_si)

psi_P = psi_t*np.log(parse.N_A.item()/n_i.item())
Cox = epsilon_sio2 / parse.t_ox.item()
gamma = np.sqrt(2*q*epsilon_si*parse.N_A.item())/Cox
v_t = 2*psi_P + gamma*np.sqrt(2*psi_P)

print(
    f"tsi = {t_si}, and t_ox = {parse.t_ox}, and psi_t = {psi_t}, and A = {A}, and Vgs =  {Vgs} and psi_F  {psi_F} and a*tsi = {a * t_si}")

del A, a, N_A, psi_F, Cox, C

# class act_fun(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         act1 = nn.Tanh()
#         # act1 = nn.Sigmoid()
#         # return x*act1.forward(x)
#         return act1.forward(x)
#         # return x**3
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
            nn.Tanh(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden, dtype=torch.float64),
                # nn.BatchNorm1d(n_hidden, dtype=torch.float64),
                nn.Tanh(),
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
        self.net_y1 = NN1(4, 1, mid_neurons, n_hid) # (20, 5)

    def forward(self, y, vgs, tox, na, ret_first_derivative=False, ret_second_derivative=False, ret_first_derivative_vgs=False, return_pred_tsi=False, ret_second_derv_vgs=False, ret_zero=False):
        y_ = (y - mean1) / std1
        vgs_ = (vgs - mean2) / std2
        tox_ = (tox - mean3) / std3
        na_ = (na - mean4) / std4
        input = torch.cat((y_, vgs_, tox_, na_), axis=1)

        x =  self.net_y1.forward(input)

        # calculating the model value at zero
        y1 = torch.zeros(y.shape, dtype=torch.float64, requires_grad=True).to(device)
        y1_ = (y1 - mean1) / std1

        bc1 = torch.cat((y1_, vgs_, tox_, na_), axis=1)

        x1 =  self.net_y1.forward(bc1)
        x1_prime = torch.autograd.grad(x1, y1, torch.ones_like(x1), create_graph=True)[0]

        # calculating the model derivative value at t_si
        if return_pred_tsi:
            y1_ = t_si*torch.ones(y.shape, dtype=torch.float64, requires_grad=False).to(device)
            y1__ = (y1_ - mean1) / std1
            bc1_ = torch.cat((y1__, vgs_, tox_, na_), axis=1)

            x1_ =  self.net_y1.forward(bc1_).to(device)
            return x1_

        if ret_zero:
            return x1, x1_prime

        pred = x
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


def combine_L2_error():
    temp_N_AA = torch.linspace(5e23, 1e24, 6, dtype=torch.float64).to(device)
    temp_t_oox = t_oox = torch.linspace(0.8e-9, 1.5e-9, 8, dtype=torch.float64).to(device)
    error_list = []
    legend_list = []
    for doping in temp_N_AA:
        for tox in temp_t_oox:
            for voltage in array:
                voltage = round(voltage, 5)
                outputs, _ = main(VGS=voltage, Tox=tox.item(), NA=doping.item())
                outputs = outputs[0].reshape((test_samples, 1))
                temp_tox = tox * torch.ones_like(data_test_y).to(device)
                temp_N_A = doping * torch.ones_like(data_test_y).to(device)
                temp_vgs = voltage * torch.ones_like(data_test_y).to(device)
                data_test_y.requires_grad = True
                prediction = model.forward(data_test_y, temp_vgs, temp_tox, temp_N_A).detach().cpu().numpy()
                error_list.append(np.linalg.norm(outputs-prediction, 2)/(np.linalg.norm(outputs, 2) + 1e-25))
            plt.plot(array, np.array(error_list))
            error_list.clear()
            legend_list.append(f't_ox = {tox.item() :e}')
            print(f'Done for t_ox = {tox.item()}')
        plt.title(f'Relative L2 Error plot for N_A = {doping.item() :e}')
        plt.legend(legend_list, fontsize='5')
        legend_list.clear()
        plt.ylabel('Relative L2 Error')
        plt.xlabel('Vgs')
        plt.grid(True)
        plt.savefig(f'Error_plot_for_N_A={doping.item() :e}.png', dpi=250)
        plt.close()

    error_list = []
    legend_list = []
    test_surface_vgs__ = torch.from_numpy(array).reshape(-1, 1).to(device)
    for tox in temp_t_oox:
        for doping in temp_N_AA:
            test_surface_y__ = torch.zeros_like(test_surface_vgs__).to(device)
            temp_tox = tox * torch.ones_like(test_surface_vgs__).to(device)
            temp_N_A = doping * torch.ones_like(test_surface_vgs__).to(device)
            surface_prediction = model.forward(test_surface_y__, test_surface_vgs__, temp_tox, temp_N_A).detach().cpu().numpy()
            _, psi0 = main(True, psi0_samples=test_surface_vgs__.shape[0], VGS=parse.Vgs, Tox=tox.item(), NA=doping.item())
            error_list.append(np.linalg.norm(psi0 - surface_prediction, 2) / (np.linalg.norm(psi0, 2)))
        plt.plot(temp_N_AA.detach().cpu().numpy(), np.array(error_list))
        error_list.clear()
        legend_list.append(f't_ox = {tox.item() :e}')
        print(f'Done for t_ox = {tox.item()}')
    plt.title(f'L2 relative error for Surface Potential')
    plt.legend(legend_list, fontsize='5')
    legend_list.clear()
    plt.ylabel('Relative L2 Error')
    plt.xlabel('N_A')
    plt.grid(True)
    plt.savefig(f'Surface_Error_plot_L2_relative.png', dpi=250)
    plt.close()


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
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.96)
        train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
        if parse.train_continue:
            model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}.pth').state_dict())
            optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
            epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
            print(f"Loaded Saved Model")
        else:
            # initialise_weights(model)
            print(f"Started Training From Scratch")
        # loss_measure = nn.L1Loss(reduction='sum')
        loss = nn.MSELoss()
        # loss_gt = nn.MSELoss(reduction='sum')
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
                    y = y.reshape((-1, 1)).to(device)
                    vgs = vgs.reshape((-1, 1)).to(device)
                    y.requires_grad = True
                    pred, pred2 = model.forward(y, vgs,  t_oox[indices], N_AA[indices1], ret_second_derivative=True)
                    pred_t_si = model.forward(y, vgs,  t_oox[indices], N_AA[indices1], return_pred_tsi=True)
                    pred_zero, pred1_zero = model.forward(y, vgs,  t_oox[indices], N_AA[indices1], ret_zero=True)
                    y.requires_grad = False
                    with torch.no_grad():
                        A = q*N_AA[indices1]/epsilon_si
                        a = epsilon_sio2/(t_oox[indices] * epsilon_si)
                        psi_F = psi_t * torch.log(N_AA[indices1] / n_i)

                        check_loss += (loss(-(torch.exp(-pred/psi_t) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/psi_t) - 1)) * A, pred2).to(device) + (l_1)*loss(-a*(vgs - pred_zero), pred1_zero).to(device) + (l_2)*loss(pred_t_si, torch.zeros_like(pred_t_si).to(device))).item()
                model.requires_grad_(True)
            else:
                check_loss = np.sum(np.array(batch_loss_list))
                batch_loss_list.clear()
            loss_list.append(check_loss)

            if (max_loss > check_loss) and parser.parse_args().save_model:
                max_loss = check_loss
                torch.save(model, f'Poisson_model_{parse.save_model_tag}.pth')
                torch.save(optimizer, f'optimizer_mul_{parse.save_model_tag}.pth')
                torch.save(epoch_scheduler, f'epoch_scheduler_{parse.save_model_tag}.pth')
                print("Epoch: ", epoch)
                print(f"Model saved with sum batch loss log10 as {np.log10(max_loss)}")
                last_epoch_saved = epoch

            for index, (y, vgs) in enumerate(train_loader):
                y = y.reshape((-1, 1)).to(device)
                vgs = vgs.reshape((-1, 1)).to(device)
                y.requires_grad = True

                indices = np.round_(np.random.uniform(0, 28, size=(y.shape[0], 1)))
                indices1 = np.round_(np.random.uniform(0, 40, size=(y.shape[0], 1)))

                pred, pred2 = model.forward(y, vgs, t_oox[indices], N_AA[indices1], ret_second_derivative=True)
                pred_t_si = model.forward(y, vgs,  t_oox[indices], N_AA[indices1], return_pred_tsi=True)
                pred_zero, pred1_zero = model.forward(y, vgs,  t_oox[indices], N_AA[indices1], ret_zero=True)

                # y.requires_grad = False
                # pred1_tsi = model.forward(y, vgs, return_pred1_tsi=True)
                # reweight = torch.zeros_like(y, dtype=torch.float64, requires_grad=False)
                # reweight[vgs >= 0] = 1
                # reweight[vgs < 0] = 1e4

                # if torch.min(pred).item() >= -2 and torch.max(pred).item() <= 2: #Testing Condition
                # if epoch % 5 in [0, 1, 2]:
                # loss_ = (loss(torch.asinh((-(torch.exp(-pred/(scale_factor*psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(scale_factor*psi_t)) - 1))) * A), torch.asinh(pred2))).to(device)
                # A = q * 1e24 / epsilon_si #N_A here is 1e24
                # psi_F = psi_t * torch.log(1e24 / n_i).to(device)

                A = q * N_AA[indices1] / epsilon_si
                a = (epsilon_sio2 / t_oox[indices]) / epsilon_si
                psi_F = psi_t*torch.log(N_AA[indices1]/n_i)

                loss_ = (loss(-(torch.exp(-pred/psi_t) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/psi_t) - 1)) * A, pred2).to(device) + (l_1)*loss(-a*(vgs - pred_zero), pred1_zero).to(device) + (l_2)*loss(pred_t_si, torch.zeros_like(pred_t_si).to(device))).to(device)

                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_loss_list.append(loss_.item())

            avg_epoch_time += time.time() - start_time_epoch
            if epoch % 25 == 0:
                print(f'This is the epoch number {epoch} and this is the log10 training loss {np.log10(np.sum(np.array(batch_loss_list)))}')
                print("Last Epoch saved", last_epoch_saved, f" Learning Rate: ")
                # print(f"This is the model: {model.param}\n")
                for p in optimizer.param_groups: print(p['lr'])
                print(f"Average Time required for an epoch is {avg_epoch_time/25} Seconds")
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
    check_epoch = '86k'
    model = torch.load(f'Poisson_model_{parse.save_model_tag}.pth').to(device)
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
                                   if_print=False)  # romb needs 2^k + 1 points
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
        df.to_csv(f'Charge_predictions.csv')
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
        exit('\nExit at 633 after ending the integration process and plotting for eta0 and vt for the desired')
    # for i in model.parameters():
    #     print(i)
    # print(f"\n Sleeping for 20 seconds")
    # time.sleep(20)
    predp = True
    if parse.batch_size == 0:
        array = v_t * np.array([Vgs], dtype=np.float64)
    else:
        if Vgs > 0:
            array = np.arange(-Vgs, Vgs + 1e-12, 0.1)
        else:
            array = np.arange(Vgs, -Vgs + 1e-12, 0.1)

    array = v_t.item() * np.array([-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3])
    print(f"TAKING ONLY THOSE WHICH WE ARE INTERESTED IN")
    print(['$V_{gs}/V_T$=' + str(np.round_(i / v_t.item(), 5)) for i in array])

    color = np.random.uniform(0, 1, size=(2 * array.shape[0], 3))
    color = np.append(color, np.ones((color.shape[0], 1)), axis=1)
    color_list = list(color)
    # shuffle(color_list)
    y_test_list = []
    if os.path.exists(f'./{parse.save_model_tag}_outputs.docx'):
        os.remove(f'./{parse.save_model_tag}_outputs.docx')
    for index, Vgs in enumerate(array):
        Vgs = round(Vgs, 5)
        np.save(f'./{parse.save_model_tag}_vgs.npy', Vgs)
        psi0_required = True if (index + 1 == len(array) and parse.batch_size != 0) else False
        test_surface_vgs = torch.round(
            (torch.linspace(-parse.Vgs * v_t, parse.Vgs * v_t, 300, dtype=torch.float64)).reshape((-1, 1)),
            decimals=5).to(device)

        psi0_required = False
        print(
            "Skip on demand of not solving for the surface potential... may get error for surface potential as its not being calculated for saving time and keeping psi0_required=False")

        if psi0_required:
            vg = test_surface_vgs[-1].detach().cpu().numpy()
            # vg = parse.Vgs
        else:
            vg = None

        if not psi0_required:
            required_array, psi0_list = main(psi0_required, psi0_samples=300, VGS=vg)
        else:
            psi0_required = False
            required_array, psi0_list = main(psi0_required, psi0_samples=300, VGS=vg)

        train_plot = (np.linspace(0, t_si, test_samples, dtype=np.float64)).reshape((-1, 1))
        data_test_y = torch.linspace(0, t_si, test_samples, dtype=torch.float64).reshape((-1, 1)).to(device)
        data_test_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)
        temp2_y = torch.zeros((test_samples, 1), dtype=torch.float64, requires_grad=True).reshape((-1, 1)).to(device)
        temp2_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)

        vgs_width = (test_surface_vgs[1] - test_surface_vgs[0]).detach().cpu().numpy()
        # test_surface_vgs = torch.round((torch.linspace(-parse.Vgs, parse.Vgs, 300, dtype=torch.float64)).reshape((-1, 1)), decimals=5).to(device)
        test_surface_vgs.requires_grad = True

        test_surface_y = torch.zeros(test_surface_vgs.shape, dtype=torch.float64).to(device)

        surface_potential = model.forward(test_surface_y, test_surface_vgs,
                                          parse.t_ox * torch.ones_like(test_surface_y),
                                          parse.N_A * torch.ones_like(test_surface_y)).detach().cpu().numpy()
        # surface_potential /= scale_factor
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
            np.save(f'./{parse.save_model_tag}_vgs.npy', test_surface_vgs[-1] + vgs_width)  # This is with regard to 300
            psi0_list.append(main(need_psi0=True))
            surface_potential_derivative_true = list(
                np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0) / (
                    vgs_width))  # This is with regard to 300
            np.save(f'./{parse.save_model_tag}_vgs.npy',
                    test_surface_vgs[-1] + 2 * vgs_width)  # This is with regard to 300
            psi0_list.append(main(need_psi0=True))
            surface_potential_second_derivative_true = list(
                np.diff(np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0),
                        axis=0) / vgs_width ** 2)  # This is with regard to 300
            psi0_list.pop()
            psi0_list.pop()

        # pred_bc = model.forward(temp2_y, temp2_vgs, parse.t_ox * torch.ones_like(temp2_y)).to(device)
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
            # surface_potential_error = np.abs(np.array(psi0_list, dtype=np.float64) - np.array(psi0_list, dtype=np.float64))

        if parse.batch_size == 0:
            df = pd.DataFrame({'Vgs': (np.ones_like(prediction) * (array / v_t)).reshape(-1, ),
                               'Vgs/v_t': (np.ones_like(prediction) * array).reshape(-1, ),
                               'v_t': (np.ones_like(prediction) * v_t).reshape(-1, ),
                               'Prediction_NN': (prediction).reshape(-1, ), 'FEM Solution': (y_test).reshape(-1, ),
                               'Absolute Error': np.abs((prediction).reshape(-1, ) - (y_test).reshape(-1, )),
                               'y': data_test_y.detach().cpu().numpy().reshape((-1,))})
            df.to_csv(f'vgs={parse.Vgs}_{parse.save_model_tag}.csv')
            df = pd.DataFrame(
                {'vgs': test_surface_vgs.reshape(-1, ), 'predicted surface potential': surface_potential.reshape(-1, )})
            df.to_csv(f'surface_potential_only_predictions{parse.save_model_tag}.csv')
            plot_many_sp(model, v_t)
            print(
                f'Saved the predictions for vgs = {v_t * parse.Vgs}, equivalent to vgs/v_t = {parse.Vgs} as !!vgs={parse.Vgs}_{parse.save_model_tag}.csv!!')
            print(
                f'Saved the surface potential predictions for vgs in -{parse.Vgs}vt to {parse.Vgs}vt, and saved as !!surface_potential_only_predictions{parse.save_model_tag}.csv!!')
            fem_predictions, _ = main(y_new=np.linspace(0, t_si, 3000, dtype=np.float64), VGS=parse.Vgs * v_t.item(),
                                      NA=parse.N_A.item(), Tox=parse.t_ox.item(), if_print=False)
            predict(model, None, parse.Vgs * v_t, parse.t_ox, parse.N_A,
                    save_name=f'normv={parse.Vgs}_NA={parse.N_A.item() :e}_tox={parse.t_ox.item() :e}',
                    fem_predictions=fem_predictions[0], v_t=v_t.item(),
                    if_use_y=torch.linspace(0, t_si, 3000, dtype=torch.float64).reshape(-1, ).to(device))  # tox,na,vgs
            exit('Exit at line 724 after saving the predictions for the desired voltages in numpy format')

        if parse.batch_size == 2 and parse.batch_size != 0:
            if predp:
                predp_list = []
                predp_list.append(prediction)
                predp = False
            else:
                predp_list.append(prediction)
            # plt.scatter(train_plot, prediction)
            plt.plot(train_plot * 10 ** 9, y_test / psi_P, marker='o', markersize=5, markevery=15, mfc='none',
                     linewidth=0.0000001, c=color_list[index])
            # plt.plot(train_plot, y_test, linewidth=0.5, mfc='none', c=color_list[index] )
            if index == len(array) - 1:
                # plt.legend(list(array))
                plt.xlabel("y (nm)")
                plt.ylabel("$\Psi$ / $\Phi_B$")
                plt.legend(['$V_G/V_T$=' + str(np.round_(i / v_t.item(), 5)) for i in array])
                for ind, i in enumerate(predp_list):
                    plt.plot(train_plot * 10 ** 9, i / psi_P, linewidth=2, mfc='none', c=color_list[ind])
                    # plt.plot(train_plot, i, marker='o', markersize=3, markevery=3, mfc='none', linewidth=0.0000001, c=color_list[ind])
                plt.tight_layout()
                plt.xlim(-1, 80)
                plt.savefig('zoom_in_Poisson_Model_Prediction_vs_actual_all.png', dpi=200)
                plt.xlim(-1, 205)
                plt.text(140, -0.3, f'Epoch: 50975', fontsize=13)
                plt.text(25, 2.25, '$\lambda_1 = 1$, $\lambda_2 = 1$')
                plt.savefig('Poisson_Model_Prediction_vs_actual_all.png', dpi=200)
                plt.close()
                print('Saved all the plots in a single plot')
                for ind, i in enumerate(predp_list):
                    plt.semilogy(train_plot * 10 ** 9, np.abs(i - y_test_list[ind]), c=color_list[ind], mfc='none')
                plt.xlabel('y (nm)')
                plt.ylabel('$|\Psi_{Numerical} - \Psi_{ML, PINN}|$')
                plt.xlim(-1)
                plt.tight_layout()
                plt.savefig('Absolute_error_plot.png', dpi=200)
                plt.close()
            else:
                continue
        plt.plot(train_plot, prediction)
        plt.plot(train_plot, y_test)
        plt.title(f"Model prediction vs Actual for Vgs = {Vgs}")
        plt.legend(['Actual_prediction', 'Ground_Truth'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction_vs_actual.png")
        plt.close()

        plt.plot(train_plot, prediction_derivative)
        plt.plot(train_plot, y_test_derivative)
        plt.legend(['Actual_prediction_derivative', 'Ground_Truth_derivative'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction_vs_actual_derivative.png")
        plt.close()

        plt.plot(train_plot, prediction_second_derivative)
        plt.plot(train_plot, y_test_second_derivative)
        plt.legend(['Actual_prediction_second_derivative', 'Ground_Truth_second_derivative'])
        plt.savefig("Poisson_Model_Prediction_vs_actual_second_derivative.png")
        plt.close()

        plt.plot(y_test, train_plot)
        plt.grid(True)
        plt.savefig("Poisson_actual.png")
        plt.close()

        plt.plot(train_plot, y_test)
        plt.grid(True)
        plt.savefig("Poisson_Model_Ground_Truth.png")
        plt.close()

        plt.plot(train_plot, prediction)
        plt.grid(True)
        plt.savefig("Poisson_Model_Prediction.png")
        plt.close()

        plt.plot(train_plot, test_loss)
        plt.title(f'log10 Absolute_error_Plot_for_Vgs = {Vgs}')
        plt.legend(['Loss Curve'])
        plt.grid(True)
        plt.savefig("Poisson_Model_Loss.png")
        plt.close()
        create_document(parse.save_model_tag,
                        [f'./Poisson_Model_Prediction_vs_actual.png', f'./Poisson_Model_Loss.png'])

        plt.plot(train_plot, test_loss_derivative)
        plt.title('Absolute_error_Plot_derivative')
        plt.grid(True)
        plt.legend(['Loss Curve'])
        plt.savefig("Poisson_Model_Loss_derivative.png")
        plt.close()

        # training_error = np.load(f'{parse.save_model_tag}_training_loss.npy')
        # plt.plot(range(training_error.size), np.log10(training_error))
        # plt.title('log10 Training Error plot')
        # plt.grid(True)
        # plt.legend(['log10 Training Error'])
        # plt.savefig("log10_Training_Error.png")
        # plt.close()

        if index == len(array) - 1 and parse.batch_size != 0:
            plt.plot(test_surface_vgs / v_t.item(), surface_potential)
            plt.plot(test_surface_vgs / v_t.item(), psi0_list)
            # plt.plot(test_surface_vgs, psi0_list, marker='o', linewidth=0.0000001, markersize=3, markevery=3,
            #          mfc='none')
            plt.title("Surface Potential")
            plt.xlabel("Vgs/Vt")
            plt.ylabel("psi(0)")
            plt.legend(["Predicted Surface Potential", "Actual Surface Potential"])
            plt.grid(True)
            plt.savefig("Surface_Potential.png")
            plt.close()

            plt.plot(test_surface_vgs / v_t.item(), np.log10(surface_potential_error))
            plt.title("log10 Surface Potential absolute error")
            plt.xlabel('$V_{gs}/V_T$')
            plt.grid(True)
            plt.savefig("Surface_Potential_absolute_error.png")
            plt.close()

            plt.plot(test_surface_vgs / v_t.item(), surface_potential_derivative)
            plt.plot(test_surface_vgs / v_t.item(), surface_potential_derivative_true)
            # plt.plot(test_surface_vgs, surface_potential_derivative_true, marker='o', linewidth=0.0000001, markersize=3,
            #          markevery=3, mfc='none')
            plt.title("Surface Potential Derivative")
            plt.xlabel("Vgs/Vt")
            plt.ylabel("$d\Psi(0)/dV_{gs}$")
            plt.legend(["Predicted Surface Potential Derivative", "Actual Surface Potential Derivative"])
            plt.grid(True)
            plt.savefig("Surface_Potential_derivative.png")
            plt.close()

            pd.DataFrame({'Vgs': np.array(test_surface_vgs).reshape(-1, ),
                          'Vg/v_t': (test_surface_vgs / v_t.item()).reshape(-1, ),
                          'SP_derv_ML': np.array(surface_potential_derivative).reshape(-1, ),
                          'SP_derv_FEM': np.array(surface_potential_derivative_true).reshape(-1, ),
                          'Psi0_FEM': np.array(psi0_list).reshape(-1, ),
                          'Psi0_NN': np.array(surface_potential).reshape(-1, )}).to_csv(f'SP_derv.csv')
            print('Saved the Predictions and actual values of surface potential derivatives in SP_derv.csv')

            plt.plot(test_surface_vgs, surface_potential_second_derivative)
            plt.plot(test_surface_vgs, surface_potential_second_derivative_true)
            plt.grid(True)
            # plt.plot(test_surface_vgs, surface_potential_second_derivative_true, marker='o', linewidth=0.0000001,
            #          markersize=3, markevery=3, mfc='none')
            plt.title("Surface Potential Second Derivative")
            plt.xlabel("Vgs")
            plt.ylabel("$d^{2}\Psi(0)/{d{V^2}_{gs}}$")
            plt.legend(["Predicted Surface Potential Second Derivative", "Actual Surface Potential Second Derivative"])
            plt.savefig("Surface_Potential_second_derivative.png")
            plt.close()
            create_document(parse.save_model_tag,
                            [f'./Surface_Potential.png', f'./Surface_Potential_absolute_error.png',
                             f'Surface_Potential_derivative.png', f'Surface_Potential_second_derivative.png'])

    print(f"Time Required to Execute the Training is {time.time() - start_time} Seconds")