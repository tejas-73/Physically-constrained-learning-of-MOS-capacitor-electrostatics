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
mid_neurons = 50

if parse.do_training and not parse.train_continue:
    print(f"Device being used is '{device}'")
    code_file = open('./no_device_params.py')
    code = ""
    code = code.join(code_file.readlines())
    code_file.close()
    model_notes = str(f"1) Model name is {parse.save_model_tag} and batch size is {parse.batch_size}\n"
                 +"2) Model has second derivative loss\n"
                 +f"3) Model has 1 model with ({mid_neurons}, {n_hid}) , with Tanh() activation function; The Model is new. Please refer the code, the model is joint. Has scaling factor as {scale_factor}. Model as multiplication of vgs ahead of it.\n"
                 +f"4) Model is trained with learning rate {parse.lr} and has 0.96 for 1000 epochs\n"
                 +f"5) Model has the training dataset as {parse.training_data_reference} and has the test dataset as {parse.test_data_reference}\n"
                  +"6) Model was not introduced with psi_p\n"
               +"7) tsi was taken as 50nm\n"
                +"8) N_A was taken as 1e24\n"
                 +"9) Model was trained with data dependent statistics\n"
                  +"10) Model doesnot have L1 loss with 3 x torch.sqrt before. And taking the $sinh^{-1}$ of the components in the loss  \n"
                      +"11) Model optimizer has no weight decay\n")
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

t_ox = 1.5e-9 #1.5nm
t_si = 200e-9
Vgs = parse.Vgs
do_training = parse.do_training
n_epochs = 150000
train_samples = parse.train_samples
test_samples = train_samples
lr = parse.lr
psi_t = 26e-3
batch_size = parse.batch_size
k = 1e8

N_A = torch.tensor(1e23, requires_grad=False) #10^{23}
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0 * 11.9
epsilon_sio2 = epsilon_0 * 3.9
delta_psi_MS = 0.21
n_i = torch.tensor(1e16, requires_grad=False)
psi_F = psi_t * torch.log(N_A / n_i).to(device)
q = 1.6e-19
Cox = epsilon_sio2 / t_ox
A = q * N_A / epsilon_si
B = epsilon_sio2 / (t_ox * epsilon_si)
C = (n_i / N_A) ** 2
a = (Cox / epsilon_si)
gamma = torch.sqrt(2*q*epsilon_si*N_A)/Cox
v_t = 2*psi_F + gamma*torch.sqrt(2*psi_F)

print(
    f"tsi = {t_si}, and t_ox = {t_ox}, and psi_t = {psi_t}, and A = {A}, and Vgs =  {Vgs} and psi_F  {psi_F} and a*tsi = {a * t_si} and v_t = {v_t.item()}")


class act_fun(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        act1 = nn.Tanh()
        return act1.forward(x)


class NN1(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.fcs = nn.Sequential(*[
            nn.Linear(n_input, n_hidden, dtype=torch.float64),
            act_fun(),
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden, dtype=torch.float64),
                act_fun(),
            ]) for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output, dtype=torch.float64)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class NN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.net_y1 = NN1(2, 1, mid_neurons, n_hid)

    def forward(self, y, vgs, ret_first_derivative=False, ret_second_derivative=False, ret_first_derivative_vgs=False, return_pred1_tsi=False, ret_second_derv_vgs=False):
        y_ = (y - mean1) / std1
        vgs_ = (vgs - mean2) / std2

        input = torch.cat((y_, vgs_), axis=1)

        x =  self.net_y1.forward(input)

        # calculating the model value at zero
        y1 = torch.zeros(y.shape, dtype=torch.float64, requires_grad=True).to(device)
        y1_ = (y1 - mean1) / std1

        bc1 = torch.cat((y1_, vgs_), axis=1)

        x1 =  self.net_y1.forward(bc1)
        x1_prime = torch.autograd.grad(x1, y1, torch.ones_like(x1), create_graph=True)[0]
        B = (a*t_si*torch.exp(x1))/(1 + a*t_si + t_si*x1_prime)
        pred = vgs*B*(1-y/t_si)*torch.exp(-x)
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

    if do_training:
        model = NN(2, 1, mid_neurons, n_hid).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.96)
        train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
        if parse.train_continue:
            model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}.pth').state_dict())
            optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
            epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
            print(f"Loaded Saved Model")
        else:
            print(f"Started Training From Scratch")
        loss = nn.L1Loss(reduction='sum')
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
        for epoch in range(n_epochs):
            start_time_epoch = time.time()
            epoch_scheduler.step()
            check_loss = 0
            model.train()
            if epoch == 0:
                model.requires_grad_(False)
                for index, (y, vgs) in enumerate(train_loader):
                    y = y.to(device)
                    vgs = vgs.to(device)
                    y.requires_grad = True
                    pred, pred2 = model.forward(y.reshape((-1, 1)), vgs.reshape((-1, 1)),
                                          ret_second_derivative=True)
                    y.requires_grad = False
                    with torch.no_grad():
                        check_loss += (((torch.pow(((((loss((-(torch.exp(-pred/(scale_factor*psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(scale_factor*psi_t)) - 1)))*A, pred2))))), 1)))).item()
                model.requires_grad_(True)
            else:
                check_loss = np.sum(np.array(batch_loss_list))
                batch_loss_list.clear()
            np.save(f'{parse.save_model_tag}_training_loss.npy', np.array(loss_list))
            if (max_loss > check_loss) and parser.parse_args().save_model:
                max_loss = check_loss
                torch.save(model, f'Poisson_model_{parse.save_model_tag}.pth')
                torch.save(optimizer, f'optimizer_mul_{parse.save_model_tag}.pth')
                torch.save(epoch_scheduler, f'epoch_scheduler_{parse.save_model_tag}.pth')
                print("Epoch: ", epoch)
                print(f"Model saved with sum batch loss log10 as {np.log10(max_loss)}")
                last_epoch_saved = epoch
            elif math.isnan(check_loss): #handling nans generated in a way that we reload the previous trained model and leanr with 90% of the previous learning rate
                nans_encountered += 1
                model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}.pth').state_dict())
                optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
                epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9 #decreasing the learning rate to 90%
                loss_list.pop()
            print(f'Number of nans encountered: {nans_encountered}')

            for index, (y, vgs) in enumerate(train_loader):
                y = y.reshape((-1, 1)).to(device)
                vgs = vgs.reshape((-1, 1)).to(device)
                y.requires_grad = True
                vgs.requires_grad = False
                pred, pred2 = model.forward(y, vgs, ret_second_derivative=True)
                y.requires_grad = False
                loss_ = torch.log(loss((-(torch.exp(-pred / psi_t) - 1 - torch.exp(-2 * psi_F / psi_t) * (torch.exp(pred / psi_t) - 1))) * A, pred2)).to(device)
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss_list.append(np.exp(loss_.item()))
            if epoch % 25 == 0 or epoch < 25:
                print(f'This is the epoch number {epoch} and this is the log10 training loss {np.log10(np.sum(np.array(batch_loss_list)))}')
                print("Last Epoch saved", last_epoch_saved, f" Learning Rate: ")
                for p in optimizer.param_groups: print(p['lr'])
                print(f"Time required for an epoch is {time.time() - start_time_epoch} Seconds")

        plt.title("Training Plot for Ground Truth")
        plt.plot(np.array(epoch_list), np.array(loss_list))
        plt.legend(['First', 'BC1', 'BC2'])
        plt.savefig('Model_Training_Error.png')
        plt.close()
    background = '#D7E5E5'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 6
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['legend.title_fontsize'] = 6
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['legend.fontsize'] = 4
    mpl.rcParams['axes.labelweight'] = 'heavy'
    torch.set_printoptions(threshold=torch.inf, precision=20)
    model = torch.load(f'Poisson_model_{parse.save_model_tag}.pth').to(device)
    predp = True
    if parse.batch_size == 0:
        array = np.array([Vgs], dtype=np.float64)
    else:
        if Vgs > 0:
            array = np.arange(-Vgs, Vgs + 1e-12, 0.1)
        else:
            array = np.arange(Vgs, -Vgs + 1e-12, 0.1)

    color = np.random.uniform(0, 1, size=(2 * array.shape[0], 3))
    color = np.append(color, np.ones((color.shape[0], 1)), axis=1)
    color_list = list(color)
    shuffle(color_list)
    if os.path.exists(f'./{parse.save_model_tag}_outputs.docx'):
        os.remove(f'./{parse.save_model_tag}_outputs.docx')
    for index, Vgs in enumerate(array):
        Vgs = round(Vgs, 5)
        np.save(f'./{parse.save_model_tag}_vgs.npy', Vgs)
        psi0_required = True if index + 1 == len(array) else False
        required_array, psi0_list = main(psi0_required)
        train_plot = (np.linspace(0, t_si, test_samples, dtype=np.float64)).reshape((-1, 1))
        data_test_y = torch.linspace(0, t_si, test_samples, dtype=torch.float64).reshape((-1, 1)).to(device)
        data_test_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)
        temp2_y = torch.zeros((test_samples, 1), dtype=torch.float64, requires_grad=True).reshape((-1, 1)).to(device)
        temp2_vgs = Vgs * torch.ones((test_samples, 1), dtype=torch.float64).reshape((-1, 1)).to(device)

        test_surface_vgs = (torch.linspace(-Vgs, Vgs, 500, dtype=torch.float64)).reshape((-1, 1)).to(device)
        test_surface_vgs.requires_grad = True

        test_surface_y = torch.zeros(test_surface_vgs.shape, dtype=torch.float64).to(device)

        surface_potential = model.forward(test_surface_y, test_surface_vgs).detach().cpu().numpy()
        surface_potential /= scale_factor
        surface_potential_derivative = model.forward(test_surface_y, test_surface_vgs,
                                                     ret_first_derivative_vgs=True).detach().cpu().numpy()
        surface_potential_second_derivative = model.forward(test_surface_y, test_surface_vgs,
                                                            ret_second_derv_vgs=True).detach().cpu().numpy()
        test_surface_vgs = test_surface_vgs.detach().cpu().numpy()

        if psi0_required:
            np.save(f'./{parse.save_model_tag}_vgs.npy', Vgs + Vgs / 250)  # This is with regard to 500
            psi0_list.append(main(need_psi0=True))
            surface_potential_derivative_true = list(
                np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0) / (
                            Vgs / 250))  # This is with regard to 500
            np.save(f'./{parse.save_model_tag}_vgs.npy', Vgs + Vgs / 125)  # This is with regard to 500
            psi0_list.append(main(need_psi0=True))
            surface_potential_second_derivative_true = list(
                np.diff(np.diff(np.array(psi0_list, dtype=np.float64).reshape((-1, 1)), axis=0), axis=0) / (
                            Vgs / 250) ** 2)  # This is with regard to 500
            psi0_list.pop()
            psi0_list.pop()

        pred_bc = model.forward(temp2_y, temp2_vgs).to(device)
        data_test_y.requires_grad = True
        prediction = model.forward(data_test_y, data_test_vgs).to(device)
        prediction_derivative = model.forward(data_test_y, data_test_vgs, ret_first_derivative=True).to(device)
        _, prediction_second_derivative = model.forward(data_test_y, data_test_vgs, ret_second_derivative=True)

        prediction = prediction.detach().cpu().numpy()
        prediction /= scale_factor
        prediction_derivative = prediction_derivative.detach().cpu().numpy()
        prediction_second_derivative = prediction_second_derivative.detach().cpu().numpy()
        y_test = required_array[0].reshape((test_samples, 1))
        y_test_second_derivative = -A * (
                    np.exp(-y_test / psi_t) - 1 - np.exp(-2 * psi_F.item() / psi_t) * (np.exp(y_test / psi_t) - 1))
        y_test_derivative = required_array[1].reshape((test_samples, 1))
        test_loss = np.abs(prediction - y_test)
        test_loss_derivative = np.abs(prediction_derivative - y_test_derivative)

        if psi0_required:
            surface_potential_error = np.abs(surface_potential - np.array(psi0_list, dtype=np.float64))

        if parse.batch_size == 2:
            if predp:
                predp_list = []
                predp_list.append(prediction)
                predp = False
            else:
                predp_list.append(prediction)
            plt.plot(train_plot * 10 ** 9, y_test, marker='o', markersize=2, markevery=15, mfc='none',
                     linewidth=0.0000001, c=color_list[index])
            if index == len(array) - 1:
                plt.xlabel("Vertical Distance, y(nm)")
                plt.ylabel("Potential, $\psi(y)$ (V)")
                plt.xlim(0)
                plt.legend(['$V_{gs}$=' + str(i) + ' V' for i in np.round(array, 5)])
                for ind, i in enumerate(predp_list):
                    plt.plot(train_plot * 10 ** 9, i, linewidth=0.5, mfc='none', c=color_list[ind])
                    # plt.plot(train_plot, i, marker='o', markersize=3, markevery=3, mfc='none', linewidth=0.0000001, c=color_list[ind])
                plt.savefig('Poisson_Model_Prediction_vs_actual_all.png', dpi=200)
                plt.close()
                print('Saved all the plots in a single plot')
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

        plt.plot(train_plot, np.log10(test_loss))
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

        if index == len(array) - 1:
            plt.plot(test_surface_vgs, surface_potential)
            plt.plot(test_surface_vgs, psi0_list)
            # plt.plot(test_surface_vgs, psi0_list, marker='o', linewidth=0.0000001, markersize=3, markevery=3,
            #          mfc='none')
            plt.title("Surface Potential")
            plt.xlabel("Vgs")
            plt.ylabel("psi(0)")
            plt.legend(["Predicted Surface Potential", "Actual Surface Potential"])
            plt.grid(True)
            plt.savefig("Surface_Potential.png")
            plt.close()

            plt.plot(test_surface_vgs, np.log10(surface_potential_error))
            plt.title("log10 Surface Potential absolute error")
            plt.xlabel('$V_{gs}$')
            plt.grid(True)
            plt.savefig("Surface_Potential_absolute_error.png")
            plt.close()

            plt.plot(test_surface_vgs, surface_potential_derivative)
            plt.plot(test_surface_vgs, surface_potential_derivative_true)
            # plt.plot(test_surface_vgs, surface_potential_derivative_true, marker='o', linewidth=0.0000001, markersize=3,
            #          markevery=3, mfc='none')
            plt.title("Surface Potential Derivative")
            plt.xlabel("Vgs")
            plt.ylabel("$d\Psi(0)/dV_{gs}$")
            plt.legend(["Predicted Surface Potential Derivative", "Actual Surface Potential Derivative"])
            plt.grid(True)
            plt.savefig("Surface_Potential_derivative.png")
            plt.close()

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