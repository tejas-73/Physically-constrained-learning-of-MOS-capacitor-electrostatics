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
                                                 'If batch size = 0, then predictions for a given Vgs is calculated. And also it plots the surface potential characteristics for V_G in [-3Vt, 3Vt]. Saves a csv file of the predictions. It also makes surface potential predictions with different device parameters.\n' +
                                                 'If batch size = 2, then saves a combine plot for $V_G$ in [-3Vt, 3Vt] and surface potential as well\n' +
                                                 'If batch size other than above, then a document is created in which we can scrutinizingly observe the profile predictions.')
parser.add_argument('lr', type=float, help='learning rate of the adam optimizer')
parser.add_argument('do_training', type=int, help='Set 1 to perform Training, else setting this to zero will do the task of inference.')
parser.add_argument('train_continue', type=int, help='Set 1 to train from the previously stored model else 0. This is for the case in which the preloaded model has to start training again')
parser.add_argument('save_model', type=int, help='save the model during training, this is for crosscheking. set 1 to save else 0.' +
                                                 'during inference, we use this as the epoch number at which we need the outputs')
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
mid_neurons = 60 #
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
               +"6) tsi was taken as 200nm\n"
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


t_ox = 1e-9
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

class act_fun(nn.Module): #creating an activation function
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
        self.net_y1 = NN1(4, 1, mid_neurons, n_hid)

    def forward(self, y, vgs, tox, na, ret_first_derivative=False, ret_second_derivative=False, ret_first_derivative_vgs=False, return_pred1_tsi=False, ret_second_derv_vgs=False):
        # Standardizing the inputs
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

        #calculating the derivative of neural network as y=0
        x1_prime = torch.autograd.grad(x1, y1, torch.ones_like(x1), create_graph=True)[0]

        C_ox = epsilon_sio2/tox
        a = C_ox/epsilon_si

        B = (a*t_si*torch.exp(x1))/(1 + a*t_si + t_si*x1_prime) #This is equivalent to $\lambda$ in the paper


        pred = vgs*B*(1 - y/t_si)*torch.exp(-x) #This is the actual prediction for $\Psi$ as presented in the paper

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

    data_f_test = pd.read_csv(f'../Data/data_y_100_100_58_vg_100_50_100_{parse.test_data_reference}.csv')
    y_test = data_f_test['y'].values.astype(np.float64)
    vgs_test = data_f_test['Vgs'].values.astype(np.float64)
    psi_test = data_f_test['psi'].values.astype(np.float64)
    test = torch.tensor(y_test, dtype=torch.float64)
    train_target_test = torch.tensor(vgs_test, dtype=torch.float64)
    train_target1_test = torch.tensor(psi_test, dtype=torch.float64)
    del y_test, vgs_test, psi_test, data_f_test

    #loading the data and calculating standardizing coefficients
    data_f = pd.read_csv(f'../Data/data_y_100_100_58_vg_100_50_100_{parse.training_data_reference}.csv')
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

    # Range of values of t_{ox}
    t_oox = torch.linspace(0.8e-9, 1.5e-9, 29, dtype=torch.float64).to(device)
    mean3 = torch.mean(t_oox).to(device)
    std3 = torch.std(t_oox).to(device)

    # Range of values of N_{A}
    N_AA = torch.linspace(1e23, 1e24, 41, dtype=torch.float64).to(device)
    mean4 = torch.mean(N_AA).to(device)
    std4 = torch.std(N_AA).to(device)

    if do_training:
        model = NN(2, 1, 20, 4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.96) #1000, 0.96
        train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
        if parse.train_continue:
            continue_epoch = 0 #We consider to restart training from this epoch. Better to keep the most recent saved model
            print(f'Continuing the model training from epoch: {continue_epoch}. Make sure that thhis is where you wish to start.\n WIll wait for 10 seconds to check')
            time.sleep(10)
            model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_{continue_epoch}.pth').state_dict())
            optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
            epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
            print(f"Loaded Saved Model")
        else:
            print(f"Started Training From Scratch")
            continue_epoch = 0
        loss = nn.L1Loss(reduction='sum')
        epoch_list = []
        loss_list = []
        batch_loss_list = []
        max_loss = np.inf #for the case that we save th model with what ever loss it is initially
        last_epoch_saved = 0
        len_of_loader = train_loader.__len__()
        print(f"These are the number of datapoints: {len_of_loader * parse.batch_size}")
        avg_epoch_time = 0
        nans_encountered = 0
        for epoch in range(continue_epoch, n_epochs):
            start_time_epoch = time.time()
            epoch_scheduler.step()
            check_loss = 0
            model.train()
            if epoch == continue_epoch: #this is for the case when we restart training, then we have a base model which we need a better model to
                model.requires_grad_(False)
                for index, (y, vgs) in enumerate(train_loader):
                    indices = np.round_(np.random.uniform(0, 28, size=(y.shape[0], 1))) #sampling random indices for selecting t_{ox}
                    indices1 = np.round_(np.random.uniform(0, 40, size=(y.shape[0], 1))) #sampling random indices for selecting N_A
                    y = y.to(device)
                    vgs = vgs.to(device)
                    y.requires_grad = True
                    pred, pred2 = model.forward(y.reshape((-1, 1)), vgs.reshape((-1, 1)), t_oox[indices].reshape((-1, 1)), N_AA[indices1].reshape((-1, 1)), ret_second_derivative=True)
                    psi_F = psi_t * torch.log(N_AA[indices1] / n_i).to(device)
                    A = q * N_AA[indices1] / epsilon_si
                    y.requires_grad = False
                    with torch.no_grad():
                        check_loss += (((torch.pow(((((loss((-(torch.exp(-pred/(scale_factor*psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(scale_factor*psi_t)) - 1))) * A, (pred2)))))), 1)))).item()
                model.requires_grad_(True)
            else:
                check_loss = np.sum(np.array(batch_loss_list))
                batch_loss_list.clear()
            loss_list.append(check_loss)
            epoch_list.append(epoch)

            if (max_loss > check_loss) and parse.save_model: #criterion to save the model
                max_loss = check_loss
                torch.save(model, f'Poisson_model_{parse.save_model_tag}_epoch_{epoch}.pth')
                torch.save(optimizer, f'optimizer_mul_{parse.save_model_tag}.pth')
                torch.save(epoch_scheduler, f'epoch_scheduler_{parse.save_model_tag}.pth')
                print("Epoch: ", epoch)
                print(f"Model saved with sum batch loss log10 as {np.log10(max_loss)}")
                last_epoch_saved = epoch
            elif math.isnan(check_loss): #handling nans generated in a way that we reload the previous trained model and leanr with 90% of the previous learning rate
                nans_encountered += 1
                model.load_state_dict(torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_{last_epoch_saved}.pth').state_dict())
                optimizer.load_state_dict(torch.load(f'optimizer_mul_{parse.save_model_tag}.pth').state_dict())
                epoch_scheduler.load_state_dict(torch.load(f'epoch_scheduler_{parse.save_model_tag}.pth').state_dict())
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9 #decreasing the learning rate to 90%
                loss_list.pop()
            print(f'Number of nans encountered: {nans_encountered}')
            np.save(f'{parse.save_model_tag}_training_loss.npy', np.array(loss_list)) #this is to observe how the loss is decreasing in subsequent epochs


            for index, (y, vgs) in enumerate(train_loader):
                y = y.reshape((-1, 1)).to(device)
                vgs = vgs.reshape((-1, 1)).to(device)
                y.requires_grad = True

                #shuffling the contents of the required sets
                # shuffle(t_oox)
                # shuffle(N_AA)

                #selecting random indices to select the required elements from sets
                indices = np.random.choice(28, size=y.shape, replace=True)
                indices1 = np.random.choice(40, size=y.shape, replace=True)

                pred, pred2 = model.forward(y, vgs, t_oox[indices], N_AA[indices1], ret_second_derivative=True) #getting the desired to optimize the loss function

                A = q * N_AA[indices1] / epsilon_si #calculating all the terms dependent on $N_A$
                psi_F = psi_t * torch.log(N_AA[indices1] / n_i).to(device)

                loss_ = ((torch.log((((((loss((-(torch.exp(-pred/(psi_t)) - 1 - torch.exp(-2*psi_F/psi_t) * (torch.exp(pred/(psi_t)) - 1))) * A, (pred2)))))))))).to(device)

                loss_.backward()

                optimizer.step()
                optimizer.zero_grad()

                batch_loss_list.append(np.exp(loss_.item()))

            avg_epoch_time += time.time() - start_time_epoch #collecting the time required per epoch
            print(f'The maximum batch loss is: {np.max(np.array(batch_loss_list)) :e} and the minimum batch loss is: {np.min(np.array(batch_loss_list)) :e} and the total loss is {np.exp(loss_.item()) :e}')

            if epoch % 25 == 0 or epoch < 25: #printing neccessary contents in during training
                print(f'This is the epoch number {epoch} and this is the log10 training loss {np.log10(np.sum(np.array(batch_loss_list)))}')
                print("Last Epoch saved", last_epoch_saved, f" Learning Rate: ")
                for p in optimizer.param_groups: print(p['lr'])
                print(f"Average Time required for an epoch is {avg_epoch_time if epoch < 25 else avg_epoch_time/25} Seconds")
                avg_epoch_time = 0
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
    try:
        model = torch.load(f'Poisson_model_{parse.save_model_tag}_epoch_{check_epoch}.pth').to(device)
    except:
        raise Exception(f'No model saved with epoch number {check_epoch}')
    print(f'These are all the arguments: {parse}')
    if parse.batch_size == -1: #these are used when we need charge, $\eta_0$ and $V_T$ characteristics
        y = torch.ones((1, 1), dtype=torch.float64).to(device) * 200e-9  # in nm
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
                                      parse.N_A * torch.ones_like(y_prime, dtype=torch.float64)).detach().cpu().numpy() #model predictions
            int_val_list.append(
                integrate(model, y, vgs, parse.t_ox, parse.N_A.detach().cpu().numpy().astype(np.float64).reshape(-1, ),
                          discrete=predictions_model.reshape((-1,)).astype(np.float64))) #saving the integration values
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
        plt.legend(['From ML model', 'From FEM', 'Anly'])
        plt.grid(True)
        plt.title('Q vs Vg/Vt')
        plt.xlabel('Vg/Vt')
        plt.ylabel('Q')
        plt.savefig('QvsVg_Vt.png') #saving plot for Q vs V_G/V_T
        plt.close()

        plt.semilogy(volt_list, (np.abs(np.array(int_val_list))).reshape(-1, 1))
        plt.semilogy(volt_list, (np.abs(np.array(fem_val_list))).reshape(-1, 1))
        plt.semilogy(volt_list, np.abs(np.array(anly_list)).reshape(-1, 1))
        plt.legend(['From ML model', 'From FEM'])
        plt.grid(True)
        plt.title('Q vs Vg/Vt')
        plt.xlabel('Vg/Vt')
        plt.ylabel('log10(|Q|)')
        plt.savefig('QvsVg_Vt_log10.png') #saving plot for Q vs V_G/V_T
        plt.close()

        print(
            f'\nPredicted value of eta0 = {calc_eta0(np.log10(np.abs(np.array(int_val_list))).reshape(-1, 1), np.array(volt_list))} and Analytical value = {1 + (gamma / (2 * np.sqrt(2 * psi_P)))}')
        calc_multiple_eta0(model, main)
        vt_vs_tox(model)
        #exiting so that the model is not further executed.
        exit('\nExit at line 433 after ending the integration process and plotting for eta0 and vt for the desired. Plotted \n a) |Q| vs $V_G$ plot. \n b) $\eta_0$ vs $t_{ox}$ plot. \n c) $V_T$ vs $t_{ox}$ plot.')
    predp = True
    if parse.batch_size == 0:
        array = v_t * np.array([Vgs], dtype=np.float64)
    else:
        array = v_t.item() * np.array([-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3]) #taking values of those voltages that we are interested in

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

        if psi0_required:
            vg = test_surface_vgs[-1].detach().cpu().numpy()
        else:
            vg = None

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
            #exiting so that further code is not executed
            exit('Exit at line 543 after saving the predictions for the desired voltages in numpy format')

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
            plt.plot(test_surface_vgs / v_t.item(), np.arrray(psi0_list)/psi_F.item())
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
