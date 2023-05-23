import numpy as np
import torch
import math
import scipy.integrate as integral
from vt_vs_tox import calc_vt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_t = 26e-3
n_i = 1e16
q = 1.6e-19
t_si = 200e-9
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0 * 11.9
epsilon_sio2 = epsilon_0 * 3.9

# def taylor_expansion(model, vgs, t_ox, N_A, tol=1000): #gives a vector such that integration can be easily performed with it.
#     for param in model.parameters():
#         param.requires_grad = False
#     y = torch.zeros((1, 1), dtype=torch.float64).to(device)
#     y.requires_grad = True
#     pred = model(y, vgs, t_ox, N_A) #\psi(y)
#     pred = torch.exp(pred/psi_t) * 1e-9 #pred = $e^{\frac{\psi(y)}{\psi_t}} * 1E-9$
#     value = 0
#     taylor_list = []
#     taylor_list.append(pred.item())
#     counter = 1
#     while np.abs(pred.item()) > tol:
#         counter += 1
#         pred = (pred * (1e-9))/counter
#         pred = torch.autograd.grad(pred, y, torch.ones_like(pred), create_graph=True)[0]
#         taylor_list.append(pred.item())
#         print(f'This is the pred value = {pred.item()}')
#     return np.array(taylor_list)
#
# def integrate(model, y, vgs, t_ox, N_A, tol=1000):
#     vector = taylor_expansion(model, vgs, t_ox, N_A, tol=tol)
#     y = y * 1e9 #converting to the form considered
#     integration_value = 0
#     for i in range(1, vector.size+ 1):
#         integration_value += (y**i)*(vector[i-1])
#     return integration_value, vector

def get_vgs():
    Vgs = np.load(f'vgs.npy')
    return Vgs

def get_t_ox():
    t_ox = np.load(f't_ox.npy')
    return t_ox

def get_N_A():
    N_A = np.load(f'N_A.npy')
    return N_A

def get_model():
    model = torch.load(f'model.pth')
    return model

def func(y):
    model = get_model().to(device)
    y = torch.tensor(y, dtype=torch.float64).reshape(-1, 1).to(device)
    vgs = torch.tensor(get_vgs(), dtype=torch.float64).to(device)*torch.ones_like(y).to(device)
    t_ox = torch.tensor(get_t_ox(), dtype=torch.float64).to(device)*torch.ones_like(y).to(device)
    N_A = torch.tensor(get_N_A(), dtype=torch.float64).to(device)*torch.ones_like(y).to(device)
    psi = model(y, vgs, t_ox, N_A).detach().cpu().numpy()
    n_0 = n_i**2/get_N_A()

    Q = -n_0 * np.exp(psi/psi_t) * q #returning the value of inversion charge per unit area
    return Q

def integrate(model, y, vgs, t_ox, N_A, discrete=None, function=func):
    if discrete is None:
        vgs = vgs.detach().cpu().numpy()
        t_ox = t_ox.detach().cpu().numpy()
        N_A = N_A.detach().cpu().numpy()
        torch.save(model, f'model.pth')
        np.save(f'vgs.npy', vgs) #
        np.save(f't_ox.npy', t_ox)
        np.save(f'N_A.npy', N_A)
        Q = integral.romberg(function, 0, y.detach().cpu().numpy())
    else:
        np.save(f'N_A.npy', N_A)
        n_0 = n_i ** 2 / get_N_A()
        Q = integral.trapz(-n_0 * np.exp(discrete/psi_t) * q, dx=(200*1e-9)/8000) #romb needs 2^k + 1
    return Q

def calc_eta0(logQ, vgs):
    indices = np.argwhere(vgs > 0.4)
    indices = np.intersect1d(indices, np.argwhere(vgs < 0.8))
    indices = indices[:2]
    delta_logQ = logQ[indices[1]] - logQ[indices[0]]
    delta_vgs = vgs[indices[1]] - vgs[indices[0]]
    delta_vgs /= delta_logQ
    inv_derv = delta_vgs
    n_o = inv_derv/(2.3*psi_t)
    return n_o

def calc_multiple_eta0(model, main):
    y_prime = torch.linspace(0, t_si, 8000, dtype=torch.float64).reshape(-1, 1).to(device)
    t_ox = 1e-9*torch.linspace(0.5, 2.1, 17, dtype=torch.float64).reshape(-1, 1).to(device) #0.8 - 1.5
    # N_A = 1e24*torch.linspace(0.08, 1.2, 12, dtype=torch.float64).reshape(-1, 1).to(device) #0.1 - 1
    N_A = 1e24*torch.tensor([0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float64).to(device)
    volt_array = (torch.linspace(0, 3.6, 20, dtype=torch.float64)).reshape((-1, 1)).to(device) #Assuming that 3*v_t doesnot cross 4
    legend_list = []
    volt_list = []
    int_val_list = []
    fem_val_list = []
    t_ox_eta0_anly_list = []
    t_ox_eta0_fem_list = []
    t_ox_eta0_nn_list = []
    df = {}
    df['t_ox'] = t_ox.detach().cpu().numpy().reshape(-1, )
    for n in tqdm(N_A):
        for t in tqdm(t_ox):
            for vgs in volt_array:
                volt_list.append(vgs.item())  # plotting against vg/v_t
                fem_solution, _ = main(y_new=np.linspace(0, t_si, 8000, dtype=np.float64), VGS=vgs.item(), NA=n.item(), Tox=t.item(), if_print=False)  # romb needs 2^k + 1 points
                predictions_model = model(y_prime, vgs * torch.ones_like(y_prime, dtype=torch.float64), t * torch.ones_like(y_prime, dtype=torch.float64), n * torch.ones_like(y_prime, dtype=torch.float64)).detach().cpu().numpy()
                int_val_list.append(integrate(model, y_prime, vgs, t.item(), n.item(), discrete=predictions_model.reshape((-1,)).astype(np.float64)))
                fem_val_list.append(integrate(model, y_prime, vgs, t.item(), n.item(), discrete=fem_solution[0].astype(np.float64)))
            t_ox_eta0_fem_list.append(calc_eta0(np.log10(np.abs(np.array(fem_val_list))), np.array(volt_array.detach().cpu().numpy())))
            t_ox_eta0_nn_list.append(calc_eta0(np.log10(np.abs(np.array(int_val_list))), np.array(volt_array.detach().cpu().numpy())))

            psi_P = psi_t * np.log(n.item() / n_i)
            Cox = epsilon_sio2 / t.item()
            gamma = np.sqrt(2 * q * epsilon_si * n.item()) / Cox
            t_ox_eta0_anly_list.append((1 + (gamma / (2 * np.sqrt(2 * psi_P)))))

            int_val_list.clear()
            fem_val_list.clear()
        df[f'eta0_fem_NA_{n.item() :e}'] = np.array(t_ox_eta0_fem_list).reshape(-1, )
        df[f'eta0_nn_NA_{n.item() :e}'] = np.array(t_ox_eta0_nn_list).reshape(-1, )
        df[f'eta0_anly_NA_{n.item() :e}'] = np.array(t_ox_eta0_anly_list).reshape(-1, )
        legend_list.append(f'NA={n.item() :e}_FEM')
        legend_list.append(f'NA={n.item() :e}_NN')
        legend_list.append(f'NA={n.item() :e}_anly')
        plt.plot(t_ox.detach().cpu().numpy().reshape(-1, ), np.array(t_ox_eta0_fem_list).reshape(-1, ))
        plt.plot(t_ox.detach().cpu().numpy().reshape(-1, ), np.array(t_ox_eta0_nn_list).reshape(-1, ))
        plt.plot(t_ox.detach().cpu().numpy().reshape(-1, ), np.array(t_ox_eta0_anly_list).reshape(-1, ))
        t_ox_eta0_fem_list.clear()
        t_ox_eta0_nn_list.clear()
        t_ox_eta0_anly_list.clear()
    df = pd.DataFrame(df)
    df.to_csv('eta0_t_ox_NA.csv')
    plt.grid(True)
    plt.xlabel('t_ox')
    plt.ylabel('eta0')
    plt.legend(legend_list)
    plt.title('eta0 vs t_ox plot for different N_A')
    plt.savefig('eta0_vs_tox.png')
    plt.close()
    print('\nSaved a csv file with different predictions for eta0 vs t_ox for varying values of N_A as !eta0_t_ox_NA.csv!')
    
