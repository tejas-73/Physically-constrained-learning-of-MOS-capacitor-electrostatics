import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available else 'cpu'

epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0 * 11.9
q = 1.6e-19
epsilon_sio2 = epsilon_0 * 3.9
n_i = torch.tensor(1e16, requires_grad=False)
psi_t = 26e-3


def calc_vt(model, psi_b, N_A, t_ox):
    step = 1e-4 #can set to convenience
    y = torch.zeros((1, 1), dtype=torch.float64).to(device)
    vgs = 2 * psi_b * torch.ones((1, 1), dtype=torch.float64).to(device)
    pred_psi = model(y.reshape(-1, 1), vgs.reshape(-1, 1),  t_ox.reshape(-1, 1), N_A.reshape(-1, 1),)
    while not torch.isclose(pred_psi, 2*psi_b, atol=1e-4):
        if pred_psi > 2*psi_b:
            vgs -= step
        elif pred_psi < 2*psi_b:
            vgs += step
        pred_psi = model(y.reshape(-1, 1), vgs.reshape(-1, 1), t_ox.reshape(-1, 1), N_A.reshape(-1, 1))
    return vgs #This is v_T at \psi(0, vgs) = 2*psi_b


def vt_vs_tox(model):
    vt_pred_list = []
    vt_anly_list = []
    legend_list = []
    t_oox = 1e-9*torch.linspace(0.5, 2.1, 17, dtype=torch.float64).reshape(-1, 1).to(device) #0.8 - 1.5
    N_AA = 1e24*torch.tensor([0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float64).to(device) #0.1 - 1
    df = {}
    df['t_ox'] = t_oox.detach().cpu().numpy().reshape(-1, )
    for na in tqdm(N_AA):
        for tox in tqdm(t_oox):
            Cox = epsilon_sio2 / tox
            gamma = torch.sqrt(2 * q * epsilon_si * na) / Cox
            psi_b = psi_t * torch.log(na / n_i).to(device)
            v_t = 2 * psi_b + gamma * torch.sqrt(2 * psi_b) #actual vt
            vt_pred = calc_vt(model, psi_b, na, tox) #estimated vt
            vt_pred_list.append(vt_pred.item())
            vt_anly_list.append(v_t.item())
            print(f'Done for tox = {tox.item()} and got threshold voltages as = {v_t.item(), vt_pred.item()}, analytical and predicted')
        df[f'v_t_prediction_for_N_A={na.item() :e}'] = np.array(vt_pred_list).reshape(-1, )
        df[f'v_t_anly_for_N_A={na.item() :e}'] = np.array(vt_anly_list).reshape(-1, )
        print(f'Done for N_A = {na.item() :e}')
        plt.plot(t_oox.cpu().numpy(), vt_pred_list)
        plt.plot(t_oox.cpu().numpy(), vt_anly_list)
        vt_pred_list.clear()
        vt_anly_list.clear()
        legend_list.append(f'predicted vt for N_A = {na.item() :e}')
        legend_list.append(f'Analytic vt for N_A = {na.item() :e}')
    plt.grid(True)
    plt.xlabel('t_ox')
    plt.ylabel('V_T')
    plt.legend(legend_list)
    plt.title('V_T vs t_ox plot for different N_A')
    plt.savefig('vt_vs_tox.png')
    plt.close()
    df = pd.DataFrame(df)
    df.to_csv('V_t_vs_t_ox_for_NA.csv')
    print('\nSaved a csv file with different predictions for v_t vs t_ox for varying values of N_A as !V_t_vs_t_ox_for_NA.csv!')