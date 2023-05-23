'''

Plots the surface potential predictions for different set of device parameters.

'''


import numpy as np
import torch
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_many_sp(model, voltage, num_samples_for_voltage=500):
    vgs = torch.linspace(-3*voltage, 3*voltage, num_samples_for_voltage, dtype=torch.float64).reshape(-1, 1).to(device) #mostly voltage = v_t and hence plotting in between +- 3V_T
    y = torch.zeros_like(vgs).to(device)
    t_ox = 1e-9*torch.linspace(0.9, 1.5, 3, dtype=torch.float64).reshape(-1, 1).to(device) #set of values of t_ox considered
    N_A = 1e24*torch.linspace(0.4, 1, 4, dtype=torch.float64).reshape(-1, 1).to(device) #set of values of N_A considered
    legend_list = []
    for t in t_ox:
        for n in N_A:
            sp = model(y, vgs, t*torch.ones_like(y).reshape(-1, 1), n*torch.ones_like(y).reshape(-1, 1)).detach().cpu().numpy()
            plt.plot(vgs.detach().cpu().numpy(), sp)
            legend_list.append(f'NA={n.item() :e}, tox={t.item() :e}')
    plt.grid(True)
    plt.legend(legend_list)
    plt.savefig('variation_sp.png') #saving all the predictions for different parameters of predicted surface potential.
    plt.close()
    print(f'Plotted many surface potential plots in a single plot, from plot_many_sp.py, and plot as variation_sp.py')