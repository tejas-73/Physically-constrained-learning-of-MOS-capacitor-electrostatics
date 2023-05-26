'''

Helps in generating the collocation points or domain sampling and saves it into a csv file. We can later extract from this csv file the complete dataset.
The one that is used in the paper can be generated using this script by keeping the only command line argument as 1. The csv file also contanins the numerical method predictions

This script also plots the potential profile and surface potential for a given N_A and t_{ox}.

'''


import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib as mpl
import argparse


background = '#D7E5E5'
mpl.rcParams['font.family']= 'sans-serif'
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['legend.title_fontsize'] = 8
mpl.rcParams['savefig.facecolor']= 'white'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelweight'] = 'heavy'

parser = argparse.ArgumentParser()
parser.add_argument('save_csv_file', type=float, help='Keep 1 to save the csv file else 0')
parse = parser.parse_args()

file_name = 'v_t_swise_200nm_3.6M'

N_A = 0.1e24
t_ox = 1.5e-9
t_si = 200e-9
epsilon_0 = 8.85418781e-12
epsilon_si = epsilon_0*11.9
epsilon_sio2 = epsilon_0*3.9
delta_psi_MS = 0.21
psi_t = 26e-3
n_i = 1e16
psi_P = psi_t*np.log(N_A/n_i)
q = 1.6e-19
Cox = epsilon_sio2 / t_ox
gamma = np.sqrt(2*q*epsilon_si*N_A)/Cox
v_t = 2*psi_P + gamma*np.sqrt(2*psi_P)

print(f"This is v_t = {v_t}")


def fun(y, psi):
    A = q*N_A/epsilon_si
    first = psi[1]
    second = -A*(np.exp(-psi[0]/psi_t) - 1 - np.exp(-2*psi_P/psi_t)*(np.exp(psi[0]/psi_t) - 1))
    return np.array([first, second])


def bc(psi_a, psi_b):
    Cox = epsilon_sio2/t_ox
    B = Cox/epsilon_si
    first = +psi_a[1] + B*(Vg - psi_a[0])
    second = psi_b[0]
    return np.array([first, second])

y = np.concatenate([np.linspace(0, 10e-9, 500, endpoint=False), np.linspace(10e-9, 50e-9, 1000, endpoint=False), np.linspace(50e-9, 130e-9, 1000, endpoint=False), np.linspace(130e-9, t_si, 500)]) #stepwise sampling

psi = np.zeros((2, y.size))

fig, ax = plt.subplots(figsize=(12, 10))
fig.set_dpi(200)
if not parse.save_csv_file:
    V_g = v_t * np.array([-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3])
else:
    V_g = np.linspace(-3*v_t, 3*v_t, 1200)
psi_out = []
for i in V_g:
    Vg = i
    print(f"Solved for V_G = {i}")
    sol = solve_bvp(fun, bc, y, psi, tol=1e-6, max_nodes=20000)
    plt.plot(y, (sol.sol(y)[0]), label='$V_{G}$=' + '%.2f'%i + ' V')
    psi_out.append(list(sol.sol(y)[0]))
plt.xlabel("Vertical Distance, y(nm)", fontsize=20)
plt.ylabel("Potential, $\Psi(y)$    (V)", fontsize=20)
plt.legend(['$V_{G}$=' + str(i) + ' V' for i in list(V_g)])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ticks = np.round_(ax.get_xticks()[1:]*10**9)
ax.set_xticklabels(ticks)
plt.xlim(0)
plt.savefig('Ground_Truth.png')
plt.close()

psi_out = np.array(psi_out)
psi_out = psi_out.flatten()
V_g_1 = []
loss_weight = []
for i in V_g:
    V_g_1 += [i]*len(list(y))
    loss_weight.append(weight_to_loss)

df = {'y':list(y)*len(list(V_g)), 'V_g':V_g_1, 'psi': list(psi_out)}
df = pd.DataFrame(df)

df_b_0 = df[df['y'] == 0]
df_b_1 = df[df['y'] == 2e-7]

if parse.save_csv_file:
    df.to_csv(f'../Data/data_y_100_100_58_vgs_100_50_100_{file_name}.csv', header=True, index=False)
    df_b_0.to_csv('../Data/data_bc1_vgs_100_50_100.csv', header=True, index=False)
    df_b_1.to_csv('../Data/data_bc2_vgs_100_50_100.csv', header=True, index=False)

yy = np.linspace(0, t_si, 1000)
psii = np.zeros((2, yy.size))
V_g = np.linspace(-5, 5, 50)
psi_zero = []
psi_zero_pred = []
for i in V_g:
    Vg = i
    sol = solve_bvp(fun, bc, yy, psii, tol=1e-3, max_nodes=20000)
    psi_zero.append(sol.sol(yy)[0][0])


fig, ax = plt.subplots(figsize=(12, 8))
fig.set_dpi(200)
ax.axvline(-0.6, ls='--', lw=1, c='#000000')
ax.axvline(1.2, ls='--', lw=1, c='#000000')
plt.plot(V_g/v_t, np.array(psi_zero)/psi_P)
plt.xlabel("$V_{G}/V_T$", fontsize=20)
plt.ylabel("$\Psi_s/Psi_B$", fontsize=20)
plt.xticks([i for i in range(-5, 6)], fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('Surface_Potential_Plot.png')
