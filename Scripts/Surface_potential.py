'''
Generates analytic solution for surface potential, given some device parameters in the command line

'''


import scipy.integrate as spy
from scipy.optimize import newton,bisect
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
# from pytictoc import TicToc
# T = TicToc()
from decimal import *
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Variables')
parser.add_argument('y_samples', type=int, help='number of y samples')
parser.add_argument('vgs_samples', type=int, help='number of vgs samples')
parser.add_argument('N_A', type=float, help='in multiples of 1e24')
parser.add_argument('t_ox', type=float, help='in multiples of 1e-9')
parse = parser.parse_args()
setcontext(ExtendedContext)

q = 1.6e-19
v_T = 0.026 #0.0259
N_a_0 = 1e18 # doping concentration per cm^3
N_a = parse.N_A*N_a_0*1e6  # per m^3
n_0 = 1e32/N_a # for Silicon only #n_i^2/N_a
ep0 = 8.85418781e-12   # in F/m
epox = 3.9*ep0
epsi = 11.9*ep0
tox = parse.t_ox*1e-9       # in m
Cox = epox/tox   # in F/m^2
ymax = 200e-9    # in m

n = np.sqrt(2*q*epsi*N_a)/Cox
G = n/np.sqrt(v_T)
phif_2 = 2*v_T*np.log(N_a/1e16) #n_i = 1.5e16
gamma = np.sqrt(2*q*epsi*N_a)/Cox
v_t = phif_2 + gamma*np.sqrt(phif_2)

print(f'This is v_t {v_t}')

Vg = np.linspace(-4.5*v_t, 4.5*v_t, parse.vgs_samples)
phi_s = np.zeros(len(Vg))
X_s = np.zeros(len(Vg))
##-----------------------------------------------------MOSCAP----------------------------------------------------##
for i in range(0, len(Vg), 1):
    ##-----------------------------Solution of Poissons's Equation---------------------------##
    def fun(X, Y):
        return np.vstack([Y[1], (-q / epsi) * ((N_a * np.exp(-Y[0] / v_T) - (n_0 * np.exp(Y[0] / v_T)) - N_a))])

    def bc(ya, yb):
        return np.array([(-Cox / epsi) * (Vg[i] - ya[0]) - ya[1], yb[0]])

    X = np.linspace(0, ymax, parse.y_samples)    # defining initial mesh ## array type
    Y = np.zeros((2, X.size))         # initial guess for y

    result = spy.solve_bvp(fun, bc, X, Y, max_nodes=10000)
    Y_plot = result.sol(X)[0]

    if Vg[i] >= 0: ## as it is n-channel
        phi_s[i] = max(Y_plot)
    else:
        phi_s[i] = min(Y_plot)        
    ##-----------------------------Solution of Poissons's Equation---------------------------##

    def func(Xs):
        H = (v_T*np.exp(-Xs/v_T)) + Xs - v_T + (np.exp(-phif_2/v_T)*((v_T*np.exp(Xs/v_T)) - Xs - v_T))
        func_1 = pow((Vg[i]-Xs)/n, 2) - H
        return (func_1)
    
    if Vg[i]<0:
        int_a = Vg[i]
        int_b = 0
    elif Vg[i]==0:
        int_a = -1
        int_b = 0
    else:
        int_a = 0
        int_b = Vg[i]

    X_s[i] = bisect(func, int_a, int_b)
plt.plot(Vg, X_s)
plt.scatter(Vg, phi_s)
plt.xlabel('Vgs')
plt.ylabel('Surface_potential')
plt.title('Surface Potential Plot')
plt.legend(['Analytical solution', 'numerical solution'])
plt.grid(True)
plt.savefig('Surface_potential_anyltic_numerical.png')
plt.close()

plt.plot(Vg, abs(phi_s - X_s))
plt.xlabel('Vgs')
plt.ylabel('Absolute Error')
plt.title(f'Error plot for numerical and analytical solution for y samples = {parse.y_samples}')
plt.grid(True)
plt.savefig('Surface_potential_error_anyltic_numerical.png')
plt.close()

df = pd.DataFrame({'Vgs': Vg, 'sp_numerical': phi_s, 'sp_analytic': X_s, 'V_T': v_t*np.ones_like(X_s), 'psi_p': phif_2*np.ones_like(X_s)})
df.to_csv(f'sp_comparison_{N_a:e}_{tox:e}.csv')
print(f'Saved the predictions for surface potential by analytic method and fem in -> sp_comparison_{N_a:e}_{tox:e}.csv')
print(f'Saved the comparision plot from analytic solution and numerical method as -> Surface_potential_anyltic_numerical.png')
print(f'Saved the error plot between analytic solution and numerical method as -> Surface_potential_error_anyltic_numerical.png')