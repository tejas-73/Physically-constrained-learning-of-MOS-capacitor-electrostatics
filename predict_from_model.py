'''

This script is used to predict from the model for particular values of y, vgs, t_ox and N_A. In case you give an array of y, then ot predicts for that array.

'''


import numpy as np
import torch
import argparse
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(model, y, vgs, t_ox, N_A, fem_predictions, v_t, save_name, psi_b, if_use_y=None):
    if not (if_use_y is None): #incase ifdifferent y is required.
        y = if_use_y.reshape(-1, 1)
    predictions = model(y, vgs*torch.ones_like(y).to(device), t_ox*torch.ones_like(y).to(device), N_A*torch.ones_like(y).to(device)).detach().cpu().numpy()
    np.save(f'{save_name}.npy', predictions) #saving predictions from the model, in numpy format.
    np.save(f'FEM_{save_name}.npy', fem_predictions) #saving fem predictions in  umpy format.
    y_new = y.detach().cpu().numpy().reshape(-1, )
    predictions_FEM = fem_predictions
    df = {'y': y_new, 'v_gs': (vgs*torch.ones_like(y)).detach().cpu().numpy().reshape(-1, ), 'v_gs/vt': (vgs*torch.ones_like(y)/v_t).detach().cpu().numpy().reshape(-1, ), 'psi_b': (psi_b*torch.ones_like(y).detach().cpu().numpy().reshape(-1, ))
        , 't_ox': (t_ox*torch.ones_like(y)).detach().cpu().numpy().reshape(-1, ), 'N_A': (N_A*torch.ones_like(y)).detach().cpu().numpy().reshape(-1, ), 'Predictions_ML': predictions.reshape(-1, ), 'Predictions_FEM': predictions_FEM.reshape(-1, )}
    pd.DataFrame(df).to_csv(f'{save_name}.csv') #saving all the predictions in an excel file, with name mentioned as save_name
    print(f'Saved the numpy file with name {save_name}.npy for model predictions and FEM_{save_name}.npy for fem predictions and also the predictions to {save_name}.csv')
    print(f'Saved the numpy file with name {save_name}.npy for model predictions and FEM_{save_name}.npy for fem predictions and also the predictions to {save_name}.csv')
