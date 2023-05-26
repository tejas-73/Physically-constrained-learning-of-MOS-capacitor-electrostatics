# Self-supervised learning of physical principles of metal oxide semiconductor capacitor
Recent years have seen growing interest in using machine
learning (ML) to solve differential equations. Such efforts,
so far, have mainly been focused on computational aspects
and comprehending the physical principles of a system has
received very little attention. Here, we investigate whether
an unsupervised (labelled data free) ML model can accurately replicate the fundamental physics of a metal-oxide-
semiconductor (MOS) capacitor, which is governed by the
Poisson-Boltzmann equation (PBE). The highly dynamic
nature of the PBE coupled with it’s unique physics-based
boundary conditions pose challenges in solving the problem using ML. However, by using a parametric model that
naturally satisfies the boundary conditions, the expressive
power of neural networks can be harnessed to yield excellent agreement with solutions obtained from traditional
numerical approaches. In addition the proposed model
not only captures the inception of accumulation, depletion, and inversion regions of a MOS capacitor, but it
also unravels the dependence of threshold voltage on oxide
thickness and doping concentration. Extrapolation ability of the model further confirms that model has indeed
learn the physical mechanism of the MOS capacitor rather
memorizing the training results.


-> Getting acquinted to the main scripts:

a) **_A_ODE_all_tox_NA.py_**: is the script to run model optimization as presented in the paper. 

b) **_PINN.py_**: is the model formulation with PINN approach

Remaining scripts are being utilized by these two scripts.

-> Folders:

a) **_model_checkpoint_**: contains all the models saved at different epochs in the optimization process and can be used to generate different results.

b) **_model_PINN_actual_**: contains all the PINN model at different epochs in the optimization process and can be used to generate different results. The results here are corresponding to $\lambda_1 = 10^{16}$ and $\lambda_2 = 10^{34}$

c) **_model_PINN_actual_1_**: contains all the PINN model at different epochs in the optimization process and can be used to generate different results. The results here are corresponding to $\lambda_1 = 1$ and $\lambda_2 = 1$

d) **_Scripts_**: contains following scripts:

**_generate_data.py_**: is used to generate data in a csv file and save it inot **_Data_** folder. The model is being trained by the data stored in the csv file present in the Data folder. Note that during training, we need to mention the csv file that contains the relevant data.

**_solve_bvp_any_fun.py_**: This is used to generate the spline method based solution to the PBE. This is being called by the main scripts.

**_Surface_potential.py_**: This script is used to generate the surface potential from SPE.

e) **_Figures_**: Contains all the figures presented in the paper.

f) **_Data_**: This folder contains the data in a csv file. The _generate_data.py_ script will put the data in here. Also the main scripts will extract data from this folder.



<!-- It takes the following arguments:

**Vgs**: type=float, Value of Vgs. Used during inference

**t_ox**: type=float, tox value in nm. Used during inference

**N_A**: type=float, N_A value as a coefficient to 1e24. Used during inference

**train_samples**: type=int, Number of Training Samples. This is used during inference, when we need to sample for y. This number is the number of datapoints of y, uniformly sampled

**batch_size**: type=int, Batch Size during training. During inference, this has a crucial role to play. If batch size = -1, then Inversion Charge characteristics, $V_T$ characteristics and $\eta_0$ characteristics are calculated. If batch size = 0, then predictions for a given Vgs is calculated. And also it plots the surface potential characteristics for V_G in [-3Vt, 3Vt]. Saves a csv file of the predictions. It also makes surface potential predictions with different device parameters. If batch size = 2, then saves a combine plot for $V_G$ in [-3Vt, 3Vt] and surface potential as well. If batch size other than above, then a document is created in which we can scrutinizingly observe the profile predictions.

**lr**: type=float, learning rate of the adam optimizer

**do_training**: type=int, Set 1 to perform Training, else setting this to zero will do the task of inference

**train_continue**: type=int, Set 1 to train from the previously stored model else 0. This is for the case in which the preloaded model has to start training again

**save_model**: type=int, save the model during training, this is for crosscheking. set 1 to save else 0. During inference, we use this as the epoch number at which we need the outputs

**save_model_tag**: type=str, tag with which to save the model or saved model tag for inference. A folder will be created with this name and all the files and models will be stored in this. Also, this is the name that will be used, during inference.

**training_data_reference**: type=str, tag for dataset to choose for training

**test_data_reference**: type=str, tag for dataset to choose for training. Ensure this to be same as training_data_reference

**update_text_file**: type=int, Update the text file. Ensure this to be zero. This is when in case mistakenly you type a save_model_tag to be the one which already exists. In case if you wish to update, then keep 1. -->

