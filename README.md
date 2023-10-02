# Physically constrained learning of MOS capacitor electrostatics
In recent years, neural networks have achieved phenome-
nal success across a wide range of applications. Neural net-
works have also proven useful for solving differential equa-
tions. The focus of this work is on the Poisson-Boltzmann
equation (PBE) that governs the electrostatics of a metal-
oxide-semiconductor (MOS) capacitor. We were motivated
by the question of whether a neural network can effectively
learn the solution of PBE using the methodology pioneered
by Lagaris et al. (IEEE Trans. Neural Networks 9 (1998)).
In this method, a neural network is used to generate a set
of trial solutions that adhere to the boundary conditions,
which are then optimized using the governing equation.
However, the challenge with this method is the lack of a
generic procedure for creating trial solutions for intricate
differential equations. In this work, we present an innova-
tive technique for constructing trial solutions tailored to
the highly nonlinear PBE while satisfying the Robin and
Dirichlet boundary conditions derived from MOS device
physics. Remarkably, by training the network parameters,
we can compute an optimal trial solution that accurately
captures essential physical insights, such as the depletion
width, threshold voltage, and inversion charge. Further-
more, we show that our functional solution can extend
beyond the training domain.


-> Getting acquinted to the main scripts:

a) **_A_ODE_all_tox_NA.py_**: is the script to run model optimization as presented in the paper. 

b) **_PINN.py_**: is the model formulation with PINN approach

Remaining scripts are being utilized by these two scripts.

-> Folders:

a) **_model_checkpoint_**: contains all the models saved at different epochs in the optimization process.

b) **_model_final_**: contains the final model that has been used to generate the results presented in the paper.

c) **_model_PINN_actual_**: contains all the PINN model at different epochs in the optimization process and can be used to generate different results. The results here are corresponding to $\lambda_1 = 10^{16}$ and $\lambda_2 = 10^{34}$

d) **_model_PINN_actual_1_**: contains all the PINN model at different epochs in the optimization process and can be used to generate different results. The results here are corresponding to $\lambda_1 = 1$ and $\lambda_2 = 1$

e) **_Scripts_**: contains following scripts:

**_generate_data.py_**: is used to generate data in a csv file and save it inot **_Data_** folder. The model is being trained by the data stored in the csv file present in the Data folder. Note that during training, we need to mention the csv file that contains the relevant data.

**_solve_bvp_any_fun.py_**: This is used to generate the spline method based solution to the PBE. This is being called by the main scripts.

**_Surface_potential.py_**: This script is used to generate the surface potential from SPE.

f) **_Figures_**: Contains all the figures presented in the paper.

g) **_Data_**: This folder contains the data in a csv file. The _generate_data.py_ script will put the data in here. Also the main scripts will extract data from this folder.



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

