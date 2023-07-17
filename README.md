# Physically constrained learning of MOS capacitor electrostatics
Recent years have witnessed the unprecedented success of neural network models for diverse applications, from image and speech recognition to solving differential equations. The focus of this work is on the latter applications. In particular, we were motivated by the question: can simple feedforward neural networks learn the physical principles of a MOS (metal-oxide-semiconductor) capacitor, without using labeled data? We proceeded by solving the governing Poisson-Boltzmann equation using PINNs (Physics Informed Neural Networks) which have shown much promise. We optimized the PINN model over gate voltage, oxide thickness, and doping concentration, which all together exacerbate the complexity offered by the equation. The important finding is that the model accuracy can be significantly improved by enforcing exact boundary conditions and using log-L1 loss. We demonstrate that the proposed model can accurately capture critical insights like the depletion width, threshold voltage, inversion charge, etc. We also show that the network can extrapolate beyond the sampling domain.


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

