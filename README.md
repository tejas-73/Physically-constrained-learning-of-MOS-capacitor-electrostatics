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


Getting acquinted to the scripts:

_A_ODE_all_tox_NA.py_ is the one that has the model presented in the paper. It takes the following arguments:

