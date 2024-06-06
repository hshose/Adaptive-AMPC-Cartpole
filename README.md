# Cartpole Example for Parameter-Adaptive Approximate MPC
Comprehensive pipeline for synthesizing a large dataset from a **nonlinear model-predictive controller (MPC)** with parametric sensitivities, **training a neural network approximation** of this dataset, and deploying that **neural network controller on an embedded microcontroller (STM32G4)** to control the swing-up and stabilization of a cartpole pendulum system.

## Paper
This code is supplementary material to the paper:
[Hose, Henrik, Alexander Gräfe, and Sebastian Trimpe. "Parameter-Adaptive Approximate MPC: Tuning Neural-Network Controllers without Retraining." accepted to the 6th Annual Learning for Dynamics & Control Conference L4DC (2024).](https://arxiv.org/abs/2404.05835)

Please cite our paper as:
```
@inproceedings{hose2024parameter,
  title={Parameter-Adaptive Approximate MPC: Tuning Neural-Network Controllers without Retraining},
  author={Hose, Henrik and Gr{\"a}fe, Alexander and Trimpe, Sebastian},
  booktitle={Learning for Dynamics and Control Conference},
  year={2024},
  organization={PMLR}
}
```

## Video of Experiments
A video of our experiments conducted with this code is available here:

[![Video of cartpole pendulum stabilized by parameter-adaptive approximate MPC](https://img.youtube.com/vi/o1RdiYUH9uY/0.jpg)](https://www.youtube.com/watch?v=o1RdiYUH9uY)

## Structure of this Repo
The code inside this repo is structured in the three main steps of distilling a neural network approximation from a model predictive controller (MPC):
1. **A large dataset is computed** by repeatedly solving the MPC optimization problem for random initial conditions, for which the code is provided in the [dataset_synthesis folder](dataset_synthesis). We use [Pyomo](https://github.com/Pyomo/pyomo) to formulate the optimization problem and [sIPOPT](https://link.springer.com/content/pdf/10.1007/s12532-012-0043-2.pdf) for solving it. See the respective [README.md](dataset_synthesis/README.md) for details. You can also skip this step by downloading the dataset that we used for our experiments [from Zenodo, here (3.5Gb)](https://zenodo.org/records/11093569).
2. **A neural network is trained** to approximate an explicit controller for the dataset. We use [Jax](https://github.com/google/jax) to train two neural networks, one for the optimal action sequence and one for the sensitivities. See the [README.md](neural_network_training/README.md) for details. You can skip this step and use the neural network approximation [of the optimal actions here](neural_network_training/models/model_8x50/) and the [sensitivities here](neural_network_training/models/model_8x50/) that we trained already.
3. **The parameter-adaptive approximate MPC** is deployed on an STM32G474 microcontroller to control a cartpole inverted pendulum. We use a [Nucleo-G474RE board](https://www.st.com/en/evaluation-tools/nucleo-g474re.html) with an [X-Nucleo-IHM08M1 shield](https://www.st.com/en/ecosystems/x-nucleo-ihm08m1.html) as motor driver, a [commercially available Quanser cartpole system](https://www.quanser.com/products/linear-servo-base-unit-inverted-pendulum/) and a self-build one from the [Max Planck Institute for Intelligent Systems](https://is.mpg.de/). The code running on the microcontroller uses the [modm embedded library builder](https://github.com/modm-io/modm) and some automatic code generation with [jinja templates](https://jinja.palletsprojects.com/en/3.1.x/), which is used to export the neural network [as C++ code here](neural_network_training/embedded_nn_inference). See the [pendulum_embedded](README.md) for details. You can skip the code generation also find the generated [embedded C++ implementation here](pendulum_embedded/embedded_pendulum/pendulum_swingup_nn).