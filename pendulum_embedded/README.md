# Embedded Software Control Linear Cartpole Pendulum

This is the code required for running the embedded neural network inference for controlling a cartpole pendulum.
To run compile and flash, you need the toolchain for modm embedded library builder, for which you find [installation instructions here](https://modm.io/guide/installation/).

Once installed, you should make sure that all submodules are cloned:
```
git submodule update --init --recursive
```
Now you can generate the required `modm` files by running:
```
cd embedded_pendulum
lbuild build
```
Compile the software with:
```
scons -j8
```
And flash your Nucleo G474 board with:
```
scons program
```
In case you want to export a different neural network, follow the instructions in the [`neural_network_training` folder](../neural_network_training/README.md).