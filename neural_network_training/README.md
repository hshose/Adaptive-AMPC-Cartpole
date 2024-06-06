# Neural Network Training
In this repo, you find the code for training a neural network to approximate an MPC that controls swing up and balancing of a cartpole pendulum system.

We recommend using a virtual enviornment to run this code:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Then, you can run the training with:
```
python train_hpc.py
```
And the testing with
```
python test_hpc.py
```

Once you get a satisfactory result, you can generate a stand-alone C++ implementation of the neural network:
```
cd embedded_nn_inference
python code_generation.py
```
There is a pretrained model provided in the `models` folder.