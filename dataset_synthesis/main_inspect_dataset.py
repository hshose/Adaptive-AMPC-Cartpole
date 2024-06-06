# MIT License

# Copyright (c) 2024 Henrik Hose

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import time
from dynamicscasadi import f_rod_m as f
from nlmpc import *
from utils import *
import numpy as np
import casadi as ca
import math
import os


def extract_data(x0_flat, X_flat, U_flat, J_flat, N):
    x0 = x0_flat
    U = U_flat.reshape((N, 1)).transpose()
    X = X_flat.reshape((N+1, 4)).transpose()
    J = J_flat.reshape((N, 5)).transpose()
    return x0, X, U, J

if __name__ == '__main__':
    # ---- setup controller     --------
    dt = 160e-3
    N = 25
    
    datafilepath = os.path.join("data", "20231107-184355_test_always_iter_concat")
    
    
    x0_dataset = np.loadtxt( os.path.join(datafilepath, "x0.csv") , delimiter=',')
    X_dataset  = np.loadtxt( os.path.join(datafilepath, "X.csv") , delimiter=',')
    U_dataset  = np.loadtxt( os.path.join(datafilepath, "U.csv") , delimiter=',')
    J_dataset  = np.loadtxt( os.path.join(datafilepath, "J.csv") , delimiter=',')
    
    for i in range(len(x0_dataset[:,0])):
        x0_point, X_point, U_point, _ = extract_data(x0_dataset[i], X_dataset[i], U_dataset[i], J_dataset[i], N)
               
        labels = ["point"]
        U = [U_point]
        X = [X_point]
        plot_pendulum(np.linspace(0, dt*N, N+1), 9, U, X, labels, latexify=False)