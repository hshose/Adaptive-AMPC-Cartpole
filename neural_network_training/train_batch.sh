#!/bin/bash

for ITER in 0 1 2 3 4
do
  python train.py -i $ITER
done