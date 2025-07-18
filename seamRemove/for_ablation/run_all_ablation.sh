#!/bin/bash

# Run full loss
# python train.py --use_l1 --use_grad --use_lap --exp_name ablation_all

# No Laplacian
python train.py --use_l1 --use_grad --exp_name ablation_noLap

# No Gradient
python train.py --use_l1 --use_lap --exp_name ablation_noGrad

# L1 only
python train.py --use_l1 --exp_name ablation_L1only

# Gradient + Laplacian only
python train.py --use_grad --use_lap --exp_name ablation_noL1

# # Sanity: No loss (should not learn anything)
# python train.py --exp_name ablation_none
