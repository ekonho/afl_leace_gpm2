#!/bin/bash
# DualOrtho: Dual-Orthogonal Projection Federated Fine-Tuning
# Experiment scripts for CIFAR-10 and CIFAR-100

# ============ CIFAR-10 ============
# Moderate non-IID (alpha=0.5)
python main.py \
    --dataset cifar10 \
    --model resnet18 \
    --num_clients 10 \
    --global_rounds 50 \
    --local_epochs 5 \
    --local_lr 0.01 \
    --batch_size 64 \
    --partition dirichlet \
    --alpha 0.5 \
    --memory_var_th 0.97 \
    --memory_max_rank 512 \
    --drift_var_th 0.9 \
    --drift_max_rank 32 \
    --regularization 0.01 \
    --seed 42

# Severe non-IID (alpha=0.1)
# python main.py \
#     --dataset cifar10 \
#     --model resnet18 \
#     --num_clients 10 \
#     --global_rounds 50 \
#     --local_epochs 5 \
#     --local_lr 0.01 \
#     --alpha 0.1 \
#     --drift_max_rank 64

# ============ CIFAR-100 ============
# python main.py \
#     --dataset cifar100 \
#     --model resnet18 \
#     --nf 32 \
#     --num_classes 100 \
#     --num_clients 20 \
#     --global_rounds 80 \
#     --local_epochs 5 \
#     --local_lr 0.01 \
#     --alpha 0.5 \
#     --memory_max_rank 1024 \
#     --drift_max_rank 64
