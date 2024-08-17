import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def trivariate_loss(V_pred,V_trgt):

    mean = V_pred[:, :, 0:3]
    log_stddev = V_pred[:, :, 3:6]
    stddev = torch.exp(log_stddev)  # Standard deviations
    p_xy = torch.tanh(V_pred[:, :, 6])  # Correlation coefficient x-y
    p_yz = torch.tanh(V_pred[:, :, 7])  # Correlation coefficient y-z
    p_xz = torch.tanh(V_pred[:, :, 8])  # Correlation coefficient x-z

    # Target values
    target = V_trgt[:, :, 0:3]

    # Calculate the differences
    diff = target - mean  # Difference between target and mean (N x 1 x 3)

    # Calculate the covariance matrix
    N = mean.size(0) # 12
    P = mean.size(1) # Ped

    cov = torch.zeros((N,P, 3, 3), device=mean.device)

    cov[:,:, 0, 0] = stddev[:, :, 0] ** 2
    cov[:,:, 1, 1] = stddev[:, :, 1] ** 2
    cov[:,:, 2, 2] = stddev[:, :, 2] ** 2

    cov[:,:, 0, 1] = p_xy * stddev[:, :, 0] * stddev[:, :, 1]
    cov[:,:, 1, 0] = cov[:,:, 0, 1]  # Symmetric

    cov[:,:, 1, 2] = p_yz * stddev[:, :, 1] * stddev[:, :, 2]
    cov[:,:, 2, 1] = cov[:,:, 1, 2]  # Symmetric

    cov[:,:, 0, 2] = p_xz * stddev[:, :, 0] * stddev[:, :, 2]
    cov[:,:, 2, 0] = cov[:,:, 0, 2]  # Symmetric
    cov += 1e-6 * torch.eye(3, device=mean.device).unsqueeze(0).unsqueeze(0)

    # Inverse and determinant of the covariance matrix
    cov_inv = torch.linalg.inv(cov)  # Inverse of covariance matrix
    det_cov = torch.det(cov) # Determinant of covariance matrix
    det_cov = torch.clamp(det_cov, min=1e-10)


    diff = diff.view(N, P, 3, 1)  # Keep diff in 4D shape
    maha_dist = torch.matmul(diff.permute(0, 1, 3, 2), torch.matmul(cov_inv, diff))
    maha_dist = maha_dist.view(N, P)


    # Calculate the log likelihood
    log_likelihood = -0.5 * (maha_dist + 3 * torch.log(torch.tensor(2 * torch.pi)) + torch.log(det_cov))


    # Negative log likelihood
    nll = -torch.mean(log_likelihood)
    # print(nll)

    return nll



    # normx = V_trgt[:,:,0]- V_pred[:,:,0]
    # normy = V_trgt[:,:,1]- V_pred[:,:,1]
    # normz = V_trgt[:,:,2]- V_pred[:,:,2]


    # sx = torch.clamp(torch.exp(V_pred[:,:,3]), min=1e-3)
    # sy = torch.clamp(torch.exp(V_pred[:,:,4]), min=1e-3)
    # sz = torch.clamp(torch.exp(V_pred[:,:,5]), min=1e-3)

    # p_xy = torch.tanh(V_pred[:,:,6]) 
    # p_yz = torch.tanh(V_pred[:,:,7]) 
    # p_xz = torch.tanh(V_pred[:,:,8]) 


    # C = (p_xy*normx*normy)/(sx*sy) + (p_yz*normy*normz)/(sy*sz) + (p_xz*normx*normz)/(sx*sz)
    # Q = (normx/sx)**2 + (normy/sy)**2 + (normz/sz)**2 - 2*C

    # p = 1 - (p_xy**2 + p_yz**2 + p_xz**2) + 2*p_xy*p_yz*p_xz
    # p = torch.clamp(p, min=1e-3)

    # numerator = torch.exp(-Q / (2*p))

    # denominator = np.sqrt((2 * np.pi)**3) * sx * sy * sz * torch.sqrt(p)

    # result = numerator / denominator

    # result = -torch.log(torch.clamp(result, min=1e-3))

    # result = torch.mean(result)
    # print(result)
    # return result
    
    
    


# def trivariate_gaussian_nll(V_pred, V_trgt):
#     """
#     Calculate the negative log likelihood of the trivariate Gaussian distribution.

#     Parameters:
#         V_pred: Predicted values (tensor of shape (N, D, 9)) where:
#             - N is the number of samples
#             - D is the dimension (3 for x, y, z coordinates and 6 for parameters)
#             - V_pred[:,:,0:3] : Mean (x, y, z)
#             - V_pred[:,:,3:6] : Log of standard deviations (sx, sy, sz)
#             - V_pred[:,:,6:9] : Correlation coefficients (p_xy, p_yz, p_xz)
#         V_trgt: Target values (tensor of shape (N, D, 3)) where:
#             - N is the number of samples
#             - D is the dimension (3 for x, y, z coordinates)

#     Returns:
#         nll: The negative log likelihood (scalar).
#     """
    
#     # Extract mean, stddev, and correlation coefficients
#     mean = V_pred[:, :, 0:3]
#     log_stddev = V_pred[:, :, 3:6]
#     stddev = torch.exp(log_stddev)  # Standard deviations
#     p_xy = torch.tanh(V_pred[:, :, 6])  # Correlation coefficient x-y
#     p_yz = torch.tanh(V_pred[:, :, 7])  # Correlation coefficient y-z
#     p_xz = torch.tanh(V_pred[:, :, 8])  # Correlation coefficient x-z

#     # Target values
#     target = V_trgt[:, :, 0:3]

#     # Calculate the differences
#     diff = target - mean  # Difference between target and mean (N x 3)

#     # Calculate the covariance matrix
#     cov = torch.zeros((mean.size(0), 3, 3), device=mean.device)
#     cov[:, 0, 0] = stddev[:, :, 0] ** 2
#     cov[:, 1, 1] = stddev[:, :, 1] ** 2
#     cov[:, 2, 2] = stddev[:, :, 2] ** 2
#     cov[:, 0, 1] = p_xy * stddev[:, :, 0] * stddev[:, :, 1]
#     cov[:, 1, 0] = cov[:, 0, 1]  # Symmetric
#     cov[:, 1, 2] = p_yz * stddev[:, :, 1] * stddev[:, :, 2]
#     cov[:, 2, 1] = cov[:, 1, 2]  # Symmetric
#     cov[:, 0, 2] = p_xz * stddev[:, :, 0] * stddev[:, :, 2]
#     cov[:, 2, 0] = cov[:, 0, 2]  # Symmetric

#     # Inverse and determinant of the covariance matrix
#     cov_inv = torch.linalg.inv(cov)  # Inverse of covariance matrix
#     det_cov = torch.det(cov)  # Determinant of covariance matrix

#     # Compute the Mahalanobis distance
#     maha_dist = torch.bmm(diff.view(-1, 1, 3), torch.bmm(cov_inv, diff.view(-1, 3, 1)))
#     maha_dist = maha_dist.view(-1)  # Flatten back to 1D

#     # Calculate the log likelihood
#     log_likelihood = -0.5 * (maha_dist + 3 * torch.log(2 * torch.pi) + torch.log(det_cov))

#     # Negative log likelihood
#     nll = -torch.mean(log_likelihood)

#     return nll


















    # #mux, muy, muz, sx, sy, sz, rxy, rxz, ryz

    # #assert V_pred.shape == V_trgt.shape
    # normx = V_trgt[:,:,0]- V_pred[:,:,0]
    # normy = V_trgt[:,:,1]- V_pred[:,:,1]

    # sx = torch.exp(V_pred[:,:,2]) #sx
    # sy = torch.exp(V_pred[:,:,3]) #sy
    # corr = torch.tanh(V_pred[:,:,4]) #corr
    
    # sxsy = sx * sy

    # z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    # negRho = 1 - corr**2

    # # Numerator
    # result = torch.exp(-z/(2*negRho))
    # # Normalization factor
    # denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # # Final PDF calculation
    # result = result / denom

    # # Numerical stability
    # epsilon = 1e-20

    # result = -torch.log(torch.clamp(result, min=epsilon))

    # result = torch.mean(result)

    
    # return result
   