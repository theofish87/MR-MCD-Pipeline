# -*- coding: utf-8 -*-

"""
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution.
"""

"""
Module docstring here

"""

__author__ = "Adrien Dubois and David Broadway"

import torch
import torch.nn as nn
import scipy.signal
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import json

from Propagator import Propagator
from Generator import generator_CNN, generator_MLP_bayesian
from Train import train_cnn
from  Evaluate import evaluate
import matplotlib as plt

# ============================================================================


def Magnetization(data, num_mc_samples=50):  # >>> CHANGED: added num_mc_samples for MC Dropout

  #f = open('/content/drive/MyDrive/Colab Notebooks/wp2/Blob_simulation_out_t_clean21')
  unit_conversion = 1e-18 / 9.27e-24
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Extract the data
  #MagneticField = np.asarray(data['ExperimentMagneticField']['BNV']['Data'])
  #MagneticField = np.asarray(data['MagnetisationSimulation']['MagnetisationSimulation']['BNV'])
  #Magnetization = np.array(data['MagnetisationSimulation']['MagnetisationSimulation']['SimMagnetisation'])
  #MagneticField = MagneticField.T - (MagneticField[:,0]).T
  MagneticField = ((data - data.mean()))/unit_conversion

  #display(data['ExperimentMagneticField'].keys())

  # Define the dictionary for the forward propagation


  PropagationOptions = dict()
  PropagationOptions['PixelSize'] =  50e-08
  PropagationOptions['ImageShape'] = 128
  PropagationOptions['NV'] = dict()
  PropagationOptions['NV']['FindTheta']=False
  PropagationOptions['NV']['Theta'] =45
  PropagationOptions['NV']['FindPhi']=False
  PropagationOptions['NV']['Phi'] = 0
  PropagationOptions['NV']['Height'] = 50e-08
  PropagationOptions['use_stand_off'] = True
  PropagationOptions['Magnetisation'] = dict()
  PropagationOptions['Magnetisation']['FindTheta']=False
  PropagationOptions['Magnetisation']['Theta'] = 0
  PropagationOptions['Magnetisation']['FindPhi']=False
  PropagationOptions['Magnetisation']['Phi'] = 20
  PropagationOptions['Mag_z'] = True
  PropagationOptions['unv'] = [0,0,1]
  PropagationOptions['FFT'] = dict()
  PropagationOptions["FFT"]["PaddingFactor"]= None
  PropagationOptions["FFT"]["performPadding"]= None
  PropagationOptions["in_plane_propagation"]= True
  PropagationOptions["in_plane_angle"]= 0
  PROP = Propagator(PropagationOptions, MagneticField,PropagationOptions['ImageShape'])

  PROP.reshape(PropagationOptions['ImageShape'])
  mask = np.where(PROP.MagneticFieldExtended == 0,0,1)
  mask_t=torch.FloatTensor(mask[np.newaxis,np.newaxis])

  img2 = np.where((PROP.MagneticFieldExtended<0.20)&(PROP.MagneticFieldExtended>-0.20),0,PROP.MagneticFieldExtended)
  C = torch.FloatTensor((PROP.MagneticFieldExtended)[np.newaxis,np.newaxis,:,:])
  train_data_3 = TensorDataset(C, mask_t,C)
  train_loader_3 = DataLoader( dataset=(train_data_3))

  ML_options = dict()
  ML_options['FindNVOrientation'] = False
  ML_options['mlp']=False
  ML_options['size']=PropagationOptions['ImageShape']


  G_cnn = generator_CNN(Size=2,ImageSize=PropagationOptions['ImageShape']).to(device)
  G_cnn_optimizer = torch.optim.Adam(G_cnn.parameters())


  res=train_cnn(device, G_cnn, G_cnn_optimizer, train_loader_3, 50, PROP, ML_options)

  means,stds,pred,ci_upper,ci_lower, ic_acc,ic_acc2,loss = evaluate(G_cnn, C.to(device),mask_t.to(device))

# >>> CHANGED: properly unpack the outputs of G_cnn (generator_CNN returns 6 values)
  MInt_det, conv9_det, _, _, _, _ = G_cnn(C.to(device), mask_t.to(device))  # >>> NEW
  #img = G_cnn(C.to(device),mask_t.to(device))
  res=res[0,0].detach().cpu().numpy()
  #MagnetisationMap = img[0][0,0,:,:].detach().cpu().numpy()
  MagnetisationMap = conv9_det[0,0,:,:].detach().cpu().numpy()             # >>> NEW

  border1=int((PropagationOptions['ImageShape'] -MagneticField.shape[0])/2)
  border2=int((PropagationOptions['ImageShape'] -MagneticField.shape[0])/2+MagneticField.shape[0])
  border3=int((PropagationOptions['ImageShape'] -MagneticField.shape[1])/2)
  border4=int((PropagationOptions['ImageShape'] -MagneticField.shape[1])/2+MagneticField.shape[1])

  res2 = res[border1:border2,border3:border4]
  MagnetisationMap2=MagnetisationMap[border1:border2,border3:border4]


  # ------------------------------------------------------------------
  # >>> NEW: Monte Carlo Dropout to estimate uncertainty on M(x,y)
  # ------------------------------------------------------------------
  
  G_cnn.train()   # >>> NEW: keep dropout layers active during inference
  
  for module in G_cnn.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()    #010426 ensure uncertainty map reflects genuine model uncertainty, not random fluctuations from BatchNorm statistics
        
#   def enable_mc_dropout(model):
#       """
#       Sets the model to eval mode, but forces all Dropout layers
#       to stay in train mode for MC Dropout sampling.
#       Done to prevent BatchNorm contamination during inference
#       """
#       model.eval()
#       for module in model.modules():
#           if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
#               module.train()

  #enable_mc_dropout(G_cnn)
  
  mc_maps = []    # >>> NEW: list to collect multiple stochastic reconstructions

  with torch.no_grad():   # >>> NEW: no gradient updates during MC sampling
      for t in range(num_mc_samples):   # >>> NEW: T Monte Carlo samples

          MInt_mc, conv9_mc, _, _, _, _ = G_cnn(
              C.to(device),
              mask_t.to(device),
              PositiveMagnetisationOnly=False,
              IntegerOnly=False
          )  # >>> NEW: one stochastic forward pass due to dropout

          mc_maps.append(conv9_mc.detach().cpu().numpy())  # shape: (1,1,H,W)  # >>> NEW

  mc_stack = np.stack(mc_maps, axis=0)  # shape: (T,1,1,H,W)  # >>> NEW

  # >>> NEW: mean and std across Monte Carlo samples
  mean_map = mc_stack.mean(axis=0)[0,0,:,:]   # (H, W)
  std_map  = mc_stack.std(axis=0, ddof=1)[0,0,:,:]                # unbiased estimation
  #std_map  = mc_stack.std(axis=0)[0,0,:,:]    # (H, W)

  # >>> NEW: crop mean/std to original field size (same borders as above)
  M_mean2 = mean_map[border1:border2, border3:border4]
  M_std2  = std_map[border1:border2, border3:border4]

  # ------------------------------------------------------------------
  # >>> CHANGED: extend stacked output to also include mean + std maps
  #   - index 0: res2     (same as before)
  #   - index 1: MagnetisationMap2 (same as before)
  #   - index 2: M_mean2 (MC Dropout mean)
  #   - index 3: M_std2  (MC Dropout uncertainty)
  # ------------------------------------------------------------------
  stacked = [res2, MagnetisationMap2, M_mean2, M_std2]  # >>> CHANGED
  #stacked = [res2,MagnetisationMap2]
  
  return stacked