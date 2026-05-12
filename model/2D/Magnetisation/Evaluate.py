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
Module that evaluates and plot the results 

"""

__author__ = "Adrien Dubois and David Broadway"

import numpy as np
import torch

# ============================================================================


def evaluate(regressor, img, mask, samples = 10, std_multiplier = 2):

    pred = [regressor(img,mask)[0] for i in range(samples)]

    preds = torch.stack(pred)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)

    mean = img.mean()
    std = img.std()
    ci_upper = mean + (std_multiplier * std)
    ci_lower = mean - (std_multiplier * std)
    ic_acc = (ci_lower <= img) * (ci_upper >= img)
    ic_acc2 = ic_acc.float().mean()

    se = torch.pow((img-mask),2)
    inv_std = torch.exp(-std)
    mse = (inv_std*se)
    reg = (std)
    loss = 0.5*(mse+reg)
            
    return means,stds,pred,ci_upper,ci_lower, ic_acc,ic_acc2,loss
