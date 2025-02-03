# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:17:58 2025

@author: mrkondratyev
"""


import numpy as np


def interp_stag_cell(grid, fvar1, fvar2):
    
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    var1 = 0.5 * (fvar1[1:, :] + fvar1[:-1, :]) 
    var2 = 0.5 * (fvar2[:, 1:] + fvar2[:, :-1])
    
    return var1, var2



def div_Bfld(grid, mhd):
    
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    mhd.divB = (mhd.fb1[1:, :] - mhd.fb1[0:-1, :])/grid.dx1uc + \
    (mhd.fb2[:, 1:] - mhd.fb2[:, :-1])/grid.dx2uc 
    
    
    return mhd