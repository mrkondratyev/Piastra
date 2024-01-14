# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:14:38 2023

here the auxilary data lives, like CFL multiplier, final time, physical time, types of reconstruction and flux 

@author: mrkondratyev
"""
import numpy as np

class auxData:
    
    #here we initialize all important data for our grid 
    def __init__(self, grid):
        
        #Courant-Friedrichs-Lewy multiplier, it can't be more than 2 (in 1D, and more than 1 in 2D)
        self.CFL = 1.0
        
        #reconstruction type (by now, PCM, PLM and WENO are supported)
        self.rec_type = 'WENO'
        
        #flux type
        self.flux_type = 'HLLC'
        
        #Runge-Kutta temporal order
        self.RK_order = "RK3"
        
        #in hydrodynamics we consider 5 conservation laws (density, 3 components of momentum and total energy)
        self.nVar = 5 
        
        #final time
        self.Tfin = 0.0
        
        #phys time 
        self.time = 0.0
