# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:14:38 2023

here the auxilary data lives, like CFL multiplier, final time, physical time, types of reconstruction and flux 

@author: mrkondratyev
"""

class auxData:
    
    #here we initialize all important data for our grid 
    def __init__(self, grid):
        
        #reconstruction type (by now, PCM, PLM and WENO are supported)
        self.rec_type = 'PLM'
        
        #flux type
        self.flux_type = 'HLL'
        
        #Runge-Kutta temporal order
        self.RK_order = "RK2"
        
        #final time
        self.Tfin = 0.0
        
        #phys time 
        self.time = 0.0
        
        #Courant-Friedrichs-Lewy multiplier, it can't be more than 1 (in 1D, and more than 0.5 in 2D)
        if (grid.Nx1 == 1) or (grid.Nx2 == 1):
            self.CFL = 0.6
        else:
            self.CFL = 0.4
