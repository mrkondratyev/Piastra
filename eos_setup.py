# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 14:14:38 2024
A very simple class with equation of state data.
by now, it only contains the GAMMA index and supports only ideal gas EOS
@author: mrkondratyev
"""

class EOSdata:
    
    #here we initialize all important data for equation of state
    # by now, only gamma-law EOS is supported 
    def __init__(self, gamma):
        
        #gamma-index for ideal gas EOS 
        self.GAMMA = gamma
        
