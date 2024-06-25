"""
Created on Tue Jan 09 14:14:38 2024
Here we use some routines for hydrodynamics.
"prim2cons_idealHydro" converts the primitive fluid state variables (mass density, velocities in 3 directions and pressure)
into the conservative ones (mass, momentum in 3 directions and total energy, all for unit volume)
"cons2prim_idealHydro" provides the inverse procedure, converting the conservative fluid state into the primitive one
"soundSpeed" calculates the speed of sound.

By now, all these routines suppose, that we have an ideal gas with gamma-law equation of state  
@author: mrkondratyev
"""

import numpy as np

#conservative variables (dens, mom1,2,3, Etot) recovery from primitive state (dens, vel1,2,3, pres)
def prim2cons_idealHydro(dens, vel1, vel2, vel3, pres, eos):

    #conservative variables 
    mass = dens
    mom1 = dens * vel1
    mom2 = dens * vel2
    mom3 = dens * vel3
    etot = dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 + pres / (eos.GAMMA - 1.0)
    
    return mass, mom1, mom2, mom3, etot


#primitive variables (dens, vel1,2,3, pres) recovery from conservative state (dens, mom1,2,3, Etot)
def cons2prim_idealHydro(mass, mom1, mom2, mom3, etot, eos):
    
    #primitive variables 
    dens = mass
    vel1 = mom1 / (dens + 1e-14)
    vel2 = mom2 / (dens + 1e-14)
    vel3 = mom3 / (dens + 1e-14)
    pres = (eos.GAMMA - 1.0) * (etot - dens * (vel1**2 + vel2**2 + vel3**2) / 2.0)
    
    return dens, vel1, vel2, vel3, pres

"""
"prim2cons_idealMHD" converts the primitive fluid state variables (mass density, velocities in 3 directions and pressure, as well as B-field)
into the conservative ones (mass, momentum in 3 directions, total energy and B-field (here it is copyed for clarity), all for unit volume)
"cons2prim_idealMHD" provides the inverse procedure, converting the conservative fluid state into the primitive one
By now, all these routines suppose, that we have an ideal gas with gamma-law equation of state  
@author: mrkondratyev
"""

#conservative variables (dens, mom1,2,3, Etot, bcon1,2,3) recovery from primitive state (dens, vel1,2,3, pres, bfld1,2,3)
#bfld 1,2,3 is the same for conservative and primitive state, and thus we dont change it and use it only for energy calc
def prim2cons_idealMHD(dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3, eos):

    #conservative variables 
    mass = dens
    mom1 = dens * vel1
    mom2 = dens * vel2
    mom3 = dens * vel3
    etot = dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 + pres / (eos.GAMMA - 1.0) + (bfld1**2 + bfld2**2 + bfld3**2) / 2.0
    bcon1 = bfld1
    bcon2 = bfld2
    bcon3 = bfld3
    
    
    return mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3


#primitive variables (dens, vel1,2,3, pres, bfld1,2,3) recovery from conservative state (dens, mom1,2,3, Etot, bcon1,2,3)
def cons2prim_idealMHD(mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3, eos):
    
    #primitive variables 
    dens = mass
    vel1 = mom1 / (dens + 1e-14)
    vel2 = mom2 / (dens + 1e-14)
    vel3 = mom3 / (dens + 1e-14)
    bfld1 = bcon1
    bfld2 = bcon2
    bfld3 = bcon3
    pres = (eos.GAMMA - 1.0) * (etot - dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 - (bfld1**2 + bfld2**2 + bfld3**2) / 2.0)
    
    return dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3


#this routine calculates a sound speed for gamma-law ideal gas
def soundSpeed(dens, pres, eos):
    
    sound = np.sqrt( pres * eos.GAMMA / (dens + 1e-14) )
    
    return sound 