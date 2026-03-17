"""
Created on Tue Jan 09 14:14:38 2024
Here we use some routines for speical relativistic hydrodynamics.
"prim2cons_idealHydro_SR" converts the primitive fluid state variables (mass density, velocities in 3 directions and pressure)
into the conservative ones (mass, momentum in 3 directions and total energy, all for unit volume)
"cons2prim_idealHydro_SR" provides the inverse procedure, converting the conservative fluid state into the primitive one
"soundSpeed_SR" calculates the speed of sound.

In the contrast to non-relativistic hydrodynamics, the primitive and conservative variables are coupled in a much more non-linear way,
and a non-linear equation sholud be solved in each cell to obtain the primitive state.
Here we closely follow the paper by Mignone and Bodo (2005) to recover the primitive state (density, 3-velocities and pressure), using the equation for the pressure

I've tried fsolve from scipy, but it worked too slow for 2D arrays (maybe I've done something wrong btw), so that the primitive variables recovery is implemented, using
a simple Newton-Rapson procedure, written by hand (see a function "Newton_find_pres_SR" in this file). equation for pressure is stored in a function "pres_eqn_SR"


By now, all these routines suppose, that we have an ideal gas with gamma-law equation of state 
@author: mrkondratyev
"""
import numpy as np


#sound speed calculation for ideal gamma-law EOS in special relativistic gas dynamics
def soundSpeed_SR(dens, pres, eos):
    
    #enthalpy
    ent = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    #sound speed 
    sound = np.sqrt( pres * eos.GAMMA / (dens * ent + 1e-14) )
    
    return sound 



#conservative variables (dens, mom1,2,3, Etot) recovery from primitive state (dens, vel1,2,3, pres) in special relativistic gas dynamics
def prim2cons_idealHydro_SR(dens, vel1, vel2, vel3, pres, eos):

    #gamma-factor
    gamma = 1.0 / np.sqrt(1.0 - vel1 ** 2 - vel2 ** 2 - vel3 ** 2)    
    #enthalpy - only ideal gamma-law EOS is supported by now
    ent = 1.0 + pres / dens * eos.GAMMA / (eos.GAMMA - 1.0)
    
    #conservative variables 
    mass = dens * gamma
    mom1 = mass * ent * gamma * vel1
    mom2 = mass * ent * gamma * vel2
    mom3 = mass * ent * gamma * vel3
    etot = mass * ent * gamma - pres
    
    return mass, mom1, mom2, mom3, etot



#primitive variables (dens, vel1,2,3, pres) recovery from conservative state (dens, mom1,2,3, Etot) in special relativistic gas dynamics
def cons2prim_idealHydro_SR(mass, mom1, mom2, mom3, etot, Pinit, eos):
       
    #solve the nonlinear equation for pressure
    pres = Newton_find_pres_SR(Pinit, mass, mom1, mom2, mom3, etot, eos.GAMMA)
    
    #obtain relativistic gamma-factor
    gamma = 1.0 / np.sqrt(1.0 - (mom1 ** 2 + mom2 ** 2 + mom3 **2) / (etot + pres) ** 2)
    
    #obtain density 
    dens = mass / gamma
    #enthalpy - only ideal gamma-law EOS is supported by now
    ent = 1.0 + pres / dens * eos.GAMMA / (eos.GAMMA - 1.0)
    
    #obtain velocities
    vel1 = mom1 / mass / gamma / ent
    vel2 = mom2 / mass / gamma / ent
    vel3 = mom3 / mass / gamma / ent
    
    #return recovered primitive (or physical) variables
    return dens, vel1, vel2, vel3, pres




#nonlinear equation which couples a pressure with conservative SR fluid state for constant-gamma ideal gas EOS
def pres_eqn_SR(pres, mass, mom1, mom2, mom3, etot, GAMMA):
    
    #relativistic gamma-factor
    gamma2 = 1.0 / (1.0 - (mom1 ** 2 + mom2 ** 2 + mom3 **2) / (etot + pres) ** 2)    
    gamma = np.sqrt(gamma2)
    #gamma = np.where(gamma2 >= 0.0, np.sqrt(gamma2), 1e4)
    
    #return F from eqn F(pressure) = 0
    return mass * gamma + GAMMA / (GAMMA - 1.0) * pres * gamma ** 2 - etot - pres




#a simple routine with implementation of Newton method for solution of nonlinear equation for the pressure in special relativity
def Newton_find_pres_SR(Pinit, mass, mom1, mom2, mom3, etot, GAMMA):
    
    #parameters of Newton method
    tol = 1e-8
    dx = 1e-12
    maxitr = 100
    
    #pressure = initial guess
    pres = Pinit
    
    #check if we have already converged 
    res = np.abs(pres_eqn_SR(pres, mass, mom1, mom2, mom3, etot, GAMMA)).flatten()
    eps1 = np.max(res)
    eps2 = 1.0
    
    
    #iterations
    itr = 0
    while (itr < maxitr) and (eps1 > tol) and (eps2 > tol):
        
        #update iterations counter
        itr = itr + 1
        
        # delta arg
        dp = pres*(1.0 + dx)
        
        #functions at the previous iteration and at pres + dp to find a derivative
        func1 = pres_eqn_SR(pres, mass, mom1, mom2, mom3, etot, GAMMA)
        func2 = pres_eqn_SR(pres + dp, mass, mom1, mom2, mom3, etot, GAMMA)
        
        #derivative of the lhs
        deriv = (func2 - func1) / dp 
        
        #update pressure
        pres = pres - func1 / deriv
        
        #calculate residual for the whole array
        res = np.abs(pres_eqn_SR(pres, mass, mom1, mom2, mom3, etot, GAMMA).flatten())
        eps1 = np.max(res)
        
        #check if the update is low enough
        eps2 = np.max(np.abs(func1 / deriv / pres).flatten())
        
    if (itr == maxitr):
        print('PRES DID NOT CONVERGE')
        
    #return final pressure for relativistic hydrodynamics
    return pres
    
    
    
    