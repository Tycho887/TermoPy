# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 19:12:11 2023

@author: Michael
"""
import numpy as np
import sympy as sp
import numerikk2 as nr

#%% Integrasjon og tegning

def f(x):
    return np.sin(x)

F = nr.Function(f)
F.interval(0,np.pi,100)
F.draw()

S = F.integrate(); print(S)
N = F.buelengde()


#%% Bruh

# def g(x):
#     return 1/2 * (np.cos(2*x))**2

# G = nr.Function(g,0,2*np.pi)
# S = G.simpson()/3
# print(S)

# G = nr.Function(g)
#def g(x):
#    return np.exp(x)
#G.draw(0,3)
#x = F.Find_zeros(2); print(x)
#g = F.derivative()
#Bue = F.arclength(); print(Bue)

#%% Diff likninger

# def f(t,y):
#     return 6 - 5*y/(1000+t) 

# F = nr.diffeq(f, 0, 500, 0.2)
# F.solve()
# F.draw()

#%% Statistikk

# E = nr.stats().erf(3.42)
# print(nr.stats().arcerf(E))

P_value = nr.stats().p_value(0.011, 0, 0.0495)

#%% Multivarierte funksjoner

# import numerikk2 as nm

# def f(x,y):
#     return x**2+y**2

# Func = nm.multi(f, 0,1,1,2)
# S = Func.integrate()