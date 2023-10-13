# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:14:42 2022

@author: Michael
"""
import numpy as np
import gc
import matplotlib.pyplot as plt

def P(t):
    return np.array([[-np.sin(1/(t)),t],[-1,-1]])

def g(t):
    return np.array([2,0])

class system_of_diff_eq:
    "Ønsker å skape et objekt som holder"
    "informasjonen om diff-system gjennom hver iterasjon"
    def __init__(self,yk,P=P,g=g):
        self.P = P
        self.g = g
        self.yk = yk
    def calc(self,t):
        # matrise-vektor produkt i numpy kalles ved P.dot(x) = Px
        return self.P(t).dot(self.yk)+self.g(t)

def runga_kutta(t0,stopp,h,P,g,y0):
    "Utfører runga kutta metoden slik den er beskrevet i eksempel 12, s293"
    t_group = np.arange(t0,stopp+h,h)
    yk = y0; return_list = []
    for tk in t_group:
        return_list.append(yk)
        
        K1 = system_of_diff_eq(yk).calc(tk)
        K2 = system_of_diff_eq(yk+(h/2)*K1).calc(tk+h/2)
        K3 = system_of_diff_eq(yk+(h/2)*K2).calc(tk+h/2)
        K4 = system_of_diff_eq(yk+h*K3).calc(tk+h)
        
        yk = yk + (h/6)*(K1+2*K2+2*K3+K4)
            
    "Fri opp minne ved å fjerne ubrukte objekter"
    gc.collect()
    
    return t_group, np.array(return_list)

y0 = [1,0]
        
def output(t0,stopp,h):
    "Tar inn data fra runga kutta funksjonen og plotter det i samme plot."
    x_list, y_points = runga_kutta(t0,stopp,h,P=P,g=g,y0=y0)
    for k in range(len(y_points[0])):
        y_list = y_points[:,k] # kaller k-te søyle i output matrisen
        plt.plot(x_list,y_list,label=f'funksjon index = {k}')
    plt.show()


output(0.01,10,0.01)



