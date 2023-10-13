# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:50:13 2023

@author: Michael
"""

import numpy as np
from inspect import signature as sg
import matplotlib.pyplot as plt


def F(X):
    # Integrasjonsområde
    x,y,z = X
    return np.sum(X**2)<=1 #and z>=np.sqrt(x**2+y**2)

def g(X):
    # Funksjon
    return True #np.product(X)



class MonteCarlo:
    def __init__(self,F,a,b,dim,g=None,n=1e4):
        
        self.F = F
        self.P = lambda t=0: np.random.uniform(a,b,dim)
        self.Volume = (b-a)**dim
        self.Attempts = int(n)
        
        if g == None:
            self.g = lambda t: 1
        else:
            self.g = g
            
        self.T = "Empty"
        
    def integrate(self):
        
        I = 0
        for i in range(self.Attempts):
            Point = self.P()
            I += self.F(Point)*self.g(Point)
        rate = I/self.Attempts
    
        return rate*self.Volume
    def integral_middel(self,q = 10):
        
        data = np.array([self.integrate() for i in range(q)])
        var = sum((data-data.mean())**2/data.mean())
        konf = 2.36 # 99% konfidens
        ME = np.sqrt(var/q)*konf
        self.T = f"""
Middle: {data.mean():.4f}
99% confidence: [{data.mean()-ME:.4f}, {data.mean()+ME:.4f}]
SE = {np.sqrt(var/q):.4f}
        """
        return data.mean(), data.mean()-ME, data.mean()+ME
        
    def __str__(self):
        self.integral_middel()
        return self.T
        
    
I = MonteCarlo(F, -1, 1, 3)
V = print(I)
   

# V = lambda k: MonteCarlo(F, -1, 1, k)

# print(V(15))

# X = np.arange(1,15)
# Y = [V(x) for x in X]

# for data,dim in zip(Y,X):
#     for content in data:
#         plt.scatter(dim,content,c='blue')

# plt.show()

