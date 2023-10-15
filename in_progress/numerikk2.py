# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:39:05 2022

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
import math
import scipy
import sympy as sp

class Function:
    def __init__(self,func,a=None,b=None,n=1e4,polar=False):
        """
        a: float or list/array. Either start value or list/array
           containing data.
        b: end point of interval.
        n: number of intervals
        """
        self.func = func
        
        #Definer skjente nullpunkter
        self.zeros = []
        
        #Definer området før du kan arbeide
        self.a = a; self.b = b
        
        #Definer x og y punkter
        self.n = math.ceil(n) # antall inndelinger
        self.dx = None
        self.x = None
        self.y = None
        
        #Definer integralsummen
        self.S = None
        
        #misc
        self.h = 1e-4
        self.polar=polar
        self._auto = False # bestemmer om enkelte prosesser skal skjøres automatisk
    
    "Oppsett"
    
    def interval(self,a,b,n=None):
        try:
            self.a = float(a)
        except Exception:
            print('Startverdi må være et gyldig tall')
        try:
            self.b = float(b)
        except Exception:
            print('Startverdi må være et gyldig tall')
        try:
            self.n = int(n)
        except Exception:
            print('antall intervaller må være et gyldig tall')
        return self.a,self.b,self.n
    
    def _setup_values(self,func=None):
        if func==None:
            func = self.func
        if self.a==None or self.b==None:
            raise ValueError("Parametere er udefinert")
        self.x = np.linspace(self.a,self.b,self.n+1)
        self.dx = (self.b-self.a)/self.n
        self.y = func(self.x)
        #return self.x,self.y,self.dx
    
    "Derivering"
    
    def derivative(self,x):
        return (self.func(x+self.h)-self.func(x))/self.h
    def second_deriv(self,x):
        return (self.func(x+self.h)-2*self.func(x)+self.func(x-self.h))/(self.h**2)
        
    "Newtons metode"
    
    def Find_zeros(self,init_x,iterations=1000,deriv_func=None):
        """
        Finn nullpunkter ved Newtons metode
        """
        #assert self.a < init_x < self.b
        assert isinstance(init_x, (float,int,complex))
        x = init_x
        iters = 0
        if deriv_func==None:
            while abs(self.func(x)) > self.h and iters < iterations:
                x -= self.func(x)/self.derivative(x)
            return x
        elif callable(deriv_func):
            while iters < iterations:
                x -= self.func(x)/deriv_func(x)
            return x
        
        
    "Integrering"
    
    def integrate(self,method='simpson',func=None):
        """
        Initialiserer
        """
        if func==None:
            func=self.func
        self._setup_values(func)
        self.S = self.y[0]+self.y[-1]
        
        if method=='simpson':

            self.S += sum([4*i for i in self.y[1:-1:2]])+sum([2*i for i in self.y[2:-1:2]]) 
            self.S *= (self.dx/3)
        
        elif method=='trapes':
    
            self.S += sum([2*i for i in self.y[1:-1]])
            self.S *= 0.5*self.dx
    
        elif method=='midtpunkt':
            
            self.S += sum([func(i) for i in list(self.x[:-1]+(self.dx/2))])
            self.S *= self.dx
        
        else:
            raise Exception('Definer en type')
    
        #print(self.y)
        #print(self.func(2))
        
        
        return self.S
                
    
    "Buelengde"
    
    def buelengde(self):
        data = [self.x,self.y,self.func]; S = 0
        if self.polar:
            func = lambda t : np.sqrt((self.func(t))**2+(self.derivative(t))**2)
            S=self.integrate(func=func)
        else:
            func = lambda t : np.sqrt(1+(self.derivative(t))**2)
            S=self.integrate(func=func)
        self.x,self.y,self.func = data[0],data[1],data[2]
        return S
        
    "polart areal"

    def polar_area(self):
        assert self.polar
        data = [self.x,self.y,self.func]
        func = lambda t: 0.5*(self.func(t))**2
        S = self.integrate(func=func)
        self.x,self.y,self.func = data[0],data[1],data[2]
        return S
        
    "Visualisering"
    
    def draw(self,a=None,b=None,n=None,polar=False):
        fig = plt.figure(dpi=200)
        if polar:
            self.polar=True
        if self.polar:
            
            try:
                self.a = float(a)
                self.b = float(b)
                if n!=None:
                    self.n = int(n)
            except Exception:
                pass
            self._setup_values()
            
            
            ax = fig.add_subplot(projection='polar')
            plt.polar(self.x,self.y,marker='o')
        else:
            self.interval(a, b, n)
            self._setup_values()
            plt.plot(self.x,self.y)
        plt.show()
    
    def getPoints(self):
        return self.x,self.y


class diffeq:
    def __init__(self,func,x0,xn,y0,h=1e-3,numsteps=3000):
        self.func = func
        self.h = h
        
        self.y = [y0]
        self.x = np.arange(x0,xn,h)
        if len(self.x) > numsteps:
            self.x = np.linspace(x0,xn,numsteps)
        
    def solve(self,method='euler'):
        if method=='euler':
            for x in self.x[:-1]:
                yn = self.y[-1] + self.h * self.func(x,self.y[-1])
                self.y.append(yn)
            
        elif method=='heun':
            for x in self.x[:-1]:
                K1 = self.func(x,self.y[-1])
                K2 = self.func(x+self.h,self.y[-1]+self.h*K1)
                yn = self.y[-1] + 0.5*self.h*(K1+K2)
                self.y.append(yn)
        
        return self.x,self.y
    
    def draw(self):
        fig = plt.figure(dpi=1024)
        assert len(self.x)==len(self.y)
        plt.plot(self.x,self.y)
        plt.show()

class Vector:
    def __init__(self,F,C,a=None,b=None,n=1e4):
        self.F = F
        self.C = C
        self.n = n
        
    def __derivative(self,f,P,h=1e-4):
        return (f(P+h)-f(P))/h
    
    def __integrate(self,f,a,b,n):
        I = f(np.linspace(a,b,int(n)))
        S = I[0]+I[-1]
        S += sum([4*i for i in I[1:-1:2]])+sum([2*i for i in I[2:-1:2]])
        S *= (b-a)/(3*n)
        return S
    
    def __circulation(self,t):
        return sum(self.F(self.C(t))*self.__derivative(self.C,t))
    
    def curveIntegral(self,a,b):
        W = self.__integrate(self.__circulation,a,b,self.n)
        return W

# class multi(Function):
#     def __init__(self,func,g1,g2,a,b):
#         self.func = func
#         self.c = a  ; self.d = b
        
#         if callable(g1):
#             self.g1 = g1
#         elif not callable(g1):
#             self.g1 = lambda x: g1
#         if callable(g2):
#             self.g2 = g2
#         elif not callable(g2):
#             self.g2 = lambda x: g2
        
#     def simpson(self,values,dx):
#         S = values[0]+values[-1]
#         S += sum([4*i for i in values[1:-1:2]])+sum([2*i for i in values[2:-1:2]]) 
#         S *= (dx/3)
#         return S
        
    # def integrate(self,h=1e-4):
        
    #     #x_område = lambda x: np.arange(self.g1(x),self.g2(x),h)
    #     y_strips = []
    #     for dx in list(np.arange(self.c, self.d,h)):
    #         Func = Function(lambda y : self.func(dx,y),self.g1(dx),self.g1(dx))
    #         y_strips.append(Func.integrate())
    #         print(y_strips)
    #     return self.simpson(y_strips, h)
    #     #return Volume
            
        
        
        
        


class stats(Function):
    def __init__(self):
        
        self._constant_1 = 1/(math.sqrt(2*math.pi))
        self.mean = None
        self.avvik = None
        self.n = None
        self.alpha = None
    
    def normal(self,x,mean,avvik):
        const = self._constant_1/avvik
        exponent = ((mean-x)**2)/(-2*avvik**2)
        return const*np.exp(exponent)
    
    # def _normal(self,x):
    #     const = self._constant_1/self.avvik
    #     exponent = ((self.mean-x)**2)/(-2*self.avvik**2)
    #     return const*np.exp(exponent)
    
    def poisson():
        pass
    def binomial():
        coeff = scipy.special.binom()
        
    def SE(self,p0,n):
        assert p0 < 1
        return np.sqrt(p0*(1-p0)/n)
    
    def Z_score(self,p,p0,SE):
        return (p-p0)/SE
    
    def erf(self,x):
        self.func = lambda t: np.exp(-t**2)
        self.interval(0, x, 1000)
        return 2*self.integrate()/math.sqrt(math.pi)
    
    def arcerf(self,k):
        """
        Invers til error-funksjonen, bruk denne til å finne de kritiske verdiene z* for
        tilsvarer forskjellige konfidensintervaller.
        """
        assert k<1
        kn = 1
        erf = lambda t: np.sqrt(np.pi)/2*(self.erf(t) - k)
        deriv_erf = lambda t: np.exp(-t**2)
        # Bruker newtons metode til å finne inversverdien
        for i in range(25):
            kn -= erf(kn)/deriv_erf(kn)
        return kn
    
    def p_value(self,p,p0,SE,left=False,right=False):
        kritisk_verdi = p-p0
        self.func = lambda x: self.normal(x=x,mean=p0,avvik=SE)
        
        # implementer switch statements når du oppgraderer til 
        if not(left and right):
            "Tosidig p-verdi"
            self.interval(p0-kritisk_verdi, p0+kritisk_verdi,100000)
        elif left:
            "Venstresidig p-verdi"
            self.interval(p0-kritisk_verdi, SE*10,100000)
        elif right:
            "Høyresidig p-verdi"
            self.interval(SE*-10,+kritisk_verdi, 100000)
        self.integrate()
        return 1-self.S
        
class Normal(stats):
    def __init__(self,mean,avvik):    
        pass

# class Binomial(Stats):
#     def __init__(self):
#         return
    

    



class Sympy:
    def __init__(self):
        pass
    def derivative(self):
        pass
    def arclength():
        pass
    def newton():
        pass
    