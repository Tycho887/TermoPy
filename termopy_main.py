import tp_processes as tpp
import numpy as np
import unittest

version = '0.1.0'

class Fluid(tpp.Static):
    def __init__(self,P=None,V=None,T=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.monatomic = monatomic; self.diatomic = diatomic
        self.processes = []
        self.heat = []
        self.work = []
        self.entropy = []

    def isothermal(self, P=None, V=None):
        process = tpp.Isothermal(P=self.pressure[-1], 
                                 T = self.temperature[-1], 
                                 V=self.volume[-1], 
                                 n=self.n, 
                                 gas=self.name, 
                                 monatomic=self.monatomic, 
                                 diatomic=self.diatomic)
        self.mass = self.n * self.M
        process.final(P=P,V=V)
        self.processes.append(process)

O2 = Fluid(P=1e5, V=1e-3, T=500, gas='O2')
O2.isothermal(P=2e5)
print(O2)