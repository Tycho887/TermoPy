import tp_processes as tpp
import numpy as np
import unittest


class Fluid(tpp.Static):
    def __init__(self,P=None,V=None,T=None,n=None,fluid=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,fluid=fluid,monatomic=monatomic,diatomic=diatomic)
        self.processes = []
        self.heat = []
        self.work = []
        self.entropy = []

    def isothermal(self, P=None, V=None):
        process = tpp.Isothermal(P=self.pressure[-1], 
                                 T = self.temperature[-1], 
                                 V=self.volume[-1], 
                                 n=self.moles[-1], 
                                 fluid=self.fluid, 
                                 monatomic=self.monatomic, 
                                 diatomic=self.diatomic)
        process.final(P=P,V=V)
        self.processes.append(process)

O2 = Fluid(P=1e5, V=1e-3, n=1, fluid='O2')
O2.isothermal(P=2e5)
