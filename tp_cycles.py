import numpy as np
import unittest
import tp_processes as tpp

version = '0.9.0'


class cycle_base(tpp.Static):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T_hot,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        # we assume that the cycle starts at the hot temperature
        self.processes = []
        self.entropy = []
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.compression_ratio = compression_ratio
        self.work = []
        self.heat = []
        self.gas = self.name
        self.heat_added = 0
        self.heat_removed = 0

    def get_efficiency(self):
        heat_in = []; heat_out = []
        for process in self.processes:
            heat = process.heat[-1]
            if heat > 0:
                heat_in.append(heat)
                self.heat_added += heat
            elif heat < 0:
                heat_out.append(heat)
                self.heat_removed += heat
            self.work.append(process.work)
            self.entropy.append(process.dS)
        self.work = sum(self.work)
        self.heat = sum(heat_in+heat_out)
        self.entropy = sum(self.entropy)
        print(self.entropy)
        self.efficeny = abs(self.work/sum(heat_in))
        self.COP = heat_in/abs(self.work)

        assert 0 <= self.efficeny <= 1, 'Efficiency is not between 0 and 1'

    def assert_equal(self,expected,actual):
        assert expected == actual, f'Expected {expected} but got {actual}'
