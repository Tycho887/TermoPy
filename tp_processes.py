import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime
import pandas as pd
import json
import unittest


atm = 101300; L = 0.001; R = 8.314; k = 1.38064852e-23 # Boltzmanns konstant
K = 10000; allowed_error = 1e-4 # number of steps and allowed error

print("Loading data...")

with open("data/substances.JSON","r") as file:
    substances = json.load(file)

class StaticGas:
    def __init__(self,P=None,V=None,T=None,n=None,monatomic=False,diatomic=False,name=None):
        
        if monatomic:
            self.fluid = {"M": 4.002602, "Cv": 3/2 * R, "Cp": 5/2 * R, "gamma": 5/3, "formula": "He"}
            self.name = "Helium"
        elif diatomic:
            self.fluid = {"M": 28.0134, "Cv": 5/2 * R, "Cp": 7/2 * R, "gamma": 7/5, "formula": "N2"}
            self.name = "Nitrogen"
        # gas is a dict, so we need to check if it is in the database
        elif name in substances.keys():
            self.fluid = substances[name]
            self.name = name
        # in case the user inputs a formula instead of a name
        elif name in [substances[i]["formula"] for i in substances]:
            self.fluid = substances[[i for i in substances if substances[i]["formula"]==name][0]]
            self.name = [i for i in substances if substances[i]["formula"]==name][0]
        elif name == None and not monatomic and not diatomic:
            self.fluid = substances["Air"]
            self.name = "Air"
        else:
            raise ValueError("The gas you entered is not in the database")
        
        assert diatomic or monatomic or self.fluid != None

        self.M = self.fluid["M"]
        self.Cv = self.fluid["Cv"]
        self.Cp = self.fluid["Cp"]
        self.gamma = self.fluid["gamma"]
        self.formula = self.fluid["formula"]
        
        if P == None:
            self.pressure = np.array([n * R * T / V])
            self.volume = np.array([V])
            self.temperature = np.array([T])
            self.n = n
        elif V == None:
            self.volume = np.array([n * R * T / P])
            self.pressure = np.array([P])
            self.temperature = np.array([T])
            self.n = n
        elif T == None:
            self.temperature = np.array([P * V / (n * R)])
            self.pressure = np.array([P])
            self.volume = np.array([V])
            self.n = n
        elif n == None:
            self.n = P * V / (R * T)
            self.pressure = np.array([P])
            self.volume = np.array([V])
            self.temperature = np.array([T])
        else:
            raise ValueError("You must define three of the variables P, V, T and n")
    
    def __str__(self):
        return f"""Gas: {self.name} [{self.formula}] 
temperature: {self.temperature[-1]}
pressure: {self.pressure[-1]}
volume: {self.volume[-1]}
number of moles: {self.n}
Molar mass: {self.M}
Cv: {self.Cv}
Cp: {self.Cp}
gamma: {self.gamma}"""

#print(StaticGas(P=1*atm,V=1,T=300,monatomic=True))

class Dynamic(StaticGas):
    def __init__(self,n=None,P=None,V=None,T=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,monatomic=monatomic,diatomic=diatomic)
        """
        n: number of moles
        gamma: ratio of specific heats
        P1: initial pressure (Pa)
        V1: initial volume (M^3)
        T1: initial temperature (K)
        """

        self.P = self.pressure[0]
        self.V = self.volume[0]
        self.T = self.temperature[0]


        self.molar_mass = None

        self.internal_energy = None
        self.entropy = None


        self.work_done_by = 0
        self.work_done_on = 0
        self.heat_absorbed = 0
        self.heat_released = 0
        
        self.diameter = 3e-10 # 1 angstrom by default
        self.nv = None # number of particles per volume

        self.rms = None
        self.mean_free_path = None
        self.mean_free_time = None
        
        self.title = ""

        self.ideal_gas_law_consistency = None
        self.first_law_consistency = None

    
    def define_first_law_error(self):
        if abs(self.work+self.heat_absorbed+(self.internal_energy[-1]-self.internal_energy[0])) < allowed_error:
            self.first_law_consistency = 1
        if abs(self.internal_energy[-1]-self.internal_energy[0]) < allowed_error:
            self.first_law_consistency = abs(self.work/self.heat_absorbed)
        elif abs(self.work) < allowed_error:
            self.first_law_consistency = abs(self.heat_absorbed/(self.internal_energy[-1]-self.internal_energy[0]))
        elif abs(self.heat_absorbed) < allowed_error:
            self.first_law_consistency = abs(self.work/(self.internal_energy[-1]-self.internal_energy[0]))
        else:
            self.first_law_consistency = abs((self.work+self.heat_absorbed)/(self.internal_energy[-1]-self.internal_energy[0]))
    
    def _generate_extra_data(self,show):
        assert self.volume is not None and self.pressure is not None and self.temperature is not None, "Volume, pressure and temperature must be defined"
        assert self.n is not None, "Number of moles must be defined"
        assert self.Cv is not None, "Cv must be defined"
        self.internal_energy = self.Cv*self.n*self.temperature
        self.entropy = self.n*R*np.log(self.volume)+self.Cv*np.log(self.temperature)
        self.entropy_change = self.entropy[-1]-self.entropy[0]
        self.ideal_gas_law_consistency = np.mean((self.pressure*self.volume)/(self.n*R*self.temperature))

        self.get_work()
        self.new_work()

        if self.first_law_consistency != 1:
            self.define_first_law_error()
        else:
            self.first_law_consistency = 1
        
        
        if self.molar_mass != None:
            self.rms = np.sqrt(3*self.temperature*R/self.molar_mass)
            self.nv = self.n*6.022e23/self.volume
            self.atomic_mass = self.molar_mass/6.022e23
            self.mean_free_path = 1/(np.sqrt(2)*np.pi*self.diameter**2*self.nv)
            self.mean_free_time = self.mean_free_path/self.rms

        # we want to check if "first law" is a nonetype
        if isinstance(self.first_law_consistency, type(None)):
            print("NONETYPE ERROR:", self.work, self.heat_absorbed, self.internal_energy[-1]-self.internal_energy[0])
        elif 1.02 < self.first_law_consistency < 0.98:
            print("VALUE ERROR:", self.work, self.heat_absorbed, self.internal_energy[-1]-self.internal_energy[0])

    def is_ideal_gas(self):
        result = abs(self.ideal_gas_law_consistency-1) < allowed_error
        if result:
            return True
        else:
            #print(f"Error: {self.ideal_gas_law_consistency}")
            return False
    
    def is_first_law_satisfied(self):
        result = abs(self.first_law_consistency-1) < allowed_error
        if result:
            return True
        else:
            print(f"Error: {self.internal_energy[-1]-self.internal_energy[0]} vs {self.work+self.heat_absorbed}")
            return False
    
    def new_work(self):
        # we calculate the work done using numerical integration
        self.work = np.trapz(self.pressure,self.volume)
        self.heat = self.heat_absorbed

class Isothermal(Dynamic):
    def __init__(self,n=None,T=None,V=None,P=None,monatomic=False,diatomic=False):
        """
        n: number of moles
        T: temperature (K)
        """
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isotermisk prosess"
    
    def get_work(self):
        self.work_done_by = R*self.n*self.temperature[0]*np.log(self.volume[-1]/self.volume[0])
        self.work_done_on = -self.work_done_by
        self.heat_absorbed = -self.work_done_on
        self.heat_released = -self.heat_absorbed
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)            
            self.temperature = self.temperature[0]*np.ones(steps)
            self.volume = self.n * R * self.temperature / self.pressure
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.pressure = self.n * R * self.temperature / self.volume
        elif T is not None:
            self.temperature = self.temperature[0]*np.ones(steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.first_law_consistency = 1
        else:
            raise ValueError("P, V or T must be defined")
        self._generate_extra_data(False)                  
class Isobaric(Dynamic):
    def __init__(self,n=None,P=None,T=None,V=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isobar prosess"
    
    def get_work(self):
        self.work_done_by = self.P*(self.volume[-1]-self.volume[0])
        self.work_done_on = -self.work_done_by
        self.heat_absorbed = self.n*self.Cp*(self.temperature[-1]-self.temperature[0])
        self.heat_released = -self.heat_absorbed
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = self.pressure[0]*np.ones(steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.first_law_consistency = 1
            self.ideal_gas_law_consistency = 1
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.temperature = self.pressure*self.volume/(self.n*R)
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.volume = self.n*R*self.temperature/self.pressure
        else:
            raise ValueError("P, V or T must be defined")
        self._generate_extra_data(False)
class Isochoric(Dynamic):
    def __init__(self,n=None,V=None,T=None,P=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isokor prosess"
    
    def get_work(self):
        self.work_done_by = 0
        self.work_done_on = 0
        self.heat_absorbed = self.n*self.Cv*(self.temperature[-1]-self.temperature[0])
        self.heat_released = -self.heat_absorbed
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.temperature = self.pressure * self.volume / (self.n * R)
        elif V is not None:
            self.volume = self.V*np.ones(steps)
            self.pressure = self.P*np.ones(steps)
            self.temperature = self.T*np.ones(steps)
            self.first_law_consistency = 1
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.volume = self.V*np.ones(steps)
            self.pressure = self.n*R*self.temperature/self.volume
        else:
            raise ValueError("P, V or T must be defined")
        self._generate_extra_data(False)
class Adiabatic(Dynamic):
    def __init__(self,gamma=None,n=None,P=None,V=None,T=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        if gamma != None:
            self.gamma = gamma
        self.title = "Adiabatisk prosess"

    def get_work(self):
        self.work_done_on = self.n * self.Cv * (self.temperature[-1]-self.T)
        self.work_done_by = -self.work_done_on
        self.heat_absorbed = 0
        self.heat_released = 0

    def final(self,P=None,V=None,T=None, steps = K):
        # we want to implement a switch case here
        if P is not None:
            self.pressure = np.linspace(self.P,P,steps)
            self.volume = self.V*(self.P/self.pressure)**(1/self.gamma)
            self.temperature = self.T*(self.pressure/self.P)**((self.gamma-1)/self.gamma)        
        elif V is not None:
            self.volume = np.linspace(self.V,V,steps)
            self.pressure = self.P*(self.V/self.volume)**self.gamma
            self.temperature  = self.T*(self.V/self.volume)**(self.gamma-1)
        elif T is not None:
            self.temperature = np.linspace(self.T,T,steps)
            self.pressure = self.P*(self.temperature/self.T)**(self.gamma/(self.gamma-1))
            self.volume = self.V*(self.T/self.temperature)**(1/(self.gamma-1))
        else:
            raise ValueError("P, V or T must be defined")
        self._generate_extra_data(False)
