import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime
import pandas as pd
import json
import unittest

# Constants

atm = 101300; L = 0.001; R = 8.314; k = 1.38064852e-23 # Boltzmanns konstant
K = 10000; allowed_error = 5e-2 # number of steps and allowed error

# Data

with open("data/substances.JSON","r") as file:
    substances = json.load(file)

version = "1.1.2"

class Static:
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
class Dynamic(Static):
    def __init__(self,n=None,P=None,V=None,T=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,monatomic=monatomic,diatomic=diatomic)

        self.static = False
        self.first_law = None
    
    def _generate_extra_data(self,show):
        assert self.volume is not None and self.pressure is not None and self.temperature is not None, "Volume, pressure and temperature must be defined"
        assert self.n is not None, "Number of moles must be defined"
        assert self.Cv is not None, "Cv must be defined"

        self.work = np.trapz(self.pressure,self.volume)
        self.internal_energy = self.Cv*self.n*self.temperature
        self.entropy = self.n*R*np.log(self.volume)+self.Cv*np.log(self.temperature)

        self.dS = self.entropy[-1]-self.entropy[0]
        self.dE = self.internal_energy[-1]-self.internal_energy[0]
        self.dU = self.heat - self.work

        self.ideal_gas_law = np.mean((self.pressure*self.volume)/(self.n*R*self.temperature))-1

        if self.static:
            self.first_law = 0
            assert self.dU < allowed_error and self.dE < allowed_error, f"dU: {self.dU}, dE: {self.dE}\nheat: {self.heat}, work: {self.work}"
        else:
            self.first_law = ( (self.dE) - (self.heat - self.work)) / np.max([abs(self.dE),abs(self.heat),abs(self.work)])

        self.is_ideal_gas = abs(self.ideal_gas_law) < allowed_error
        self.follows_first_law = abs(self.first_law) < allowed_error

        self.rms = np.sqrt(3*self.temperature*R/self.M)
        self.diameter = 3e-10 # 3 angstrom by default
        self.nv = self.n*6.022e23/self.volume
        self.atomic_mass = self.M/6.022e23
        self.mean_free_path = 1/(np.sqrt(2)*np.pi*self.diameter**2*self.nv)
        self.mean_free_time = self.mean_free_path/self.rms
class Isothermal(Dynamic):
    def __init__(self,n=None,T=None,V=None,P=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isotermisk prosess"
    
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)            
            self.temperature = self.temperature[0]*np.ones(steps)
            self.volume = self.n * R * self.temperature / self.pressure
            self.static = False
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.pressure = self.n * R * self.temperature / self.volume
            self.static = False
        elif T is not None:
            self.temperature = self.temperature[0]*np.ones(steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.static = True
        else:
            raise ValueError("P, V or T must be defined")

        self.heat = R*self.n*self.temperature[0]*np.log(self.volume[-1]/self.volume[0])

        self._generate_extra_data(False)                  
class Isobaric(Dynamic):
    def __init__(self,n=None,P=None,T=None,V=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isobar prosess"
    
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = self.pressure[0]*np.ones(steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.static = True
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.temperature = self.pressure*self.volume/(self.n*R)
            self.static = False
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.volume = self.n*R*self.temperature/self.pressure
            self.static = False
        else:
            raise ValueError("P, V or T must be defined")
        
        self.heat = self.n*self.Cp*(self.temperature[-1]-self.temperature[0])
        
        self._generate_extra_data(False)
class Isochoric(Dynamic):
    def __init__(self,n=None,V=None,T=None,P=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isokor prosess"
    
    
    def final(self,P=None,V=None,T=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.temperature = self.pressure * self.volume / (self.n * R)
            self.static = False
        elif V is not None:
            self.volume = self.volume[0]*np.ones(steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.static = True
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.pressure = self.n*R*self.temperature/self.volume
            self.static = False
        else:
            raise ValueError("P, V or T must be defined")
        
        self.heat = self.n*self.Cv*(self.temperature[-1]-self.temperature[0])

        self._generate_extra_data(False)
class Adiabatic(Dynamic):
    def __init__(self,n=None,P=None,V=None,T=None,monatomic=False,diatomic=False):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic)
        self.title = "Adiabatisk prosess"


    def final(self,P=None,V=None,T=None, steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)
            self.volume = self.volume[0]*(self.pressure[0]/self.pressure)**(1/self.gamma)
            self.temperature = self.temperature[0]*(self.pressure/self.pressure[0])**((self.gamma-1)/self.gamma)        
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.pressure = self.pressure[0]*(self.volume[0]/self.volume)**self.gamma
            self.temperature  = self.temperature[0]*(self.volume[0]/self.volume)**(self.gamma-1)
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.pressure = self.pressure[0]*(self.temperature/self.temperature[0])**(self.gamma/(self.gamma-1))
            self.volume = self.volume[0]*(self.temperature[0]/self.temperature)**(1/(self.gamma-1))
        else:
            raise ValueError("P, V or T must be defined")
        
        self.heat = 0

        self._generate_extra_data(False)

num_tests = 100
class ProcessTester(unittest.TestCase):

    def initial_state(self):
        P = np.random.uniform(0.1*atm,10*atm)
        V = np.random.uniform(1e-4,10)
        T = np.random.uniform(10,1000)
        n = P*V/(R*T)
        monatomic = np.random.choice([True,False]) # FIKS DETTE
        return P,V,T,n,monatomic

    def standard_line(self,type, first_law_errors, ideal_gas_errors):
        return f"\nTesting data generation from {type} change: PASSED\n\n\tFirst law inconsistency: {np.max(first_law_errors)} \n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)}"

    def header(self,text):
        return f"\n{'-'*int((len(text)-1)/2)}{text}{'-'*int((len(text)-1)/2)}"
    
    def assertions(self,process):
        self.assertTrue(process.is_ideal_gas,f"P: {process.pressure[-1]}, V: {process.volume[-1]}, T: {process.temperature[-1]}, n: {process.n}")
        self.assertTrue(process.follows_first_law,f"dE = {process.dE:.4e}, dU = {process.dU:.4e}, heat = {process.heat:.4e}, work = {process.work:.4e}, error: {process.first_law:.4e}")
    
    
    def process_loop_methods(self,get_process):
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            P, V, T, n, mono = self.initial_state()
            # we want to choose a random number between 0 and 100)
            process = get_process(P=P,V=V,T=T,monatomic=mono)
            process.final(P = P*np.random.uniform(0,10))
            self.assertions(process)

            process2 = get_process(P=P,V=V,T=T,monatomic=mono)
            process2.final(P = P/np.random.uniform(0,10))
            self.assertions(process2)

            process3 = get_process(P=P,V=V,T=T,monatomic=mono)
            process3.final(V = V*np.random.uniform(0,10))
            self.assertions(process3)

            process4 = get_process(P=P,V=V,T=T,monatomic=mono)
            process4.final(V = V/np.random.uniform(0,10))
            self.assertions(process4)

            process5 = get_process(P=P,V=V,T=T,monatomic=mono)
            process5.final(T = T*np.random.uniform(0,10))
            self.assertions(process5)

            process6 = get_process(P=P,V=V,T=T,monatomic=mono)
            process6.final(T = T/np.random.uniform(0,10))
            self.assertions(process6)

            first_law_errors.append(process.first_law)
            ideal_gas_errors.append(np.max(process.ideal_gas_law))
        #print(f"{first_law_errors}")
        return first_law_errors, ideal_gas_errors

    def test_Isothermal(self):
        print(self.header("Running isothermal test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: Isothermal(P=P,V=V,T=T,monatomic=monatomic))
        print(self.standard_line("isothermal",Error_first_law,Error_ideal_gas_law))

    def test_Isobaric(self):
        print(self.header("Running isobaric test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: Isobaric(P=P,V=V,T=T,monatomic=monatomic))
        print(self.standard_line("isobaric",Error_first_law,Error_ideal_gas_law))

    def test_Isochoric(self):
        print(self.header("Running isochoric test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: Isochoric(P=P,V=V,T=T,monatomic=monatomic))
        print(self.standard_line("isochoric",Error_first_law,Error_ideal_gas_law))

    def test_Adiabatic(self):
        print(self.header("Running adiabatic test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: Adiabatic(P=P,V=V,T=T,monatomic=monatomic))
        print(self.standard_line("adiabatic",Error_first_law,Error_ideal_gas_law))


if __name__ == "__main__":
    unittest.main()