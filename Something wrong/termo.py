import numpy as np
import matplotlib.pyplot as plt
import json

R = 8.31446261815324
atm = 101325
NA = 6.02214076e23
k = R/NA

with open("data/substances.JSON") as file:
    substances = json.load(file)

# We want to create a dictionary with the properties of most common substances

class Process:
    def __init__(self,P,V,T,n,Cv,Cp,gamma):
        self.P = np.array([P])
        self.V = np.array([V])
        self.T = np.array([T])
        self.n = n
        self.Cv = Cv
        self.Cp = Cp
        self.gamma = gamma
    def generate_data_from_process(self)
        


class IdealGas:
    def __init__(self, P=None,V=None,T=None,n=None,gas=None):
        
        P1,V1,T1,self.n = self.find_missing_values(P,V,T,n)

        self.P = np.array([P1])
        self.V = np.array([V1])
        self.T = np.array([T1])

        self.set_internal_values(gas)

    def __str__(self):
        return f"""The gas is {self.name} [{self.formula}]
The current temperature is {self.T[-1]} K
The current pressure is {self.P[-1]} Pa
The current volume is {self.V[-1]} m3
The current number of moles is {self.n} mol
and molar mass {self.M} g/mol
The heat capacity at constant volume is {self.Cv} J/mol*K
The heat capacity at constant pressure is {self.Cp} J/mol*K
The adiabatic index is {self.gamma}"""

    def set_internal_values(self,gas):
        if gas in substances:
            self.material = substances[gas]
            self.name = gas
        # in case the user inputs a formula instead of a name
        elif gas in [substances[i]["formula"] for i in substances]:
            self.material = substances[[i for i in substances if substances[i]["formula"]==gas][0]]
            self.name = [i for i in substances if substances[i]["formula"]==gas][0]
        else:
            print("The gas you entered is not in the database")
            self.material = substances["Air"]
            self.name = "Air"
        self.M = self.material["M"]
        self.Cv = self.material["Cv"]
        self.Cp = self.material["Cp"]
        self.gamma = self.material["gamma"]
        self.formula = self.material["formula"]
        
    def find_missing_values(self,P,V,T,n):

        assert (P is None) + (V is None) + (T is None) + (n is None) == 1, "You must specify exactly one missing value"
        
        if P is None:
            P = n*R*T/V
        elif V is None:
            V = n*R*T/P
        elif T is None:
            T = P*V/(n*R)
        elif n is None:
            n = P*V/(R*T)
        else:
            assert np.abs(P*V - n*R*T) < 1e-10, "The values you entered are not consistent"
        return P,V,T,n

    def isothermal

Oxygen = IdealGas(gas="O2",P=atm,V=1e-3,T=300)
print(Oxygen.formula)
print(Oxygen.T)
print(Oxygen)
