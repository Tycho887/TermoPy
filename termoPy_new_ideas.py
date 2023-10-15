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

class IdealGas:
    def __init__(self, P=None,V=None,T=None,n=None,gas=None):
        self.P = P
        self.V = V
        self.T = T
        self.n = n

        self.set_internal_values(gas)

    def set_internal_values(self,gas):
        if gas in substances:
            self.material = substances[gas]
        else:
            
            self.material = substances[gas]
        else:
            self.material = substances["Air"]
        self.M = self.material["M"]
        self.Cv = self.material["Cv"]
        self.Cp = self.material["Cp"]
        self.gamma = self.material["gamma"]
        self.formula = self.material["formula"]
        

Oxygen = IdealGas(gas="O2")
print(Oxygen.formula)
