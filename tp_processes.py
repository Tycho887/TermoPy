import numpy as np
import json
import unittest

# Constants

atm = 101300
L = 0.001
R = 8.314
k = 1.38064852e-23 # Boltzmanns konstant
K = 10000
allowed_error = 1e-6 # number of steps and allowed error

# Data

with open("data/substances.JSON","r") as file:
    substances = json.load(file)

version = "1.2.6"

class Static:
    def __init__(self,P=None,V=None,T=None,n=None,monatomic=False,diatomic=False,gas=None):
        
        if monatomic:
            self.properties = {"M": 4.002602, "Cv": 3/2 * R, "Cp": 5/2 * R, "gamma": 5/3, "formula": "He"}
            self.name = "Helium"
        elif diatomic:
            self.properties = {"M": 28.0134, "Cv": 5/2 * R, "Cp": 7/2 * R, "gamma": 7/5, "formula": "N2"}
            self.name = "Nitrogen"
        # gas is a dict, so we need to check if it is in the database
        elif gas in substances.keys():
            self.properties = substances[gas]
            self.name = gas
        # in case the user inputs a formula instead of a name
        elif gas in [substances[i]["formula"] for i in substances]:
            self.properties = substances[[i for i in substances if substances[i]["formula"]==gas][0]]
            self.name = [i for i in substances if substances[i]["formula"]==gas][0]
        elif gas == None and not monatomic and not diatomic:
            self.properties = substances["Air"]
            self.name = "Air"
        else:
            raise ValueError("The gas you entered is not in the database")
        
        assert diatomic or monatomic or self.properties != None

        self.M = self.properties["M"]
        self.Cv = self.properties["Cv"]
        self.Cp = self.properties["Cp"]
        self.gamma = self.properties["gamma"]
        self.formula = self.properties["formula"]
        self.diameter = 3e-10 # 3 angstrom by default
        self.atomic_mass = self.M/6.022e23
        
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
        elif P != None and V != None and T != None and n != None:
            assert P * V - n * R * T < allowed_error, "The variables you entered are not consistent with the ideal gas law"
            self.pressure = np.array([P])
            self.volume = np.array([V])
            self.temperature = np.array([T])
            self.n = n
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
    def __init__(self,n,P,V,T,monatomic,diatomic,gas):
        super().__init__(P=P,V=V,T=T,n=n,monatomic=monatomic,diatomic=diatomic,gas=gas)

        self.static = False
        self.first_law = None
    
    def _generate_extra_data(self):
        assert self.volume is not None and self.pressure is not None and self.temperature is not None, "Volume, pressure and temperature must be defined"
        assert self.n is not None, "Number of moles must be defined"
        assert self.Cv is not None, "Cv must be defined"

        self.work = np.trapz(self.pressure,self.volume)
        self.internal_energy = self.Cv*self.n*self.temperature
        self.entropy = self.n*R*np.log(self.volume)+self.Cv*np.log(self.temperature)

        self.dS = self.entropy[-1]-self.entropy[0]
        self.dE = self.heat - self.work

        self.ideal_gas_law = self.pressure*self.volume-self.n*R*self.temperature
        self.is_ideal_gas = np.max(self.ideal_gas_law) < allowed_error

        self.rms = np.sqrt(3*self.temperature*R/self.M)

        self.nv = self.n*6.022e23/self.volume
        self.mean_free_path = 1/(np.sqrt(2)*np.pi*self.diameter**2*self.nv)
        self.mean_free_time = self.mean_free_path/self.rms
        self.collision_rate = self.nv/self.mean_free_path

class Isothermal(Dynamic):
    def __init__(self,n=None,T=None,V=None,P=None,monatomic=False,diatomic=False,gas=None):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic,gas=gas)
        self.title = "Isotermisk prosess"
    
    def final(self,P=None,V=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)            
            self.temperature = self.temperature[0]*np.ones(steps)
            self.volume = self.n * R * self.temperature / self.pressure
        elif V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.temperature = self.temperature[0]*np.ones(steps)
            self.pressure = self.n * R * self.temperature / self.volume
        else:
            raise ValueError("P or V must be defined")

        self.heat = R*self.n*self.temperature[0]*np.log(self.volume/self.volume[0])

        self._generate_extra_data()                  
class Isobaric(Dynamic):
    def __init__(self,n=None,P=None,T=None,V=None,monatomic=False,diatomic=False,gas=None):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic,gas=gas)
        self.title = "Isobar prosess"
    
    
    def final(self,V=None,T=None,steps = K):
        if V is not None:
            self.volume = np.linspace(self.volume[0],V,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.temperature = self.pressure*self.volume/(self.n*R)
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.pressure = self.pressure[0]*np.ones(steps)
            self.volume = self.n*R*self.temperature/self.pressure
        else:
            raise ValueError("V or T must be defined")
        
        self.heat = self.n*self.Cp*(self.temperature-self.temperature[0])
        
        self._generate_extra_data()
class Isochoric(Dynamic):
    def __init__(self,n=None,V=None,T=None,P=None,monatomic=False,diatomic=False,gas=None):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic,gas=gas)
        self.title = "Isokor prosess"
    

    def final(self,P=None,T=None,steps = K):
        if P is not None:
            self.pressure = np.linspace(self.pressure[0],P,steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.temperature = self.pressure * self.volume / (self.n * R)
        elif T is not None:
            self.temperature = np.linspace(self.temperature[0],T,steps)
            self.volume = self.volume[0]*np.ones(steps)
            self.pressure = self.n*R*self.temperature/self.volume
        else:
            raise ValueError("P or T must be defined")
        
        self.heat = self.n*self.Cv*(self.temperature-self.temperature[0])

        self._generate_extra_data()
class Adiabatic(Dynamic):
    def __init__(self,n=None,P=None,V=None,T=None,monatomic=False,diatomic=False,gas=None):
        super().__init__(n,P=P,V=V,T=T,monatomic=monatomic,diatomic=diatomic,gas=gas)
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

        self._generate_extra_data()

num_tests = 100
class ProcessTester(unittest.TestCase):

    def initial_state(self):
        P = np.random.uniform(0.1*atm,10*atm)
        V = np.random.uniform(1e-4,10)
        T = np.random.uniform(10,1000)
        n = P*V/(R*T)
        #monatomic = np.random.choice([True,False])
        fluid = np.random.choice(list(substances.keys()))
        return P,V,T,n,fluid

    def standard_line(self, ideal_gas_errors):
        return f"\n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)}"

    def header(self,text):
        return f"\n{'-'*int((len(text)-1)/2)}{text}{'-'*int((len(text)-1)/2)}"
    
    def assertions(self,process):
        self.assertTrue(process.is_ideal_gas,f"Process is not ideal gas: {process.name}, {process}")
    
    
    def process_loop_methods(self,get_process):
        ideal_gas_errors = []
        for i in range(num_tests):
            P, V, T, n, gas = self.initial_state()
            process = get_process(P=P,V=V,T=T,gas=gas)
            try: 
                process.final(P = P*np.random.uniform(0,10))
                self.assertions(process)

                process.final(P = P/np.random.uniform(0,10))
                self.assertions(process)
            except TypeError:
                pass
            try:
                process.final(V = V*np.random.uniform(0,10))
                self.assertions(process)

                process.final(V = V/np.random.uniform(0,10))
                self.assertions(process)
            except TypeError:
                pass
            try:
                process.final(T = T*np.random.uniform(0,10))
                self.assertions(process)

                process.final(T = T/np.random.uniform(0,10))
                self.assertions(process)
            except TypeError:
                pass
            ideal_gas_errors.append(np.max(process.ideal_gas_law))
        return ideal_gas_errors

    def test_Isothermal(self):
        print(self.header("Running isothermal test"))
        Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,gas: Isothermal(P=P,V=V,T=T,gas=gas))
        print(self.standard_line(Error_ideal_gas_law))

    def test_Isobaric(self):
        print(self.header("Running isobaric test"))
        Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,gas: Isobaric(P=P,V=V,T=T,gas=gas))
        print(self.standard_line(Error_ideal_gas_law))

    def test_Isochoric(self):
        print(self.header("Running isochoric test"))
        Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,gas: Isochoric(P=P,V=V,T=T,gas=gas))
        print(self.standard_line(Error_ideal_gas_law))

    def test_Adiabatic(self):
        print(self.header("Running adiabatic test"))
        Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,gas: Adiabatic(P=P,V=V,T=T,gas=gas))
        print(self.standard_line(Error_ideal_gas_law))

if __name__ == "__main__":
    unittest.main()
