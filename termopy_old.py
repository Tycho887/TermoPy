import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# we need to know what time it is
import datetime
import pandas as pd
import json

atm = 101300; L = 0.001; R = 8.314; k = 1.38064852e-23 # Boltzmanns konstant
K = 10000; allowed_error = 1e-4 # number of steps and allowed error

# we want to read the file "data/substances.JSON" as a dictionary

with open("data/substances.JSON","r") as file:
    substances = json.load(file)


class Gas:
    def __init__(self,name=None,monatomic=False,diatomic=False):
        if monatomic:
            self.fluid = substances["Helium"]
        elif diatomic:
            self.fluid = substances["Nitrogen"]
        # gas is a dict, so we need to check if it is in the database
        if name in substances.keys():
            self.fluid = substances[name]
            self.name = name
        # in case the user inputs a formula instead of a name
        elif name in [substances[i]["formula"] for i in substances]:
            self.fluid = substances[[i for i in substances if substances[i]["formula"]==name][0]]
            self.name = [i for i in substances if substances[i]["formula"]==name][0]
        elif name == None:
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
    
    def __str__(self):
        return f"Gas: {self.name}\nMolar mass: {self.M}\nCv: {self.Cv}\nCp: {self.Cp}\ngamma: {self.gamma}\n"

print(Gas("Acetone"))

class IdealGas:
    def __init__(self,n=None,P1=None,V1=None,T1=None,monatomic=False,diatomic=False):
        """
        n: number of moles
        gamma: ratio of specific heats
        P1: initial pressure (Pa)
        V1: initial volume (M^3)
        T1: initial temperature (K)
        """
        self.n = n
        self.P1 = P1
        self.V1 = V1
        self.T1 = T1

        self.volume = None
        self.pressure = None
        self.temperature = None

        if monatomic:
            self.Cv = 3/2
            self.Cp = 5/2
            self.gamma = 5/3
        elif diatomic:
            self.Cv = 5/2
            self.Cp = 7/2
            self.gamma = 7/5
        else:
            self.Cv = 3/2
            self.Cp = 5/2
            self.gamma = 5/3
        
        self.molar_mass = None

        self.internal_energy = None
        self.entropy = None


        self.work_done_by = 0
        self.work_done_on = 0
        self.heat_absorbed = 0
        self.heat_released = 0
        
        self.diameter = 3e-10 # 1 angstrom by default
        self.nv = None # number of particles per volume

        self.rms_speed = None
        self.mean_free_path = None
        self.mean_free_time = None
        
        self.title = ""

        self.ideal_gas_law_consistency = None
        self.first_law_consistency = None

    def _generate_extra_data(self,show):
        assert self.volume is not None and self.pressure is not None and self.temperature is not None, "Volume, pressure and temperature must be defined"
        assert self.n is not None, "Number of moles must be defined"
        assert self.Cv is not None, "Cv must be defined"
        self.internal_energy = self.Cv*self.n*R*self.temperature
        self.entropy = self.n*R*np.log(self.volume)+self.Cv*np.log(self.temperature)
        self.ideal_gas_law_consistency = self.pressure*self.volume-(self.n*R*self.temperature)
        self.calculate_work_done_by()
        self.calculate_heat_absorbed()
        self.work_done_on = -self.work_done_by
        self.heat_released = -self.heat_absorbed

        
        if self.molar_mass != None:
            self.rms_speed = np.sqrt(3*self.temperature*R/self.molar_mass)
            self.nv = self.n*6.022e23/self.volume
            self.atomic_mass = self.molar_mass/6.022e23
            self.mean_free_path = 1/(np.sqrt(2)*np.pi*self.diameter**2*self.nv)
            self.mean_free_time = self.mean_free_path/self.rms_speed
        
        self.first_law_consistency = self.work_done_on+self.heat_absorbed-self.internal_energy[-1]+self.internal_energy[0]

        if show: self.plot_PV()

    def P(self,V,T):
        return self.n*R*T/V
    
    def V(self,P,T):
        return self.n*R*T/P
    
    def T(self,P,V):
        return P*V/(self.n*R)
    
    def get_n(self,P,V,T):
        return P*V/(R*T)
    
    def generate_data_from_dV(self,V2,show=False,steps=K):
        self.volume      = self.V1*np.ones(steps)
        self.pressure    = self.P1*np.ones(steps)
        self.temperature = self.T1*np.ones(steps)
        self._generate_extra_data(show)
        return self.volume,self.pressure
    
    def generate_data_from_dP(self,P2,show=False,steps=K):
        return self.generate_data_from_dV(self.V1,show=show,steps=steps)
    
    def generate_data_from_dT(self,T2,show=False,steps=K):  
        return self.generate_data_from_dV(self.V1,show=show,steps=steps)
    
    def maxwell_boltzmann_speed_distribution(self,T=None):
        if T == None:
            T = np.max(self.temperature)
        v_max = np.sqrt(2*k*T/self.atomic_mass)
        standard_deviation = np.sqrt(k*T/self.atomic_mass)
        v = np.linspace(v_max-3*standard_deviation,v_max+3*standard_deviation,1000)
        return 4/np.sqrt(np.pi)*(self.atomic_mass/(k*T))**(3/2)*v**2*np.exp(-self.atomic_mass*v**2/(2*k*T))
    
    def calculate_work_done_by(self):
        self.work_done_by = self.P1*(self.volume[-1]-self.V1)
        self.work_done_on = -self.work_done_by
        return self.work_done_by
    
    def calculate_heat_absorbed(self):
        self.heat_absorbed = self.n*(self.Cv+1)*R*(self.temperature[-1]-self.T1)
        self.heat_released = -self.heat_absorbed
        return self.heat_absorbed

    def _show_picture(self,save=False,name=None,xlabel=None,ylabel=None,type=""):
        plt.grid()
        plt.legend()
        plt.title(f"{self.title}")
        if xlabel != None and ylabel != None:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        # we want to save the fig with the date as a name
        
        if save and name==None: 
            now = datetime.datetime.now()
            plt.savefig(f"{self.title}_{type}_{now.strftime('%Y_%m_%d')}.png",dpi=1024)
        elif save: plt.savefig(f"{name}.png",dpi=1024)
        plt.show()

    def plot_PV(self,save=False,name=None):
        plt.plot(self.volume,self.pressure,label=self.title)
        plt.scatter(self.volume[0],self.pressure[0],label="Startpunkt")
        plt.scatter(self.volume[-1],self.pressure[-1],label="Sluttpunkt")
        self._show_picture(save,name,"Volum [m^3]", "Trykk [Pa]","PV")

    def plot_PT(self,save=False,name=None):
        plt.plot(self.temperature,self.pressure,label=self.title)
        plt.scatter(self.temperature[0],self.pressure[0],label="Startpunkt")
        plt.scatter(self.temperature[-1],self.pressure[-1],label="Sluttpunkt")
        self._show_picture(save,name,"Temperatur [K]", "Trykk [Pa]", "PT")

    def plot_VT(self,save=False,name=None):
        plt.plot(self.temperature,self.volume,label=self.title)
        plt.scatter(self.temperature[0],self.volume[0],label="Startpunkt")
        plt.scatter(self.temperature[-1],self.volume[-1],label="Sluttpunkt")
        self._show_picture(save,name,"Temperatur [K]", "Volum [m^3]", "VT")

    def plot_ST(self,save=False,name=None):
        plt.plot(self.temperature,self.entropy,label=self.title)
        plt.scatter(self.temperature[0],self.entropy[0],label="Startpunkt")
        plt.scatter(self.temperature[-1],self.entropy[-1],label="Sluttpunkt")
        self._show_picture(save,name,"Temperatur [K]", "Entropi [J/K]", "ST")

    def plot_PVT(self,save=False,name=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.volume,self.pressure,self.temperature,label=self.title)
        ax.scatter(self.volume[0],self.pressure[0],self.temperature[0],label="Startpunkt")
        ax.scatter(self.volume[-1],self.pressure[-1],self.temperature[-1],label="Sluttpunkt")
        ax.set_xlabel("Volum [m^3]")
        ax.set_ylabel("Trykk [Pa]")
        ax.set_zlabel("Temperatur [K]")
        self._show_picture(save,name,type="PVT")

    def _find_missing(self):
        assert (self.P1 != None) + (self.V1 != None) + (self.T1 != None) + (self.n != None) > 2, "Tre av P1,V1,T1 eller n må være definert"
        if self.P1 == None:
            #print("P1 er ikke definert, regner ut P1")
            self.P1 = self.P(self.V1,self.T1)
        elif self.V1 == None:
            #print("V1 er ikke definert, regner ut V1")
            self.V1 = self.V(self.P1,self.T1)
        elif self.T1 == None:
            #print("T1 er ikke definert, regner ut T1")
            self.T1 = self.T(self.P1,self.V1)
        elif self.n == None:
            #print("n er ikke definert, regner ut n")
            self.n = self.get_n(self.P1,self.V1,self.T1)
    
    def __str__(self): # the __str__ method is used when printing the object
        return f"n: {self.n}\nP1: {self.P1}\nV1: {self.V1}\nT1: {self.T1}\nCv: {self.Cv}\ngamma: {self.gamma}\n"

    def is_ideal_gas(self):
        return np.all(np.abs(self.ideal_gas_law_consistency) < allowed_error)
    
    def is_first_law_satisfied(self):
        return np.all(np.abs(self.first_law_consistency) < allowed_error)
class Isothermal(IdealGas):
    def __init__(self,n=None,T1=None,V1=None,P1=None,monatomic=False,diatomic=False):
        """
        n: number of moles
        T: temperature (K)
        """
        super().__init__(n,P1=P1,V1=V1,T1=T1,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isotermisk prosess"
        self._find_missing()

    def calculate_work_done_by(self):
        self.work_done_by = R*self.n*self.T1*np.log(self.volume[-1]/self.V1)
        self.work_done_on = -self.work_done_by
        return self.work_done_by

    def calculate_heat_absorbed(self):
        self.heat_absorbed = -self.work_done_on
        self.heat_released = -self.heat_absorbed
        return self.heat_absorbed

    def generate_data_from_dV(self,V2,show=False,steps=K):
        self.volume = np.linspace(self.V1,V2,steps)
        self.pressure = self.P(self.volume,self.T1)
        self.temperature = self.T1*np.ones(len(self.volume))
        self._generate_extra_data(show)
        return self.volume,self.pressure

    def generate_data_from_dP(self,P2,show=False,steps=K):
        self.pressure = np.linspace(self.P1,P2,steps)
        self.volume = self.V(self.pressure,self.T1)
        self.temperature = self.T1*np.ones(len(self.pressure))
        self._generate_extra_data(show)
        return self.volume,self.pressure
class Isobaric(IdealGas):
    def __init__(self,n=None,P1=None,T1=None,V1=None,monatomic=False,diatomic=False):
        super().__init__(n,P1=P1,V1=V1,T1=T1,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isobar prosess"
        self._find_missing()
    
    def calculate_heat_absorbed(self):
        self.heat_absorbed = self.n*self.Cp*R*(self.temperature[-1]-self.T1)
        self.heat_released = -self.heat_absorbed
        return self.heat_absorbed
    
    def calculate_work_done_by(self):
        self.work_done_by = self.P1*(self.volume[-1]-self.V1)
        self.work_done_on = -self.work_done_by
        return self.work_done_by
    
    def generate_data_from_dV(self,V2,show=False, steps = K):
        self.volume = np.linspace(self.V1,V2,steps)
        self.temperature = self.T(self.P1,self.volume)
        self.pressure = self.P1*np.ones(len(self.volume))
        self._generate_extra_data(show)
        return self.volume,self.pressure
    
    def generate_data_from_dT(self,T2,show=False, steps = K):
        self.temperature = np.linspace(self.T1,T2,steps)
        self.volume = self.V(self.P1,self.temperature)
        self.pressure = self.P1*np.ones(len(self.temperature))
        self._generate_extra_data(show)
        return self.volume,self.pressure
class Isochoric(IdealGas):
    def __init__(self,n=None,V1=None,T1=None,P1=None,monatomic=False,diatomic=False):
        super().__init__(n,P1=P1,V1=V1,T1=T1,monatomic=monatomic,diatomic=diatomic)
        self.title = "Isokor prosess"
        self._find_missing()
    
    def calculate_heat_absorbed(self):
        self.heat_absorbed = self.n*self.Cv*R*(self.temperature[-1]-self.T1)
        self.heat_released = -self.heat_absorbed
        return self.heat_absorbed
    
    def calculate_work_done_by(self):
        self.work_done_by = 0
        self.work_done_on = 0
        return self.work_done_by
    
    def generate_data_from_dT(self,T2,show=False, steps = K):
        self.temperature = np.linspace(self.T1,T2,steps)
        self.pressure = self.P(self.V1,self.temperature)
        self.volume = self.V1*np.ones(len(self.temperature))
        self._generate_extra_data(show)
        return self.volume,self.pressure
    
    def generate_data_from_dP(self,P2,show=False, steps = K):
        self.pressure = np.linspace(self.P1,P2,steps)
        self.temperature = self.T(self.pressure,self.V1)
        self.volume = self.V1*np.ones(len(self.pressure))
        self._generate_extra_data(show)
        return self.volume,self.pressure
class Adiabatic(IdealGas):
    def __init__(self,gamma=None,n=None,P1=None,V1=None,T1=None,monatomic=False,diatomic=False):
        super().__init__(n,P1=P1,V1=V1,T1=T1,monatomic=monatomic,diatomic=diatomic)
        if gamma != None:
            self.gamma = gamma
        self.title = "Adiabatisk prosess"
        self._find_missing()
    
    def P2_from_V2(self,V2):
        assert self.P1 != None and self.V1 != None, "P1,V1 må være definert"
        return self.P1*(self.V1/V2)**self.gamma
    
    def T2_from_V2(self,V2):
        assert self.V1 != None and self.T1 != None, "V1,T1 må være definert"
        return self.T1*(self.V1/V2)**(self.gamma-1)
    
    def V2_from_P2(self,P2):
        assert self.P1 != None and self.V1 != None, "P1,V1 må være definert"
        return self.V1*(self.P1/P2)**(1/self.gamma)
    
    def T2_from_P2(self,P2):
        assert self.P1 != None and self.T1 != None, "P1,T1 må være definert"
        return self.T1*(P2/self.P1)**((self.gamma-1)/self.gamma)
    
    def P2_from_T2(self,T2):
        assert self.T1 != None and self.P1 != None, "T1,P1 må være definert"
        return self.P1*(T2/self.T1)**(self.gamma/(self.gamma-1))
    
    def V2_from_T2(self,T2):
        assert self.T1 != None and self.V1 != None, "T1,V1 må være definert"
        return self.V1*(self.T1/T2)**(1/(self.gamma-1))
    
    def calculate_heat_absorbed(self):
        self.heat_absorbed = 0
        self.heat_released = 0
        return self.heat_absorbed
    
    def calculate_work_done_by(self):
        self.work_done_on = self.n * self.Cv * R * (self.temperature[-1]-self.T1)
        self.work_done_by = -self.work_done_on
        return self.work_done_by
    
    def generate_data_from_dV(self,V2,show=False, steps = K):
        self.volume = np.linspace(self.V1,V2,steps)
        self.pressure = self.P2_from_V2(self.volume)
        self.temperature = self.T2_from_V2(self.volume)
        self._generate_extra_data(show)
        return self.volume,self.pressure

    def generate_data_from_dP(self,P2,show=False, steps = K):
        self.pressure = np.linspace(self.P1,P2,steps)
        self.volume = self.V2_from_P2(self.pressure)
        self.temperature = self.T2_from_P2(self.pressure)
        self._generate_extra_data(show)
        return self.volume,self.pressure
    
    def generate_data_from_dT(self,T2,show=False, steps = K):
        self.temperature = np.linspace(self.T1,T2,steps)
        self.volume = self.V2_from_T2(self.temperature)
        self.pressure = self.P2_from_T2(self.temperature)
        self._generate_extra_data(show)
        return self.volume,self.pressure
    
class BaseCycle:
    def __init__(self,P1=None,V1=None,n=None,compression_ratio=None,T_hot=None,T_cold=None,monatomic=False,diatomic=False):
           
        self.V1 = V1
        self.P1 = P1
        self.T = T_hot
        self.n = n

        self.compression_ratio = compression_ratio
        self.T_hot = T_hot
        self.T_cold = T_cold

        self.alpha = None
        self.beta = None

        self.volume = None
        self.pressure = None
        self.temperature = None
        self.entropy = None

        self.efficiency = None

        self.work_done_by = 0
        self.work_done_on = 0
        self.heat_absorbed = 0
        self.heat_released = 0

        self.theoretical_efficiency = None

        self.diatomic = diatomic
        self.monatomic = monatomic

        if monatomic:
            self.gamma = 5/3
            self.Cv = 3/2
            self.Cp = 5/2
        elif diatomic:
            self.gamma = 7/5
            self.Cv = 5/2
            self.Cp = 7/2
        else:
            # we assume that the gas is monatomic if nothing else is specified
            self.gamma = 5/3
            self.Cv = 3/2
            self.Cp = 5/2


        self.molar_mass = None

        self.processes = []
        self.title = ""

        self.cycle_is_ideal_gas = None
        self.cycle_is_first_law_satisfied = None

    def _calculate_alpha_beta(self):
        self.alpha = 1
        self.beta  = 1

    def _calculate_work_in_cycle(self):
        self.work_done_by = [Process.work_done_by for Process in self.processes]
        self.work_done_on = [Process.work_done_on for Process in self.processes]

    def _calculate_heat_in_cycle(self):
        self.heat_absorbed = [Process.heat_absorbed for Process in self.processes]
        self.heat_released = [Process.heat_released for Process in self.processes]
    
    def _calculate_carnot_efficiency(self):
        self.theoretical_efficiency = 1-self.T_cold/self.T_hot

    def _calculate_efficiency(self):
        assert self.heat_absorbed != 0, "Heat absorbed must be calculated before efficiency"
        # we want to sum only the positive values
        Q_in = sum([Q for Q in self.heat_absorbed if Q > 0])
        Q_out = sum([Q for Q in self.heat_absorbed if Q < 0])
        work_done = np.sum(self.work_done_by)
        efficiency_from_heat = 1 - abs(Q_out/Q_in)
        efficiency_from_work = abs(work_done/Q_in)
        assert abs(efficiency_from_heat-efficiency_from_work) < allowed_error, f"Efficiency from heat: {efficiency_from_heat:.2f}, efficiency from work: {efficiency_from_work:.2f}"
        self.efficiency = np.mean([efficiency_from_heat,efficiency_from_work])



    def _find_missing(self):
        assert self.P1 != None or self.V1 != None or self.n != None or self.T != None, "Tre av P,V,n eller T må være definert"
        if self.P1 == None:
            self.P1 = self.n*R*self.T/self.V1
        elif self.V1 == None:
            self.V1 = self.n*R*self.T/self.P1
        elif self.T == None:
            self.T = self.P1*self.V1/(self.n*R)
        elif self.n == None:
            self.n = self.P1*self.V1/(self.T*R)

    def generate_processes(self):
        pass

    def _show_plt(self,save=False,name=None,xlabel=None,ylabel=None,type=""):
        plt.grid()
        plt.legend()
        plt.title(f"{self.title}: {type}")
        if xlabel != None and ylabel != None:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        if save:
            now = datetime.datetime.now()
            plt.savefig(f"{self.title}_{type}_{now.strftime('%Y_%m_%d')}.png",dpi=1024)
        plt.show()

    def process_label(self,process):
        title = process.title.split(" ")[0]
        if title != "Adiabatisk": title = title[:-3]
        if process.volume[-1] > process.volume[0]:
            return f"{title} ekspansjon"
        elif isinstance(process,Isochoric):
            return f"Isokor process"
        else:
            return f"{title} kompresjon"

    def plot_PV(self,save=False,name=None):
        for process in self.processes:
            plt.plot(process.volume,process.pressure,label=self.process_label(process))
            plt.scatter(process.volume[0],process.pressure[0])
        self._show_plt(save,name,"Volum [m^3]", "Trykk [Pa]", "Pressure-Volume")

    def plot_PT(self,save=False,name=None):
        for process in self.processes:
            plt.plot(process.temperature,process.pressure,label=self.process_label(process))
            plt.scatter(process.temperature[0],process.pressure[0])
        self._show_plt(save,name,"Temperatur [K]", "Trykk [Pa]", "Pressure-Temperature")

    def plot_VT(self,save=False,name=None):
        for process in self.processes:
            plt.plot(process.temperature,process.volume,label=self.process_label(process))
            plt.scatter(process.temperature[0],process.volume[0])
        self._show_plt(save,name,"Temperatur [K]", "Volum [m^3]", "Volume-Temperature")

    def plot_ST(self,save=False,name=None):
        for process in self.processes:
            plt.plot(process.temperature,process.entropy,label=self.process_label(process))
            plt.scatter(process.temperature[0],process.entropy[0])
        self._show_plt(save,name,"Temperatur [K]", "Entropi [J/K]", "Entropy-Temperature")
    
    def plot_PVT(self,save=False,name=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for process in self.processes:
            ax.plot(process.volume,process.pressure,process.temperature,label=self.process_label(process))
            ax.scatter(process.volume[0],process.pressure[0],process.temperature[0])
        ax.set_xlabel("Volum [m^3]")
        ax.set_ylabel("Trykk [Pa]")
        ax.set_zlabel("Temperatur [K]")
        self._show_plt(save,name,type="PVT")

    def run_cycle(self):
        self.generate_processes()
        self._calculate_work_in_cycle()
        self._calculate_heat_in_cycle()
        self._calculate_efficiency()
        self.cycle_is_first_law_satisfied = np.all([process.is_first_law_satisfied() for process in self.processes])
        self.cycle_is_ideal_gas = np.all([process.is_ideal_gas() for process in self.processes])
class Carnot(BaseCycle):
    def __init__(self,compression_ratio,T_hot,T_cold,P1=None,V1=None,n=None,monatomic=False,diatomic=False):
        super().__init__(P1=P1,V1=V1,n=n,
                         compression_ratio=compression_ratio,
                         T_hot=T_hot,T_cold=T_cold,
                         monatomic=monatomic,diatomic=diatomic)
        
        self.title = "Carnot syklus"
        self.theoretical_efficiency = 1-T_cold/T_hot
        self._find_missing()
        self._calculate_alpha_beta()
        self.run_cycle()

    def _calculate_alpha_beta(self):
        self.alpha = self.compression_ratio * (self.T_cold/self.T_hot)**(1/(self.gamma-1))
        assert self.alpha, f"Compression ratio is too low, minimum compression: {(self.T_cold/self.T_hot)**(1/(1-self.gamma)):.2f}"

        self.beta  = 1 / self.compression_ratio * (self.T_hot/self.T_cold)**(1/(self.gamma-1))
        assert self.beta * self.compression_ratio > 1, f"Compression ratio is too high"

    def generate_processes(self):
        self.processes = []
        
        isothermal_expansion = Isothermal(n=self.n,
                                          V1=self.V1,
                                          T1=self.T_hot,
                                          diatomic=self.diatomic)
        isothermal_expansion.generate_data_from_dV(self.alpha*self.V1)
        self.processes.append(isothermal_expansion)

        adiabatic_expansion = Adiabatic(n=self.n,
                                        V1=isothermal_expansion.volume[-1],
                                        T1=isothermal_expansion.temperature[-1],
                                        diatomic=self.diatomic)
        adiabatic_expansion.generate_data_from_dT(self.T_cold)
        self.processes.append(adiabatic_expansion)

        # by the definition of the compression ratio, V3 should be equal to V1*R (where R is the compression ratio)
        V3 = adiabatic_expansion.volume[-1]
        assert abs(V3-self.V1*self.compression_ratio) < 1e-6, f"Something went wrong, V3 = {V3}, V1 = {self.V1}, beta = {self.beta}"

        isothermal_compression = Isothermal(n=self.n,
                                            V1=adiabatic_expansion.volume[-1],
                                            T1=self.T_cold,
                                            diatomic=self.diatomic)
        isothermal_compression.generate_data_from_dV(self.beta*V3)
        self.processes.append(isothermal_compression)

        adiabatic_compression = Adiabatic(n=self.n,
                                          V1=isothermal_compression.volume[-1],
                                          T1=isothermal_compression.temperature[-1],
                                          diatomic=self.diatomic)
        adiabatic_compression.generate_data_from_dT(self.T_hot)
        self.processes.append(adiabatic_compression)

        assert abs(adiabatic_compression.volume[-1]-self.V1) < allowed_error, f"Something went wrong, V4 = {adiabatic_compression.volume[-1]}, V = {self.V1}, alpha = {self.alpha}"
        assert abs(adiabatic_compression.temperature[-1]-self.T_hot) < allowed_error, f"Something went wrong, T4 = {adiabatic_compression.temperature[-1]}, T = {self.T}, alpha = {self.alpha}"
        assert abs(adiabatic_compression.pressure[-1]-self.P1) < allowed_error, f"Something went wrong, P4 = {adiabatic_compression.pressure[-1]}, P = {self.P1}, alpha = {self.alpha}"
        return self.processes
class Petrol(BaseCycle):
    pass
class Diesel(BaseCycle):
    pass
class Otto(BaseCycle): 
    pass
    def __init__(self,compression_ratio,T_cold,T_hot,P1=None,V1=None,n=None,monatomic=False,diatomic=False,specific_heat=None,molar_mass=None,diameter=1e-10):
        super().__init__(P1=P1,V1=V1,n=n,
                         compression_ratio=compression_ratio,
                         T_hot=T_hot,T_cold=T_cold,
                         monatomic=monatomic,diatomic=diatomic,
                         specific_heat=specific_heat,
                         molar_mass=molar_mass,
                         diameter=diameter)
        
        self.title = "Otto syklus"
        self._find_missing()
        self._calculate_alpha_beta()
        self.run_cycle()

    def _calculate_alpha_beta(self):
        self.alpha = 1/self.compression_ratio
        self.beta  = self.compression_ratio

    def generate_processes(self):
        self.processes = []

        adiabatic_compression = Adiabatic(n=self.n,T1=self.T_cold,V1=self.V1,diatomic=self.diatomic)
        adiabatic_compression.generate_data_from_dV(self.alpha*self.V1)
        self.processes.append(adiabatic_compression)

        isochoric_heat_addition = Isochoric(n=self.n,T1=self.T_cold,V1=adiabatic_compression.volume[-1],diatomic=self.diatomic)
        isochoric_heat_addition.generate_data_from_dT(self.T_hot)
        self.processes.append(isochoric_heat_addition)

        adiabatic_expansion = Adiabatic(n=self.n,T1=self.T_hot,V1=isochoric_heat_addition.volume[-1],diatomic=self.diatomic)
        adiabatic_expansion.generate_data_from_dV(self.beta*isochoric_heat_addition.volume[-1])
        self.processes.append(adiabatic_expansion)

        isochoric_heat_rejection = Isochoric(n=self.n,T1=self.T_hot,V1=adiabatic_expansion.volume[-1],diatomic=self.diatomic)
        isochoric_heat_rejection.generate_data_from_dT(self.T_cold)
        self.processes.append(isochoric_heat_rejection)

        return self.processes

class Brayton(BaseCycle):
    pass
class Stirling(BaseCycle):
    pass
class Ericsson(BaseCycle):
    pass
class Rankine(BaseCycle):
    pass
class Kalina(BaseCycle):
    pass