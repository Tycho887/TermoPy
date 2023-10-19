import tp_processes as tpp
import tp_cycles as tpc
import matplotlib.pyplot as plt
import numpy as np
import unittest

version = '1.0.0'

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
class Fluid(tpp.Static):
    def __init__(self,P=None,V=None,T=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.monatomic = monatomic; self.diatomic = diatomic
        self.processes = []
        self.heat = []
        self.work = []
        self.entropy = []
        self.time_taken = 0
        self.heat_added = 0
        self.heat_removed = 0

    def __update__(self,process,time):
        self.temperature = np.append(self.temperature,process.temperature[1:])
        self.pressure = np.append(self.pressure,process.pressure[1:])
        self.volume = np.append(self.volume,process.volume[1:])
        process.time *= time
        process.time += self.time_taken # each process starts at the end of the previous one
        self.time_taken += time
        self.processes.append(process)
        if process.heat[-1] > 0:
            self.heat_added += process.heat
        else:
            self.heat_removed += process.heat


    def isothermal(self, P=None, V=None,time=1):

        process = tpp.Isothermal(P=self.pressure[-1], 
                                 T = self.temperature[-1], 
                                 V=self.volume[-1], 
                                 n=self.n, 
                                 gas=self.name, 
                                 monatomic=self.monatomic, 
                                 diatomic=self.diatomic)
        process.final(P=P,V=V)
        self.__update__(process,time)

    def isobaric(self, V=None, T=None, time=1):
        process = tpp.Isobaric(P=self.pressure[-1], 
                               T = self.temperature[-1], 
                               V=self.volume[-1], 
                               n=self.n, 
                               gas=self.name, 
                               monatomic=self.monatomic, 
                               diatomic=self.diatomic)
        process.final(V=V,T=T)
        self.__update__(process,time)

    def isochoric(self, P=None, T=None, time= 1):
        process = tpp.Isochoric(P=self.pressure[-1], 
                                T = self.temperature[-1], 
                                V=self.volume[-1], 
                                n=self.n, 
                                gas=self.name, 
                                monatomic=self.monatomic, 
                                diatomic=self.diatomic)
        process.final(P=P,T=T)
        self.__update__(process,time)

    def adiabatic(self, P=None, V=None, T=None, time=1):
        process = tpp.Adiabatic(P=self.pressure[-1], 
                                T = self.temperature[-1], 
                                V=self.volume[-1], 
                                n=self.n, 
                                gas=self.name, 
                                monatomic=self.monatomic, 
                                diatomic=self.diatomic)
        process.final(P=P,V=V,T=T)
        self.__update__(process,time)

class Carnot(tpc.cycle_base):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(T_hot,T_cold,compression_ratio,P=P,V=V,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.name = "Carnot"
        self.processes = []
        
        self.alpha = self.compression_ratio * (self.T_cold/self.T_hot)**(1/(self.gamma-1))
        self.beta  = 1 / self.compression_ratio * (self.T_hot/self.T_cold)**(1/(self.gamma-1))

        assert self.alpha, f"Compression ratio is too low, minimum compression: {(self.T_cold/self.T_hot)**(1/(1-self.gamma)):.2f}"
        assert self.beta * self.compression_ratio > 1, f"Compression ratio is too high"
        
        self.__carnot_cycle__()
        self.get_efficiency()

    def __carnot_cycle__(self):
        isothermal_expansion = tpp.Isothermal(n=self.n,
                                          V=self.volume[-1],
                                          T=self.T_hot,
                                          diatomic=self.diatomic,
                                          monatomic=self.monatomic,
                                          gas=self.gas)
        isothermal_expansion.final(V=self.alpha*self.volume[-1])
        self.processes.append(isothermal_expansion)

        adiabatic_expansion = tpp.Adiabatic(n=self.n,
                                          V=isothermal_expansion.volume[-1],
                                          T=isothermal_expansion.temperature[-1],
                                          diatomic=self.diatomic,
                                          monatomic=self.monatomic,
                                          gas=self.gas)
        adiabatic_expansion.final(T=self.T_cold)
        self.processes.append(adiabatic_expansion)

        # by the definition of the compression ratio, V3 should be equal to V1*R (where R is the compression ratio)
        V3 = adiabatic_expansion.volume[-1]
        assert abs(V3-self.volume[-1]*self.compression_ratio) < tpp.allowed_error, f"Something went wrong, V3 = {V3}, V1 = {self.volume[-1]}, beta = {self.beta}"

        isothermal_compression = tpp.Isothermal(n=self.n,
                                                V=adiabatic_expansion.volume[-1],
                                                T=self.T_cold,
                                                diatomic=self.diatomic,
                                                monatomic=self.monatomic,
                                                gas=self.gas)
        isothermal_compression.final(V=self.beta*V3)
        self.processes.append(isothermal_compression)

        adiabatic_compression = tpp.Adiabatic(n=self.n,
                                          V=isothermal_compression.volume[-1],
                                          T=isothermal_compression.temperature[-1],
                                          diatomic=self.diatomic,
                                          monatomic=self.monatomic,
                                          gas=self.gas)
        adiabatic_compression.final(T=self.T_hot)
        self.processes.append(adiabatic_compression)

def plot(cycle,display="PV"):
        axes = {"P":"Pressure [Pa]","V":"Volume [m^3]","T":"Temperature [K]","S":"Entropy [J/K]"}
        display_axes = []
        # we want to reverse the order of "display"
        for process in cycle.processes:
            display_values = []
            data = {"P":process.pressure,"V":process.volume,"T":process.temperature,"S":process.entropy}
            for char in display:
                if char not in "PVTS": raise ValueError("Invalid display value")
                display_values.append(data[char])
                display_axes.append(axes[char])
            if len(display_values)==1:
                plt.plot(process.time,display_values[0],label=process.title)
                # we want to add the axis name "Time [s]" to the beginning of the list
                display_axes.insert(0,"Time [s]")
            elif len(display_values)==2:
                plt.plot(display_values[0],display_values[1],label=process.title)
                plt.scatter(display_values[0][-1],display_values[1][-1],label=process.title + " final state")
            elif len(display_values)==3:
                ax.plot(display_values[0],display_values[1],display_values[2],label=process.title)
                ax.scatter(display_values[0][-1],display_values[1][-1],display_values[2][-1],label=process.title + " final state")
            else:
                raise ValueError("Invalid display value")
            
        display_values.reverse()

        if len(display_values)==1 or len(display_values)==2:
            plt.xlabel(display_axes[0])
            plt.ylabel(display_axes[1])
            plt.text(2, 7, f'Work Done: {cycle.work} J')
            plt.text(2, 8, f'Heat Added: {cycle.heat_added} J')
            plt.text(2, 9, f'Heat Removed: {cycle.heat_removed} J')
        elif len(display_values)==3:
            print(display_axes)
            ax.set_xlabel(display_axes[0])
            ax.set_ylabel(display_axes[1])
            ax.set_zlabel(display_axes[2])
            ax.text(2, 7, 10, f'Work Done: {cycle.work} J')
            ax.text(2, 8, 10, f'Heat Added: {cycle.heat_added} J')
            ax.text(2, 9, 10, f'Heat Removed: {cycle.heat_removed} J')
        else:
            raise ValueError("Invalid display value")
        
        if isinstance(cycle,Fluid): plt.title(f"Working fluid: {cycle.name}")
        else: plt.title(f"{cycle.name} cycle with {cycle.gas.lower()} working fluid")



def show():
    plt.grid()
    plt.legend()
    plt.show()

O2 = Fluid(P=1e5, V=1e-3, T=500, gas='O2')

O2.isothermal(P=2e5)
O2.adiabatic(T=600)
O2.isobaric(V=1e-3)

cycle1 = Carnot(T_hot=400,T_cold=300,compression_ratio=2,P=1e5,V=1e-3,gas='Krypton')

plot(cycle1,display="P")

show()