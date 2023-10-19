import tp_processes as tpp
import matplotlib.pyplot as plt
import numpy as np
import unittest

version = '0.9.0'

class Fluid(tpp.Static):
    def __init__(self,P=None,V=None,T=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.monatomic = monatomic; self.diatomic = diatomic
        self.processes = []
        self.heat = []
        self.work = []
        self.entropy = []
        self.time_taken = 0

    def __update__(self,process,time):
        self.temperature = np.append(self.temperature,process.temperature[1:])
        self.pressure = np.append(self.pressure,process.pressure[1:])
        self.volume = np.append(self.volume,process.volume[1:])
        self.time_taken += time
        process.time *= time + self.time_taken # each process starts at the end of the previous one
        self.processes.append(process)


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

class __plot_properties__:
    def __init__(self,axes,title):
        self.axes = axes
        self.title = title



def plot(fluid,display="PV"):
        axes = {"P":"Pressure [Pa]","V":"Volume [m^3]","T":"Temperature [K]","S":"Entropy [J/K]"}
        display_axes = []
        # we want to reverse the order of "display"
        print(display)
        for process in fluid.processes:
            display_values = []
            data = {"P":process.pressure,"V":process.volume,"T":process.temperature,"S":process.entropy}
            for char in display:
                if char not in "PVTS": raise ValueError("Invalid display value")
                display_values.append(data[char])
                display_axes.append(axes[char])
            if len(display_values)==1:
                plt.plot(np.linspace(0,process.time,tpp.K),display_values[0],label=process.title)
                display_axes.append("Time [s]")
            elif len(display_values)==2:
                plt.plot(display_values[0],display_values[1],label=process.title)
            elif len(display_values)==3:
                # we want a 3D plot
                pass
            else:
                raise ValueError("Invalid display value")
            
        display_axes.reverse()
        display_values.reverse()

        try: plt.xlabel(display_axes[0])
        except IndexError: pass
        try: plt.ylabel(display_axes[1])
        except IndexError: pass
        try: plt.title(display_axes[2])
        except IndexError: pass

        plt.title(fluid.name)


def show():
    plt.grid()
    plt.legend()
    plt.show()

O2 = Fluid(P=1e5, V=1e-3, T=500, gas='O2')

O2.isothermal(P=2e5)
O2.adiabatic(T=1200)

print(O2.processes[0].pressure)

plot(O2,display="P")
show()