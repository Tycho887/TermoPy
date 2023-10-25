import tp_processes as tpp
import matplotlib.pyplot as plt
import numpy as np
import unittest
import datetime


# we want to find the current date and time to use as a version number


version = '1.1.2'

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

class __cycle_base__(Fluid):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        super().__init__(P=P,V=V,T=T_hot,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        # we assume that the cycle starts at the hot temperature
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.compression_ratio = compression_ratio
        self.gas = self.name

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
        self.efficiency = abs(self.work/sum(heat_in))
        self.COP = 1/self.efficiency

        assert 0 <= self.efficiency <= 1, 'Efficiency is not between 0 and 1'

class Carnot(__cycle_base__):
    def __init__(self,T_hot,T_cold,compression_ratio,V,P=None,n=None,gas=None,monatomic=False,diatomic=False):
        """Otto cycle with compression ratio, compression ratio must be greater than 1, volume is starting volume"""
        super().__init__(T_hot,T_cold,compression_ratio,P=P,V=V/compression_ratio,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.title = "Carnot"
        
        self.alpha = self.compression_ratio * (self.T_cold/self.T_hot)**(1/(self.gamma-1))
        self.beta  = 1 / self.compression_ratio * (self.T_hot/self.T_cold)**(1/(self.gamma-1))

        assert self.alpha, f"Compression ratio is too low, minimum compression: {(self.T_cold/self.T_hot)**(1/(1-self.gamma)):.2f}"
        assert self.beta * self.compression_ratio > 1, f"Compression ratio is too high"
        
        self.__carnot_cycle__()
        self.get_efficiency()

    def __carnot_cycle__(self):
        self.isothermal(V=self.alpha*self.volume[-1],time=1)
        self.adiabatic(T=self.T_cold,time=1)
        self.isothermal(V=self.beta*self.volume[-1],time=1)
        self.adiabatic(T=self.T_hot,time=1)

class Stirling(__cycle_base__):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        """Stirling cycle with compression ratio, compression ratio must be greater than 1, volume is starting volume"""
        super().__init__(T_hot,T_cold,compression_ratio,P=P,V=V/compression_ratio,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.title = "Stirling"
        self.__stirling_cycle__()
        self.get_efficiency()

    def __stirling_cycle__(self):
        self.isothermal(V=self.compression_ratio*self.volume[-1],time=1)
        self.isochoric(T=self.T_cold,time=1)
        self.isothermal(V=self.volume[-1]/self.compression_ratio,time=1)
        self.isochoric(T=self.T_hot,time=1)

class Otto(__cycle_base__):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        """Otto cycle with compression ratio, compression ratio must be greater than 1, volume is starting volume"""
        super().__init__(T_hot,T_cold,compression_ratio,P=P,V=V/compression_ratio,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.title = "Otto"
        self.__otto_cycle__()
        self.get_efficiency()
    def __otto_cycle__(self):
        self.isochoric(T=self.T_cold,time=1)
        self.isothermal(V=self.compression_ratio*self.volume[-1],time=1)
        self.isochoric(T=self.T_hot,time=1)
        self.isothermal(V=self.volume[-1]/self.compression_ratio,time=1)

class Brayton(__cycle_base__):
    def __init__(self,T_hot,T_cold,compression_ratio,P=None,V=None,n=None,gas=None,monatomic=False,diatomic=False):
        """Brayton cycle with compression ratio, compression ratio must be greater than 1, volume is starting volume"""
        super().__init__(T_hot,T_cold,compression_ratio,P=P,V=V/compression_ratio,n=n,gas=gas,monatomic=monatomic,diatomic=diatomic)
        self.title = "Brayton"
        self.__brayton_cycle__()
        self.get_efficiency()
    
    def __brayton_cycle__(self):
        self.adiabatic(V=self.compression_ratio*self.volume[-1],time=1)
        self.isobaric(T=self.T_hot,time=1)
        self.adiabatic(V=self.volume[-1]/self.compression_ratio,time=1)
        self.isobaric(T=self.T_cold,time=1)

def __plot_3d__(cycle,display):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   
    axes = {"P":"Gas pressure [Pa]","V":"Piston volume [L]","T":"Gas temperature [K]","S":"Entropy of system [J/K]"}
    display_axes = []

    for process in cycle.processes:
        display_values = []
        data = {"P":process.pressure,"V":process.volume*1000,"T":process.temperature,"S":process.entropy}
        
        for char in display:
            if char not in "PVTS": raise ValueError("Invalid display value")
            display_values.append(data[char])
            display_axes.append(axes[char])
        
        ax.plot(display_values[0],display_values[1],display_values[2],label=process.title)
        ax.scatter(display_values[0][-1],display_values[1][-1],display_values[2][-1])
    
    ax.set_xlabel(display_axes[0])
    ax.set_ylabel(display_axes[1])
    ax.set_zlabel(display_axes[2])

def plot(cycle,display="PV"):

        axes = {"P":"Gas pressure [Pa]","V":"Piston volume [L]","T":"Gas temperature [K]","S":"Entropy of system [J/K]"}
        display_axes = []

        for process in cycle.processes:
            display_values = []
            data = {"P":process.pressure,"V":process.volume*1000,"T":process.temperature,"S":process.entropy}

            for char in display:
                if char not in "PVTS": raise ValueError("Invalid display value")
                display_values.append(data[char])
                display_axes.append(axes[char])

            if len(display_values)==1: 
                plt.plot(process.time,display_values[0],label=process.title)

            elif len(display_values)==2:
                plt.plot(display_values[0],display_values[1],label=process.title)
                plt.scatter(display_values[0][-1],display_values[1][-1])
                # we want to add a piece of text that displays the pressure and volume at the end of the process
                # plt.text(display_values[0][-1],display_values[1][-1],f"P: {display_values[1][-1]:.2f} Pa\nV: {display_values[0][-1]:.2f} L",transform=plt.gca().transData)
                # # we want to avoid the text overlapping other data, so we move it to the right if it is too close to the left
                # if display_values[0][-1] < 0.5*max(display_values[0]): plt.text(display_values[0][-1]+0.1*max(display_values[0]),display_values[1][-1],
                #                                                                 f"P: {display_values[1][-1]:.2f} Pa\nV: {display_values[0][-1]:.2f} L",
                #                                                                 transform=plt.gca().transData)
            
            elif len(display_values)==3: pass

            else: raise ValueError("Invalid display value")
            
        display_values.reverse()

        if len(display_values)==1:
            display_axes.insert(0,"Time [s]")
            plt.xlabel(display_axes[0])
            plt.ylabel(display_axes[1])
            # plt.text(0.5,0.5,f"Work done: {cycle.work:.2f} J\nHeat added: {cycle.heat_added:.2f} J\nHeat removed: {cycle.heat_removed:.2f} J\nEfficiency: {cycle.efficiency:.2f}",
            #          transform=plt.gca())
        elif len(display_values)==2:
            plt.xlabel(display_axes[0])
            plt.ylabel(display_axes[1])
            # plt.text(0.5,0.5,f"Work done: {cycle.work:.2f} J\nHeat added: {cycle.heat_added:.2f} J\nHeat removed: {cycle.heat_removed:.2f} J\nEfficiency: {cycle.efficiency:.2f}",
            #          transform=plt.gca().transAxes)
        elif len(display_values)==3:
            __plot_3d__(cycle,display)
        else:
            raise ValueError("Invalid display value")
        
        if isinstance(cycle,__cycle_base__): plt.title(f"{cycle.title} cycle with {cycle.gas.lower()} working fluid")
        elif isinstance(cycle,Fluid): plt.title(f"Working fluid: {cycle.name}")

def show(save=False,name="",dpi=512):
    plt.grid()
    plt.legend()
    if save: 
        now = datetime.datetime.now()
        date = f"{now.day}-{now.month}-{now.year}"
        if name == "": name = "plot"
        plt.savefig(f"{date}_{name}.png",dpi=dpi)
    plt.show()