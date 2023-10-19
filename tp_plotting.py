import tp_processes as tpp
import matplotlib.pyplot as plt
import numpy as np
import termopy_main as tp

version = '0.0.1'

class plotter:
    def __init__(self,fluid,display="PV",save=False):
        self.fluid = fluid
        self.display = list(display)
        self.save = save
        # we want to check that the display values are in the list "PVTS"
        for i in range(len(self.display)):
            if self.display[i] not in "PVTS": raise ValueError("Invalid display value")
        
    def find_process_type(process):
        if isinstance(process,tpp.Isochoric):
            if process.temperature[-1] < process.temperature[0]: return "Isochoric Cooling"
            else: return "Isochoric Heating"
        else:
            if process.volume[-1] < process.volume[0]: return f"{process.title} Compression"
            else: return f"{process.title} Expansion"
        

    def plot(self):
        for process in self.fluid.processes:
            data = {"P":process.pressure,"V":process.volume,"T":process.temperature,"S":process.entropy}
            display_values = []
            for i in self.display:
                display_values.append(data[i])
    



O2 = tp.Fluid(P=1e5, V=1e-3, T=500, gas='O2')
O2.isothermal(P=2e5)
O2.adiabatic(P=1e5)

axis = plotter(O2,display="PV",save=False)
print(axis.display)