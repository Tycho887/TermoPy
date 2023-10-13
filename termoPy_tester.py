import termoPy as tp
import unittest
import numpy as np

atm = 101325

def __chceck_consisency(process,type):
#        if process.title == "Adiabatisk prosess":
        assert (np.mean(process.consistency)) < 1e-10, "Consistency is not zero"
        print(f"{type} method is consistent. max inconsistency: {np.max(process.consistency):.2e}")

def test_process_methods(process):
    pass

def __test_processes(processes):
    for process in processes:
        print("Testing",process.title)
        try:
            process.generate_data_from_dV(process.V1*10)
            __chceck_consisency(process,"+dV")
            process.generate_data_from_dV(process.volume[-1]*0.01)
            __chceck_consisency(process,"-dV")
        except: assert False, f"The from_dV method does not work as intended on {process.title}"
        try:
            process.generate_data_from_dP(process.P1*10)
            __chceck_consisency(process,"+dP")
            process.generate_data_from_dP(process.pressure[-1]*0.01)
            __chceck_consisency(process,"-dP")
        except: assert False, f"The from_dP method does not work as intended on {process.title}"
        try: 
            process.generate_data_from_dT(process.T1*10)
            __chceck_consisency(process,"+dT")
            process.generate_data_from_dT(process.temperature[-1]*0.01)
            __chceck_consisency(process,"-dT")
        except: assert False, f"The from_dT method does not work as intended on {process.title}"



if __name__ == "__main__":
    IS = {"P":1*atm,"V":0.0470312,"T":300,"n":1}
    processes = [tp.Isothermal(n=IS["n"],V1=IS["V"],P1=IS["P"],monatomic=True),
                 tp.Isochoric(n=IS["n"],T1=IS["T"],P1=IS["P"],monatomic=True),
                 tp.Isobaric(n=IS["n"],V1=IS["V"],T1=IS["T"],monatomic=True),
                 tp.Adiabatic(n=IS["n"],V1=IS["V"],T1=IS["T"],monatomic=True)]
    __test_processes(processes)