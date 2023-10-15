import termoPy as TP
import matplotlib.pyplot as plt
import numpy as np


atm = 1.01325e5
C = lambda c: c + 273.15

T_high = 600
T_low = 300

n = 1
P1 = 1.0*atm

adiabatic_expansion_dV = TP.Adiabatic(n=n,P1=P1,T1=T_high,diatomic=True)
adiabatic_expansion_dV.generate_data_from_dV(2.50*adiabatic_expansion_dV.V1)

adiabatic_expansion_dP = TP.Adiabatic(n=n,P1=P1,T1=T_high,diatomic=True)
adiabatic_expansion_dP.generate_data_from_dP(0.5*P1)

adiabatic_expansion_dT = TP.Adiabatic(n=n,P1=P1,T1=T_high,diatomic=True)
adiabatic_expansion_dT.generate_data_from_dT(T_low)


print(np.mean(adiabatic_expansion_dV.consistency))
print(np.mean(adiabatic_expansion_dP.consistency))
print(np.mean(adiabatic_expansion_dT.consistency))
