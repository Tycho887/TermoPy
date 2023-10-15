import termoPy as TP
import matplotlib.pyplot as plt
import numpy as np

atm = 101325

gamma = 1.4
n = 1
compression_ratio = 1.5

T_hot = 300
T_cold = 200

P1 = 1*atm
V1 = n*TP.R*T_hot/P1
#%%

def find_carnot_constants(gamma,compression_ratio,T_hot,T_cold):
    alpha = compression_ratio * (T_cold/T_hot)**(1/(gamma-1))
    assert alpha, f"Compression ratio is too low, minimum compression: {(T_cold/T_hot)**(1/(1-gamma)):.2f}"

    beta  = 1 / compression_ratio * (T_hot/T_cold)**(1/(gamma-1))
    assert beta * compression_ratio > 1, f"Compression ratio is too high"
    return alpha,beta

alpha,beta = find_carnot_constants(gamma,compression_ratio,T_hot,T_cold)

print(f"alpha = {alpha:.2f}, beta = {beta:.2e}")

#%%

isothermal_expansion = TP.Isothermal(n=n,V1=V1,T1=T_hot,diatomic=True)
isothermal_expansion.generate_data_from_dV(alpha*V1)

V2 = isothermal_expansion.volume[-1]

adiabatic_expansion = TP.Adiabatic(n=n,V1=V2,T1=T_hot,diatomic=True)
adiabatic_expansion.generate_data_from_dT(T_cold)

V3 = adiabatic_expansion.volume[-1]

print(V3-V1*compression_ratio)

isothermal_compression = TP.Isothermal(n=n,V1=V3,T1=T_cold,diatomic=True)
isothermal_compression.generate_data_from_dV(V3*beta)

V4 = isothermal_compression.volume[-1]

adiabatic_compression = TP.Adiabatic(n=n,V1=V4,T1=T_cold,diatomic=True)
adiabatic_compression.generate_data_from_dT(T_hot)


#%% plot the processes

plt.plot(isothermal_expansion.volume,isothermal_expansion.pressure,label="Isothermal expansion")
plt.plot(adiabatic_expansion.volume,adiabatic_expansion.pressure,label="Adiabatic expansion")
plt.plot(isothermal_compression.volume,isothermal_compression.pressure,label="Isothermal compression")
plt.plot(adiabatic_compression.volume,adiabatic_compression.pressure,label="Adiabatic compression")
plt.legend()
plt.xlabel("Volume [m^3]")
plt.ylabel("Pressure [Pa]")

plt.show()