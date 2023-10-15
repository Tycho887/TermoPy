import termoPy as TP
import matplotlib.pyplot as plt

atm = 1.01325e5
C = lambda c: c + 273.15

# system_states = [{"P":1*atm,    "V":1.0,    "T":300,    "n":1},
#                  {"P":None,     "V":3.0,    "T":None,   "n":1},
#                  {"P":None,     "V":None,   "T":600,    "n":1},
#                  {"P":2*atm,    "V":None,   "T":None,   "n":1}]

# cycle = TP.Carnot(system_states)

# #print(cycle.system_types)
# print(cycle.processes)


P1 = 1.0*atm
n = 1

T_hot = 400; T1 = T_hot
T_cold = 273.15

gamma = 1.4

V1 = n*TP.R*T_hot/P1


compression_ratio = 4

def generate_carnot_values(V1,ratio,T_h,T_c,gamma=1.4):
    alpha = 1 / ratio * (T_h/T_c)**(gamma/(1-gamma))
    beta  = ratio*(T_h/T_c)**(gamma/(gamma-1)**2)
    return alpha, beta, ratio

alpha, beta, ratio = generate_carnot_values(V1,compression_ratio,T_hot,T_cold)


isothermal_expansion = TP.Isothermal(n=n,P1=P1,T1=T_hot,diatomic=True)
isothermal_expansion.generate_data_from_dV(isothermal_expansion.V1*alpha)

V2 = isothermal_expansion.volume[-1]
P2 = isothermal_expansion.pressure[-1]
T2 = isothermal_expansion.temperature[-1]



#assert V2 - V1*alpha < 1e-6

adiabatic_expansion = TP.Adiabatic(n=n,P1=P2,T1=T2,diatomic=True)
adiabatic_expansion.generate_data_from_dV(V1/ratio)

adiabatic_expansion2 = TP.Adiabatic(n=n,P1=P2,T1=T2,diatomic=True)
adiabatic_expansion2.generate_data_from_dT(T_cold)

V3 = adiabatic_expansion.volume[-1]
P3 = adiabatic_expansion.pressure[-1]
T3 = adiabatic_expansion.temperature[-1]



# we want to plot the iso and adiabatic curves in the same plot

#plt.plot(isothermal_expansion.volume, isothermal_expansion.pressure, label="Isothermal Expansion")
plt.plot(adiabatic_expansion.volume, adiabatic_expansion.pressure, label="Adiabatic Expansion")
plt.plot(adiabatic_expansion2.volume, adiabatic_expansion2.pressure, label="Adiabatic Expansion2")
plt.legend()
plt.xlabel("Volume (m^3)")
plt.ylabel("Pressure (Pa)")
plt.grid()
plt.show()

print(f"""
Alpha = {alpha:.3f}
Beta = {beta:.3f}
Ratio = {ratio:.3f}
P1 = {P1:.3e}, T1 = {T1:.3e}, V1 = {V1:.3e}
Predicted V2 = {V1*alpha:.3e}, Actual V2 = {V2:.3e}
Predicted P2 = {P1/alpha:.3e}, Actual P2 = {P2:.3e}
Predicted T2 = {T1:.3e}, Actual T2 = {T2:.3e}
Predicted V3 = {V1/ratio:.3e}, Actual V3 = {V3:.3e}
Predicted P3 = {P1*ratio*(T_cold/T_hot):.3e}, Actual P3 = {P3:.3e}
Predicted T3 = {T_cold:.3e}, Actual T3 = {T3:.3e}
""")

# There is a problem with predicting the temperature of the adiabatic expansion




# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(isothermal_expansion.pressure, isothermal_expansion.volume, isothermal_expansion.temperature, label="Isothermal Expansion")
# ax.plot(adiabatic_expansion.pressure, adiabatic_expansion.volume, adiabatic_expansion.temperature, label="Adiabatic Expansion")
# # ax.plot(isothermal_compression.pressure, isothermal_compression.volume, isothermal_compression.temperature, label="Isothermal Compression")
# # ax.plot(adiabatic_compression.pressure, adiabatic_compression.volume, adiabatic_compression.temperature, label="Adiabatic Compression")
# ax.legend()
# ax.set_xlabel("Pressure (Pa)")
# ax.set_ylabel("Volume (m^3)")
# ax.set_zlabel("Temperature (K)")

# print(adiabatic_expansion.consistency)
# plt.show()
