import termoPy as TP

atm = 1.01325e5

T_high = 600
T_low = 300

n = 1
P1 = 1.0*atm

adiabatic_expansion_dV = TP.Adiabatic(n=n,P1=P1,T1=T_high,diatomic=True)

test = TP.IdealGas(n=n,P1=P1,T1=T_high,diatomic=True)
test.generate_data_from_dT(test.T1*1.25)

isothermal_test = TP.Isothermal(n=n,P1=P1,T1=T_high,diatomic=True)
isothermal_test.generate_data_from_dT(isothermal_test.T1*10)

print(isothermal_test.consistency)