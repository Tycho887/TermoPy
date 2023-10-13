atm = 101325
import termoPy as TP
import matplotlib.pyplot as plt
IS = {"P":1*atm,"V":0.0470312,"T":300,"n":1}
A = TP.Isothermal(n=IS["n"],T1=IS["T"],P1=IS["P"],monatomic=True)
B = TP.Isochoric(n=IS["n"],V1=IS["V"],P1=IS["P"],monatomic=True)
C = TP.Isobaric(n=IS["n"],V1=IS["V"],P1=IS["P"],monatomic=True)
D = TP.Adiabatic(n=IS["n"],V1=IS["V"],T1=IS["T"],monatomic=True)

D.generate_data_from_dP(2*IS["P"])
C.generate_data_from_dT(1.25*IS["T"])

D.plot_ST()
#C.plot_VT(save=True)

print(C.temperature)
print(D.heat_absorbed)
print(D.work_done_by)

plt.show()
