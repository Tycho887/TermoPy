import termoPy as tp
import unittest
import numpy as np

atm = 101325
num_tests = 500; show_test = False
allowed_error = 1e-4
class Test_TermoPy_processes(unittest.TestCase):
    
    def test_Isothermal(self):
        print("\nRunning isothermal process test\n")
        for i in range(num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T = np.random.uniform(10,1000)
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True)
            
            process.generate_data_from_dP(P/10)
            process.generate_data_from_dP(P*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            
            process.generate_data_from_dV(process.volume[-1]/10)
            process.generate_data_from_dV(process.volume[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            # we want to check that the initial and final temperatures are the same

            self.assertTrue(np.abs(process.temperature[0] - process.temperature[-1]) < allowed_error)

            if show_test: print(f"Test {i} passed")
        print("\tPASSED")


    def test_Isochoric(self):
        print("\nRunning isochoric process test\n")
        for i in range(num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T = np.random.uniform(10,1000)
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=True)


            process.generate_data_from_dP(P/10)
            process.generate_data_from_dP(process.pressure[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            process.generate_data_from_dT(process.temperature[-1]/10)
            process.generate_data_from_dT(process.temperature[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            # we want to check that the initial and final volumes are the same

            self.assertTrue(np.abs(process.volume[0] - process.volume[-1]) < allowed_error)

            if show_test: print(f"Test {i} passed")
        print("\tPASSED")

    def test_Isobaric(self):
        print("\nIsobaric process test\n")
        for i in range(num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T = np.random.uniform(10,1000)
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=True)
            
            process.generate_data_from_dV(V/10)
            process.generate_data_from_dV(process.volume[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            
            process.generate_data_from_dT(process.temperature[-1]/10)
            process.generate_data_from_dT(process.temperature[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            # we want to check that the initial and final pressures are the same

            self.assertTrue(np.abs(process.pressure[0] - process.pressure[-1]) < allowed_error)

            if show_test: print(f"Test {i} passed")
        print("\tPASSED")

    def test_Adiabatic(self):
        print("\nRunning adiabatic process test\n")
        for i in range(num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T = np.random.uniform(10,1000)
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            
            process.generate_data_from_dV(V/10)
            process.generate_data_from_dV(process.volume[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            
            process.generate_data_from_dP(process.pressure[-1]/10)
            process.generate_data_from_dP(process.pressure[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            process.generate_data_from_dT(process.temperature[-1]/10)
            process.generate_data_from_dT(process.temperature[-1]*20)

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())

            if show_test: print(f"Test {i} passed")
        print("\tPASSED")

    def test_carnot(self):
        print("\nRunning Carnot cycle test\n")
        for i in range(num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T_cold = np.random.uniform(10,1000)
            T_hot = np.random.uniform(T_cold,2000)
            diatomic = np.random.choice([True,False])
            monatomic = not diatomic

            if diatomic: gamma = 1.4
            elif monatomic: gamma = 1.67

            lower_compression_limit = (T_cold/T_hot)**(1/(1-gamma))

            compression_ratio = np.random.uniform(2*lower_compression_limit,100*lower_compression_limit)

            assert compression_ratio > lower_compression_limit
            assert T_hot > T_cold

            assert compression_ratio > (T_cold/T_hot)**(1/(1-gamma))

            Carnot_cycle = tp.Carnot(P1=P,V1=V,compression_ratio=compression_ratio,T_cold=T_cold,T_hot=T_hot,monatomic=monatomic,diatomic=diatomic)

            self.assertTrue(Carnot_cycle.cycle_is_ideal_gas)
            self.assertTrue(Carnot_cycle.cycle_is_first_law_satisfied)

            self.assertTrue(Carnot_cycle.efficiency < 1)
            self.assertTrue(Carnot_cycle.efficiency > 0)

            self.assertTrue(np.abs(Carnot_cycle.efficiency - Carnot_cycle.theoretical_efficiency) < allowed_error)

            if show_test: print(f"Test {i} passed")
        print("\tPASSED")
    

if __name__ == "__main__":
    unittest.main()

