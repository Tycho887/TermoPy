import termoPy as tp
import unittest
import numpy as np

atm = 101325
num_tests = 500; show_test = False
allowed_error = 1e-4



class TestProcess:
    def __init__(self,num_tests):
        self.num_tests = num_tests

    def __test_ideal_gas(self,process,type):
        assert (np.mean(process.consistency)) < allowed_error, f"Process does not satisfy the ideal gas law, inconsistency: {np.mean(process.consistency)}"
        return np.mean(process.consistency)

    def __test_first_law(self,process,type):
        # we want to check that the values are in accordance with the first law of thermodynamics
        # d
        W_on = process.work_done_on
        Q_in = process.heat_absorbed
        W_by = process.work_done_by
        Q_out = process.heat_released

        dE = process.internal_energy[-1] - process.internal_energy[0]

        error_1 = np.abs(W_on + Q_in - dE)
        error_2 = np.abs(W_by + Q_out + dE)

        assert error_1 < allowed_error and error_2 < allowed_error, f"The first law of thermodynamics is not satisfied for {process.title}, {type}, error_1: {error_1}, error_2: {error_2}"

        return np.mean([error_1,error_2])
    
    def _test_methods(self,process):
        ideal_gas_errors = []; first_law_errors = []
        for type in ["+dV","-dV","+dP","-dP","+dT","-dT"]:
            op = "*"
            if "+" in type: op = "*"
            elif "-" in type: op = "/"
            method = f"generate_data_from_d{type[2]}(process.{type[2]}1{op}10)"
            try:
                eval(f"process.{method}")
                ideal_gas_error = self.__test_ideal_gas(process,type)
                first_law_error = self.__test_first_law(process,type)
            except AttributeError: print(f"\tThe {type} method does not exist")
        return True, ideal_gas_error, first_law_error

    def base_test(self,process_generator):
        ideal_gas_errors = []; first_law_errors = []; name = ""
        for i in range(self.num_tests):
            P = np.random.uniform(0.01*tp.atm,100*tp.atm)
            V = np.random.uniform(1e-4,10)
            T = np.random.uniform(10,1000)
            process = process_generator(P,V,T)
            name = process.title
            passed, ideal_gas_error, first_law_error = self._test_methods(process)
            if passed:
                first_law_errors.append(first_law_error)
                ideal_gas_errors.append(ideal_gas_error)
        print(f"""\n{name} PASSED
\tAverage error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}
\tAverage error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}""")

    def isothermal(self):
        isothermal_generator = lambda P,V,T: tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True)
        self.base_test(isothermal_generator)

    def isochoric(self):
        isochoric_generator = lambda P,V,T: tp.Isochoric(P1=P,V1=V,T1=T,monatomic=True)
        self.base_test(isochoric_generator)

    def isobaric(self):
        isobaric_generator = lambda P,V,T: tp.Isobaric(P1=P,V1=V,T1=T,monatomic=True)
        self.base_test(isobaric_generator)

    def adiabatic(self):
        adiabatic_generator = lambda P,V,T: tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
        self.base_test(adiabatic_generator)

# we want to rewrite the test as a unittest


class Test_TermoPy_processes(unittest.TestCase):
    
    def test_Isothermal(self):
        print("\nRunning isothermal process test")
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

    def test_Isochoric(self):
        print("\nRunning isochoric process test")
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

    def test_Isobaric(self):
        print("\nIsobaric process test")
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
            
    def test_Adiabatic(self):
        print("\nRunning adiabatic process test")
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

    def test_carnot(self):
        print("\nRunning Carnot cycle test")
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
    

if __name__ == "__main__":
    # test = TestProcess(100)
    # test.isothermal()
    # test.isochoric()
    # test.isobaric()
    # test.adiabatic()
    unittest.main()

