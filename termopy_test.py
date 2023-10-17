import tp_processes as tp
import unittest
import numpy as np

atm = 101325
num_tests = 10; show_test = False
allowed_error = 1e-3

# we want to rewrite the test as a unittest

class Initial_state:
    P = np.random.uniform(0.1*tp.atm,10*tp.atm)
    V = np.random.uniform(1e-4,10)
    T = np.random.uniform(10,1000)
    n = P*V/(tp.R*T)
    monatomic = np.random.choice([True])

def standard_line(type, first_law_errors, ideal_gas_errors):
    return f"\nTesting data generation from {type} change: PASSED\n\n\tFirst law inconsistency: {np.max(first_law_errors)} \n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)}"

def header(text):
    return f"\n{'-'*int((len(text)-1)/2)}{text}{'-'*int((len(text)-1)/2)}"

class test_TermoPy(unittest.TestCase):
    
    def assertions(self,process):
        self.assertTrue(process.is_ideal_gas(),f"P: {process.P}, V: {process.V}, T: {process.T}, n: {process.n}")
        self.assertTrue(process.is_first_law_satisfied(),f"P: {process.P}, V: {process.V}, T: {process.T}, n: {process.n}")
        self.assertTrue(process.work_done_on + process.work_done_by < allowed_error)
        self.assertTrue(process.heat_absorbed + process.heat_released < allowed_error)
    
    def process_loop_methods(self,get_process):
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, mono = case.P, case.V, case.T, case.monatomic
            # we want to choose a random number between 0 and 100)
            process = get_process(P=P,V=V,T=T,monatomic=mono)
            process.final(P = P*np.random.uniform(0,100))
            self.assertions(process)
            process.final(P = P/np.random.uniform(0,100))
            self.assertions(process)
            process.final(V = V*np.random.uniform(0,100))
            self.assertions(process)
            process.final(V = V/np.random.uniform(0,100))
            self.assertions(process)
            process.final(T = T*np.random.uniform(0,100))
            self.assertions(process)
            process.final(T = T/np.random.uniform(0,100))
            self.assertions(process)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))
        return first_law_errors, ideal_gas_errors

    def test_Isothermal(self):
        print(header("Running isothermal test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: tp.Isothermal(P=P,V=V,T=T,monatomic=monatomic))
        print(standard_line("isothermal",Error_first_law,Error_ideal_gas_law))

    def test_Isobaric(self):
        print(header("Running isobaric test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: tp.Isobaric(P=P,V=V,T=T,monatomic=monatomic))
        print(standard_line("isobaric",Error_first_law,Error_ideal_gas_law))

    def test_Isochoric(self):
        print(header("Running isochoric test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: tp.Isochoric(P=P,V=V,T=T,monatomic=monatomic))
        print(standard_line("isochoric",Error_first_law,Error_ideal_gas_law))

    def test_Adiabatic(self):
        print(header("Running adiabatic test"))
        Error_first_law, Error_ideal_gas_law = self.process_loop_methods(lambda P,V,T,monatomic: tp.Adiabatic(P=P,V=V,T=T,monatomic=monatomic))
        print(standard_line("adiabatic",Error_first_law,Error_ideal_gas_law))



        # print(header("Running Carnot cycle test"))
        # first_law_errors = []; ideal_gas_errors = []; efficiency_errors = []
        # for i in range(num_tests):
        #     state = Initial_state()
        #     P, V, T, n = state.P, state.V, state.T, state.n
        #     T_cold = np.random.uniform(10,1000)
        #     T_hot = np.random.uniform(T_cold,2000)
        #     diatomic = np.random.choice([True,False])
        #     monatomic = not diatomic

        #     if diatomic: gamma = 1.4
        #     elif monatomic: gamma = 1.67

        #     lower_compression_limit = (T_cold/T_hot)**(1/(1-gamma))

        #     compression_ratio = np.random.uniform(2*lower_compression_limit,100*lower_compression_limit)

        #     assert compression_ratio > lower_compression_limit
        #     assert T_hot > T_cold

        #     assert compression_ratio > (T_cold/T_hot)**(1/(1-gamma))

        #     Carnot_cycle = tp.Carnot(P1=P,V1=V,compression_ratio=compression_ratio,T_cold=T_cold,T_hot=T_hot,monatomic=monatomic,diatomic=diatomic)

        #     self.assertTrue(Carnot_cycle.cycle_is_ideal_gas, f"Cycle is not ideal gas: {Carnot_cycle.cycle_is_ideal_gas}")
        #     self.assertTrue(Carnot_cycle.cycle_is_first_law_satisfied, f"Cycle is not first law satisfied: {Carnot_cycle.cycle_is_first_law_satisfied}")

        #     self.assertTrue(Carnot_cycle.efficiency < 1, f"Efficiency is not less than 1: {Carnot_cycle.efficiency}")
        #     self.assertTrue(Carnot_cycle.efficiency > 0, f"Efficiency is not greater than 0: {Carnot_cycle.efficiency}")

        #     self.assertTrue(np.abs(Carnot_cycle.efficiency - Carnot_cycle.theoretical_efficiency) < allowed_error, f"Efficiency is not equal to theoretical efficiency: {Carnot_cycle.efficiency} vs {Carnot_cycle.theoretical_efficiency}")

        #     first_law_error = [process.first_law_consistency for process in Carnot_cycle.processes]
        #     ideal_gas_error = [np.max(process.ideal_gas_law_consistency) for process in Carnot_cycle.processes]

        #     first_law_errors.append(np.max(first_law_error))
        #     ideal_gas_errors.append(np.max(ideal_gas_error))

        #     efficiency_errors.append(np.abs(Carnot_cycle.efficiency - Carnot_cycle.theoretical_efficiency))

        #     if show_test: print(f"Test {i} passed")

        # print("\nTesting data from Carnot Cycle: PASSED")
        # print(f"\n\tFirst law inconsistency: {np.max(first_law_errors)} \n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)} \n\tEfficiency error: {np.max(efficiency_errors)}")



if __name__ == "__main__":
    # we want to run the tests in alhabetical order
    unittest.main()
