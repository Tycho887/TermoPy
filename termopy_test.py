import termopy_old as tp
import unittest
import numpy as np

atm = 101325
num_tests = 100; show_test = False
allowed_error = 1e-6

# we want to rewrite the test as a unittest

class Initial_state:
    P = np.random.uniform(0.01*tp.atm,100*tp.atm)
    V = np.random.uniform(1e-4,10)
    T = np.random.uniform(10,1000)
    n = P*V/(tp.R*T)
    monatomic = np.random.choice([True,False])

def standard_line(type, first_law_errors, ideal_gas_errors):
    return f"\nTesting data generation from {type} change: PASSED\n\n\tFirst law inconsistency: {np.max(first_law_errors)} \n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)}"

def header(text):
    return f"\n{'-'*int((len(text)-1)/2)}{text}{'-'*int((len(text)-1)/2)}"

class Test_TermoPy_processes(unittest.TestCase):

    def test_Isothermal_expansion(self):
        print(header("Running isothermal expansion test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dP(P/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_on - n * tp.R * T * np.log(V/process.volume[-1]) < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("pressure",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dV(V*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_on - n * tp.R * T * np.log(V/process.volume[-1]) < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))

    def test_Isothermal_compression(self):
        print(header("Running isothermal compression test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dP(P*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_on - n * tp.R * T * np.log(V/process.volume[-1]) < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dV(V/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_on - n * tp.R * T * np.log(V/process.volume[-1]) < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))

    def test_isobaric_expansion(self):
        print(header("Running isobaric expansion test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dV(V*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_by - P * (process.volume[-1] - V) < allowed_error, f"Work done on: {process.work_done_on}, expected: {P * (process.volume[-1] - V)}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dT(T*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_by - P * (process.volume[-1] - V) < allowed_error, f"Work done on: {process.work_done_on}, expected: {P * (process.volume[-1] - V)}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))

    def test_isobaric_compression(self):
        print(header("Running isobaric compression test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dV(V/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_by - P * (process.volume[-1] - V) < allowed_error, f"Work done on: {process.work_done_on}, expected: {P * (process.volume[-1] - V)}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dT(T/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.work_done_by - P * (process.volume[-1] - V) < allowed_error, f"Work done on: {process.work_done_on}, expected: {P * (process.volume[-1] - V)}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))

    def test_isochoric_heat_addition(self):
        print(header("Running isochoric heat addition test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dT(T*10)

            Q_in = process.heat_absorbed

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed - Q_in < allowed_error, f"Heat absorbed: {process.heat_absorbed}, expected: {Q_in}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dP(P*10)

            Q_in = process.heat_absorbed

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed - Q_in < allowed_error, f"Heat absorbed: {process.heat_absorbed}, expected: {Q_in}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("prssure",first_law_errors,ideal_gas_errors))

    def test_isochoric_heat_removal(self):
        print(header("Running isochoric heat removal test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dT(T/10)

            Q_out = process.heat_released

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_released - Q_out < allowed_error, f"Heat released: {process.heat_released}, expected: {Q_out}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dP(P/10)

            Q_out = process.heat_released

            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_released - Q_out < allowed_error, f"Heat released: {process.heat_released}, expected: {Q_out}")
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed") 
        print(standard_line("pressure",first_law_errors,ideal_gas_errors))

    def test_adiabatic_expansion(self):
        print(header("Running adiabatic expansion test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, case.monatomic
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dP(P/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("pressure",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, True
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dV(V*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n, mono = case.P, case.V, case.T, case.n, True
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=mono)
            process.generate_data_from_dT(T*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))

            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))  
    
    def test_adiabatic_compression(self):
        print(header("Running adiabatic compression test"))
        first_law_errors = []; ideal_gas_errors = []
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n = case.P, case.V, case.T, case.n
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dP(P*10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))
            if show_test: print(f"Test {i} passed")
        print(standard_line("pressure",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n = case.P, case.V, case.T, case.n
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dV(V/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))
            if show_test: print(f"Test {i} passed")
        print(standard_line("volume",first_law_errors,ideal_gas_errors))
        for i in range(num_tests):
            case = Initial_state()
            P, V, T, n = case.P, case.V, case.T, case.n
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            process.generate_data_from_dT(T/10)
            self.assertTrue(process.is_ideal_gas())
            self.assertTrue(process.is_first_law_satisfied())
            self.assertTrue(process.heat_absorbed < allowed_error)
            first_law_errors.append(process.first_law_consistency)
            ideal_gas_errors.append(np.max(process.ideal_gas_law_consistency))
            if show_test: print(f"Test {i} passed")
        print(standard_line("temperature",first_law_errors,ideal_gas_errors))

    def test_Carnot(self):
        print(header("Running Carnot cycle test"))
        first_law_errors = []; ideal_gas_errors = []; efficiency_errors = []
        for i in range(num_tests):
            state = Initial_state()
            P, V, T, n = state.P, state.V, state.T, state.n
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

            self.assertTrue(Carnot_cycle.cycle_is_ideal_gas, f"Cycle is not ideal gas: {Carnot_cycle.cycle_is_ideal_gas}")
            self.assertTrue(Carnot_cycle.cycle_is_first_law_satisfied, f"Cycle is not first law satisfied: {Carnot_cycle.cycle_is_first_law_satisfied}")

            self.assertTrue(Carnot_cycle.efficiency < 1, f"Efficiency is not less than 1: {Carnot_cycle.efficiency}")
            self.assertTrue(Carnot_cycle.efficiency > 0, f"Efficiency is not greater than 0: {Carnot_cycle.efficiency}")

            self.assertTrue(np.abs(Carnot_cycle.efficiency - Carnot_cycle.theoretical_efficiency) < allowed_error, f"Efficiency is not equal to theoretical efficiency: {Carnot_cycle.efficiency} vs {Carnot_cycle.theoretical_efficiency}")

            first_law_error = [process.first_law_consistency for process in Carnot_cycle.processes]
            ideal_gas_error = [np.max(process.ideal_gas_law_consistency) for process in Carnot_cycle.processes]

            first_law_errors.append(np.max(first_law_error))
            ideal_gas_errors.append(np.max(ideal_gas_error))

            efficiency_errors.append(np.abs(Carnot_cycle.efficiency - Carnot_cycle.theoretical_efficiency))

            if show_test: print(f"Test {i} passed")

        print("\nTesting data from Carnot Cycle: PASSED")
        print(f"\n\tFirst law inconsistency: {np.max(first_law_errors)} \n\tIdeal gas law inconsistency: {np.max(ideal_gas_errors)} \n\tEfficiency error: {np.max(efficiency_errors)}")

    

if __name__ == "__main__":
    # we want to run the tests in alhabetical order
    unittest.main()
