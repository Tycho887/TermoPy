import termoPy as tp
import unittest
import numpy as np

atm = 101325

allowed_error = 1e-4

def __test_ideal_gas(process,type):
    assert (np.mean(process.consistency)) < allowed_error, f"Process does not satisfy the ideal gas law, inconsistency: {np.mean(process.consistency)}"
    return np.mean(process.consistency)

def __test_first_law(process,type):
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

def __test_second_law(process,type):
    # we want to check that the values are in accordance with the second law of thermodynamics
    # d
    assert process.entropy[-1] >= process.entropy[0], f"The second law of thermodynamics is not satisfied for {process.title}, {type}"

def generate_random_state(range=5):
    # range defines the difference in order of magnitude between the minimum and maximum values
    P = np.random.uniform(tp.atm*np.exp(-range/2),tp.atm*np.exp(range/2))
    V = np.random.uniform(1e-3*np.exp(-range/2),1e-2*np.exp(range))
    T = np.random.uniform(273*np.exp(-range/2),273*np.exp(range/2))
    return P,V,T

def test_processes(P,V,T):
    processes = [tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True),
                 tp.Isochoric(P1=P,V1=V,T1=T,monatomic=True),
                 tp.Isobaric(P1=P,V1=V,T1=T,monatomic=True),
                 tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)]
    for process in processes:
        for type in ["+dV","-dV","+dP","-dP","+dT","-dT"]:
            method = f"generate_data_from_d{type[2]}(process.{type[2]}1*10)"
            try:
                eval(f"process.{method}")
                ideal_gas_error = __test_ideal_gas(process,type)
                first_law_error = __test_first_law(process,type)
            except AttributeError: print(f"\tThe {type} method does not exist")
    return True, ideal_gas_error, first_law_error

def run_process_tests(num_tests):
    ideal_gas_errors = []
    first_law_errors = []
    for i in range(num_tests):
        P,V,T = generate_random_state()
        passed, ideal_gas_error, first_law_error = test_processes(P,V,T)
        if passed:
            print(f"Test {i+1:<4} of {num_tests}. P = {P/atm:>9.3f} atm, V = {V:>9.3f} m^3, T = {T:>9.3f} K: PASSED")
            first_law_errors.append(first_law_error)
            ideal_gas_errors.append(ideal_gas_error)
    print(f"Average error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}")
    print(f"Average error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}")
               
def test_carnot_cycle():
    pass

class TestProcess:
    def __init__(self,num_tests):
        self.num_tests = num_tests

    def generate_random_state(self):
        # range defines the difference in order of magnitude between the minimum and maximum values
        P = np.random.uniform(0.01*tp.atm,100*tp.atm)
        V = np.random.uniform(1e-4,10)
        T = np.random.uniform(10,1000)
        return P,V,T

    def _test_methods(process):
        for type in ["+dV","-dV","+dP","-dP","+dT","-dT"]:
            op = "*"
            if "+" in type: op = "*"
            elif "-" in type: op = "/"
            method = f"generate_data_from_d{type[2]}(process.{type[2]}1{op}10)"
            try:
                eval(f"process.{method}")
                ideal_gas_error = __test_ideal_gas(process,type)
                first_law_error = __test_first_law(process,type)
            except AttributeError: print(f"\tThe {type} method does not exist")
        return True, ideal_gas_error, first_law_error

    def test_isothermal(self):
        ideal_gas_errors = []; first_law_errors = []
        for i in range(self.num_tests):
            P,V,T = self.generate_random_state()
            process = tp.Isothermal(P1=P,V1=V,T1=T,monatomic=True)
            passed, ideal_gas_error, first_law_error = self._test_methods(process)
            if passed:
                first_law_errors.append(first_law_error)
                ideal_gas_errors.append(ideal_gas_error)
        print(f"Isothermal PASSED: Average error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}, Average error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}")

    def test_isochoric(self):
        ideal_gas_errors = []; first_law_errors = []
        for i in range(self.num_tests):
            P,V,T = self.generate_random_state()
            process = tp.Isochoric(P1=P,V1=V,T1=T,monatomic=True)
            passed, ideal_gas_error, first_law_error = self._test_methods(process)
            if passed:
                first_law_errors.append(first_law_error)
                ideal_gas_errors.append(ideal_gas_error)
        print(f"Isochoric PASSED: Average error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}, Average error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}")

    def test_isobaric(self):
        ideal_gas_errors = []; first_law_errors = []
        for i in range(self.num_tests):
            P,V,T = self.generate_random_state()
            process = tp.Isobaric(P1=P,V1=V,T1=T,monatomic=True)
            passed, ideal_gas_error, first_law_error = self._test_methods(process)
            if passed:
                first_law_errors.append(first_law_error)
                ideal_gas_errors.append(ideal_gas_error)
        print(f"Isobaric PASSED: Average error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}, Average error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}")

    def test_adiabatic(self):
        ideal_gas_errors = []; first_law_errors = []
        for i in range(self.num_tests):
            P,V,T = self.generate_random_state()
            process = tp.Adiabatic(P1=P,V1=V,T1=T,monatomic=True)
            passed, ideal_gas_error, first_law_error = self._test_methods(process)
            if passed:
                first_law_errors.append(first_law_error)
                ideal_gas_errors.append(ideal_gas_error)
        print(f"Adiabatic PASSED: Average error in ideal gas law: {np.mean(ideal_gas_errors):<4.3e}, Average error in first law of thermodynamics: {np.mean(first_law_errors):<4.3e}")




if __name__ == "__main__":
    #run_process_tests(100)
    test = TestProcess(100)
    test.test_isothermal()
    test.test_isochoric()
    test.test_isobaric()
    test.test_adiabatic()

