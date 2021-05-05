from qiskit.circuit.library import EfficientSU2
import qiskit.tools.jupyter # import handy jupyter tools for viewing backend details and monitoring job status
from qiskit import ClassicalRegister

#Same as jon
from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, result, QuantumRegister, transpile
from qiskit.providers.aer import Aer
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators.primitive_ops import CircuitOp
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.aqua.operators.state_fns import CircuitStateFn
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn
from qiskit.transpiler import CouplingMap

import ansatz_class_package as acp
import pauli_class_package as pcp
import numpy as np
from copy import deepcopy

IBMQ.load_account() 

def create_quantum_computer_simulation(couplingmap,depolarizingnoise=False,depolarizingnoiseparameter=0,bitfliperror=False,bitfliperrorparameter=0,measerror=False,measerrorparameter=0):
    """
    Returns a dictionary, where the key is what type of simulation ("noisy_qasm", "noiseless_qasm", "real"), and the value are the objects required for that particular simulation
    """
    sims = ["noisy_qasm", "noiseless_qasm"]
    dicto = dict()
    for sim in sims:
        if sim == "noiseless_qasm":
            backend = Aer.get_backend('qasm_simulator')
            dicto[sim] = backend
        elif sim == "noisy_qasm":
            backend = Aer.get_backend('qasm_simulator')
            coupling_map = CouplingMap(couplingmap)
            noise_model = NoiseModel()
            if depolarizingnoise == True:
                depolarizingerror = depolarizing_error(depolarizingnoiseparameter, 1)
                noise_model.add_all_qubit_quantum_error(depolarizingerror, ['u1', 'u2', 'u3'])
            if bitfliperror == True:
                error_gate1 = pauli_error([('X',bitfliperrorparameter), ('I', 1 - bitfliperrorparameter)])
                noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
                error_gate2 = error_gate1.tensor(error_gate1)
                noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])
            if measerror == True:
                error_meas = pauli_error([('X',measerrorparameter), ('I', 1 - measerrorparameter)])
                noise_model.add_all_qubit_quantum_error(error_meas,"measure")
            dicto[sim] = (backend, coupling_map, noise_model)
    return dicto



def choose_quantum_computer(hub, group, project, quantum_com):
    """
    Returns a dictionary, where the key is what type of simulation ("noisy_qasm", "noiseless_qasm", "real"), and the value are the objects required for that particular simulation
    """
    sims = ["noisy_qasm", "real", "noiseless_qasm"]
    dicto = dict()
    for sim in sims:
        if sim == "noisy_qasm":
            backend = Aer.get_backend('qasm_simulator')
            provider = IBMQ.get_provider(hub = hub, group = group, project = project)
            noisebackend = provider.get_backend(quantum_com)
            noisebackend = provider.get_backend(quantum_com)
            coupling_map = noisebackend.configuration().coupling_map 
            # noise_model = NoiseModel.from_backend(noisebackend,gate_error=False,readout_error=False,thermal_relaxation=True)
            noise_model = NoiseModel.from_backend(noisebackend)
            dicto[sim] = (backend, coupling_map, noise_model)
        elif sim == "real":
            provider = IBMQ.get_provider(hub = hub, group = group, project = project)
            backend = provider.get_backend(quantum_com)
            dicto[sim] = backend
        elif sim == "noiseless_qasm":
            backend = Aer.get_backend('qasm_simulator')
            dicto[sim] = backend
    return dicto

def measurement_error_mitigator(systemsize, sim, quantum_com_choice_results,
    shots = 8192):
    if sim == "noisy_qasm":
        backend, coupling_map, noise_model = quantum_com_choice_results[sim]

        qr = QuantumRegister(systemsize)
        meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
        print('Calibrating POVM Matrix')
        job = execute(meas_calibs,backend=backend,shots=shots,noise_model=noise_model,coupling_map=coupling_map)
        cal_results = job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        meas_filter = meas_fitter.filter
        print("Provider backend: ", backend)
        return meas_filter
    elif sim == "real":
        backend = quantum_com_choice_results[sim]

        qr = QuantumRegister(systemsize)
        meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
        print('Calibrating POVM Matrix')
        job = execute(meas_calibs,backend=backend,shots=shots)
        job_monitor(job, interval = 2)
        cal_results = job.result()
        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
        meas_filter = meas_fitter.filter
        print("Provider backend: ", backend)
        return meas_filter


class expectation_calculator(object):

    @staticmethod
    def make_qc_to_measure_pstring(initial_state_object, pauli_string_strform):
        """
        This is a helper function, won't be called directly
        """
        qc = deepcopy(initial_state_object.get_qiskit_circuit()) #we need to return a copy of the circuit, cause .get_qiskit_circuit() returns something that is mutable.
        pauli_string_strform = pauli_string_strform[::-1] #cause qiskit reads stuff in reverse 
        pauli_string = pauli_string_strform
        for qubit_index in range(len(pauli_string)):
            qubit = pauli_string[qubit_index]
            if qubit == "0":
                qc.id(qubit_index)
            elif qubit == "1":
                qc.h(qubit_index)
            elif qubit == "2":
                qc.sdg(qubit_index)
                qc.h(qubit_index)
            elif qubit == "3":
                qc.id(qubit_index)
        qc.measure_all()
        return qc

    def __init__(self, initial_state_object, sim, quantum_com_choice_results, num_shots = 8192, meas_error_mitigate = False, meas_filter = None):
        if sim == "noisy_qasm" or sim == "real":
            if meas_error_mitigate == True and meas_filter == None:
                raise(RuntimeError("no meas_filter specified, so no measurement error mitigation can be done!"))

        if sim == "noisy_qasm":
            backend, coupling_map, noise_model = quantum_com_choice_results[sim]
            self.backend = backend
            self.coupling_map = coupling_map
            self.noise_model = noise_model

        elif sim == "real" or sim == "noiseless_qasm":
            backend = quantum_com_choice_results[sim]
            self.backend = backend

        self.initial_state_object = initial_state_object
        self.sim = sim 
        self.quantum_com_choice_results = quantum_com_choice_results 
        self.num_shots = num_shots 
        self.meas_error_mitigate = meas_error_mitigate 
        self.meas_filter = meas_filter 
        self.previous_expectation_vals = dict() 
        # make_qc_to_measure_pstring = self.make_qc_to_measure_pstring

    def load_previously_calculated_expectation_vals_and_initial_state_obj(self, previous_expectation_vals_to_load, initial_state_obj):
        self.previous_expectation_vals = previous_expectation_vals_to_load
        print("Previous expectation values loaded!")
        self.initial_state_object = initial_state_obj
        print("Initial state object loaded")

    def get_calculated_expectation_vals_dict_and_initial_state_obj(self):
        """
        self.previous_expectation_vals is a dictionary, where the key is the str_form of the pauli_string, and the value is the expectation.

        E.g, the key can be "123", and the value would be <psi|X1 otimes Y2 otimes Z3| psi>.

        Also returns the initial state on which the expectation values were calculated from
        """
        return (self.previous_expectation_vals, self.initial_state_object)

    def calculate_expectation(self, pauli_string_object):
        # previous_expectation_vals = self.previous_expectation_vals 
        # initial_state_object = self.initial_state_object 
        # sim = self.sim 
        # backend = self.backend 
        # coupling_map = self.coupling_map 
        # meas_error_mitigate = self.meas_error_mitigate 
        # meas_filter = self.meas_filter        
        # noise_model = self.noise_model
        # num_shots = self.num_shots 

        pauli_string_strform = pauli_string_object.get_string_for_hash()
        pauli_string_coeff = pauli_string_object.return_coefficient()
        # print(pauli_string_coeff)
        pauli_string = pauli_string_strform
        if pauli_string in self.previous_expectation_vals.keys():
            return pauli_string_coeff * self.previous_expectation_vals[pauli_string]
        qc = self.make_qc_to_measure_pstring(self.initial_state_object, pauli_string)

        if self.sim == "noisy_qasm":
            '''NEED TO MAKE SOME CHANGES HERE FOR ARTIFICIAL NOISE MODEL: JON'''
            #results = execute(qc, backend=self.backend, shots = self.num_shots, coupling_map = self.coupling_map, noise_model = self.noise_model).result()
            
            '''Changes Here'''
            sim_noise = AerSimulator(noise_model=self.noise_model)
            circ_noise = transpile(qc,sim_noise,coupling_map=self.coupling_map)
            results = sim_noise.run(circ_noise).result()
            
            if self.meas_error_mitigate == True:
                results = self.meas_filter.apply(results)
            counts = results.get_counts()
        elif self.sim == "noiseless_qasm":
            counts = execute(qc, backend=self.backend, shots = self.num_shots).result().get_counts() 
        elif self.sim == "real":
            job = execute(qc, backend = self.backend, shots = self.num_shots)
            job_monitor(job, interval = 2)
            results = job.result()
            if self.meas_error_mitigate == True:
                results = self.meas_filter.apply(results)
            counts = results.get_counts() 
        #print("Finished shots")

        frequency_dict = dict()
        total_num_of_counts = sum(counts.values())
        for key,value in counts.items():
            frequency_dict[key] = value/total_num_of_counts
        ans = 0 + 0j
        #since we did measurement in Z basis, we must change our pauli_string.
        #Note that when we did "make qc to measure p_string", we have already
        #reversed the p_string there.  for the "counts" object, note that the
        #bitstrings are in qiskit order, i.e the rightmost bit is the 1st
        #qubit.
        new_pauli_string = []
        for char in pauli_string:
            new_pauli_string.append("1") if char != "0" else new_pauli_string.append(char)
        new_pauli_string = "".join(new_pauli_string)
        for key, value in frequency_dict.items():
            # print(key)
            coeff = np.base_repr(int(key,2) & int(new_pauli_string, 2), base = 2).count("1") #bitwise and
            ans += (-1)**coeff * value
        # print(pauli_string, ans)
        self.previous_expectation_vals[pauli_string] = ans
        return ans * pauli_string_coeff

# def make_expectation_calculator(initial_state_object, sim, quantum_com_choice_results, num_shots = 8192, meas_error_mitigate = False, meas_filter = None):
#     """
#     This upper function has a dictionary that stores the previously calculated expectation values, so we don't do any re-calculation.

#     sim can be either noisy_qasm, noiseless_qasm, or real.
#     """
#     if sim == "noisy_qasm" or sim == "real":
#         if meas_error_mitigate == True and meas_filter == None:
#             raise(RuntimeError("no meas_filter specified, so no measurement error mitigation can be done!"))
#     if sim == "noisy_qasm":
#         backend, coupling_map, noise_model = quantum_com_choice_results[sim]
#     elif sim == "real" or sim == "noiseless_qasm":
#         backend = quantum_com_choice_results[sim]
#     previous_expectation_vals = dict() 

#     def expectation_calculator(pauli_string_object):
#         pauli_string_strform = pauli_string_object.get_string_for_hash()
#         pauli_string_coeff = pauli_string_object.return_coefficient()
#         # print(pauli_string_coeff)
#         pauli_string = pauli_string_strform
#         if pauli_string in previous_expectation_vals.keys():
#             return pauli_string_coeff * previous_expectation_vals[pauli_string]
#         qc = make_qc_to_measure_pstring(initial_state_object, pauli_string)

#         if sim == "noisy_qasm":
#             results = execute(qc, backend=backend, shots = num_shots, coupling_map = coupling_map, noise_model = noise_model).result()
#             if meas_error_mitigate == True:
#                 results = meas_filter.apply(results)
#             counts = results.get_counts()
#         elif sim == "noiseless_qasm":
#             counts = execute(qc, backend=backend, shots = num_shots).result().get_counts() 
#         elif sim == "real":
#             job = execute(qc, backend = backend, shots = num_shots)
#             job_monitor(job, interval = 2)
#             results = job.result()
#             if meas_error_mitigate == True:
#                 results = meas_filter.apply(results)
#             counts = results.get_counts() 
#         #print("Finished shots")

#         frequency_dict = dict()
#         total_num_of_counts = sum(counts.values())
#         for key,value in counts.items():
#             frequency_dict[key] = value/total_num_of_counts
#         ans = 0 + 0j
#         #since we did measurement in Z basis, we must change our pauli_string.
#         #Note that when we did "make qc to measure p_string", we have already
#         #reversed the p_string there.  for the "counts" object, note that the
#         #bitstrings are in qiskit order, i.e the rightmost bit is the 1st
#         #qubit.
#         new_pauli_string = []
#         for char in pauli_string:
#             new_pauli_string.append("1") if char != "0" else new_pauli_string.append(char)
#         new_pauli_string = "".join(new_pauli_string)
#         for key, value in frequency_dict.items():
#             # print(key)
#             coeff = np.base_repr(int(key,2) & int(new_pauli_string, 2), base = 2).count("1") #bitwise and
#             ans += (-1)**coeff * value
#         # print(pauli_string, ans)
#         previous_expectation_vals[pauli_string] = ans
#         return ans * pauli_string_coeff
#     return expectation_calculator


#%% Testing
# if __name__ == "__main__":
#     num_qubits = 2
#     quantum_computer = "ibmq_rome"

#     sim = "noisy_qasm"
#     num_shots = 8192

#     # sim = "noiseless_qasm"
#     # num_shots = 100000

#     test_pstring = "23"
#     pauli_string_object = pcp.paulistring(num_qubits, test_pstring, -1j)
#     initial_state_object = acp.Initialstate(num_qubits, "efficient_SU2", 123, 3)

#     initial_statevector = initial_state_object.get_statevector()
#     print("the matrix multiplication result is",initial_statevector.conj().T @ pauli_string_object.get_matrixform() @ initial_statevector)


#     quantum_computer_choice_results = choose_quantum_computer("ibm-q-nus", group = "default", project = "reservations", quantum_com = quantum_computer)

#     #Noisy QASM
#     meas_filter = measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)
#     expectation_calculator = make_expectation_calculator(initial_state_object, sim, quantum_computer_choice_results, meas_error_mitigate = True, meas_filter = meas_filter)

#     #Noiseless QASM
#     # expectation_calculator = make_expectation_calculator(initial_state_object, sim, quantum_computer_choice_results, meas_error_mitigate = False)


#     print("the quantum result is",expectation_calculator(pauli_string_object))


# import Qiskit_helperfunctions_kh as qhf #IBMQ account is loaded here in this import
# hub, group, project = "ibm-q-nus", "default", "reservations"
# quantum_com = "ibmq_rome" 

# #Other parameters for running on the quantum computer
# sim = "noisy_qasm"
# num_shots = 8192

# quantum_computer_choice_results = qhf.choose_quantum_computer(hub, group, project, quantum_com)
# mitigate_meas_error = True 
# meas_filter = qhf.measurement_error_mitigator(num_qubits, sim, quantum_computer_choice_results, shots = num_shots)

# expectation_calculator = qhf.make_expectation_calculator(initial_state, sim, quantum_computer_choice_results, meas_error_mitigate = mitigate_meas_error, meas_filter = meas_filter)