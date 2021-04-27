import numpy as np
from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, result, QuantumRegister
from qiskit.providers.aer import Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators.primitive_ops import CircuitOp
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal

def evaluate_circuit(num_qubits,initial_state_object,paulistring_object,sim,shots,whichrealcomputer=None,noisebackend=None):
    if sim == 'noiseless':
        backend = Aer.get_backend("qasm_simulator")
    initial_state_circuit = initial_state_object.get_qiskit_circuit()
    paulistring_circuit = paulistring_object.get_qiskit_circuit()
    qc = QuantumCircuit(num_qubits)
    qc.append(initial_state_circuit,range(num_qubits))
    qc.append(paulistring_circuit,range(num_qubits))
    qc.measure_all()
    if sim == 'noiseless':
        counts = execute(qc,backend=backend,shots=shots).result().get_counts()
    newcountdict = {}
    #Need to reverse the strings (some qiskit thing)
    for key in counts.keys():
        newkey = key[::-1]
        newcountdict[newkey] = counts[key]
    result = 0
    for key in newcountdict.keys():
        subresult = 1
        for i in range(paulistring_object.get_N()):
            if int(paulistring_object.return_string()[i]) == 0:
                subresult = subresult
            else:
                if key[i] == '0':
                    subresult = subresult
                elif key[i] == '1':
                    subresult = subresult*-1
        result = result + subresult*newcountdict[key]
    return result/sum(newcountdict.values())