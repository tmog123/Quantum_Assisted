import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import post_processing as pp
import plotting_package as plotp
import warnings
import json
import Qiskit_helperfunctions as qhf
from qiskit import QuantumCircuit, execute, result, QuantumRegister, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.tools.monitor import job_monitor
from copy import deepcopy

def basic_decomp_tfi(N,rotation,J=1,g=1):#Sum -JZ_iZ_i+1 + Sum gX_i
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.rx(rotation*g,i)
    for i in range(N-1):
        qc.rzz(-rotation*J,i,i+1)
    #for i in range(N):
    #    qc.rx(rotation*g,i)
    return qc.to_gate()



def do_trotter_decomposition_observable(initialstate,decomp_function,observable,simulator,quantum_computer_dict,timestep,numberofsteps,num_shots):
    rotation = timestep*2
    if simulator == "noisy_qasm":
        backend, coupling_map, noise_model = quantum_computer_dict[simulator]
    elif simulator == "real" or simulator == "noiseless_qasm":
        backend = quantum_computer_dict[simulator]
    N = initialstate.N
    qc = QuantumCircuit(N)
    qc.append(deepcopy(initialstate.get_qiskit_circuit()),range(N))
    for i in range(numberofsteps):
        qc.append(decomp_function(N,rotation,J=1,g=1),range(N))
    #print(qc)
    ps = observable.return_paulistrings()[0].return_string()
    #print(ps)
    for i in range(len(ps)):
        if ps[i]==1:
            qc.h(i)
        if ps[i]==2:
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()
    if simulator == "noisy_qasm":
        '''Changes Here'''
        sim_noise = AerSimulator(noise_model=noise_model)
        circ_noise = transpile(qc,sim_noise,coupling_map=coupling_map)
        results = sim_noise.run(circ_noise,shots=num_shots).result()
        counts = results.get_counts()
    elif simulator == "noiseless_qasm":
        counts = execute(qc, backend=backend, shots = num_shots).result().get_counts() 
    elif simulator == "real":
        job = execute(qc, backend = backend, shots = num_shots)
        job_monitor(job, interval = 2)
        results = job.result()
        counts = results.get_counts()
    newcountdict = {}
    for key in counts.keys():
        newkey = key[::-1]
        newcountdict[newkey] = counts[key]
    result = 0
    for key in newcountdict.keys():
        subresult = 1
        for i in range(len(ps)):
            if ps[i]==0:
                subresult = subresult
            else:
                if key[i] == '0':
                    subresult = subresult
                elif key[i] == '1':
                    subresult = subresult*(-1)
        result = result + subresult*newcountdict[key]
    return result/sum(newcountdict.values()) 