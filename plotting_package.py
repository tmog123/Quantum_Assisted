import matplotlib.pyplot as plt
import numpy as np
import matrix_class_package as mcp
import pauli_class_package as pcp
from scipy.linalg import expm 

def set_axis_labels(x_label,y_label,fontsize):
    plt.xlabel(x_label,fontsize=fontsize)
    plt.ylabel(y_label,fontsize=fontsize)

def QS_plotter_foralpha(ansatz,times):
    for mom in ansatz.get_moments():
        plt.plot(times, np.abs(mom.alphas[:-1]),label=str(mom.paulistring.string))

def plotter_fortrotter(trotvalues,times):
    plt.plot(times,trotvalues,label='Trotter')

def getdata_forbetamatrix_observable(num_qubits,ansatzlist,whatKs,observable,initial_state,betamatrixlist,evalmethod='matrix_multiplication',expectation_calculator=None):
    if evalmethod == "qiskit_circuits" and expectation_calculator == None:
        raise(RuntimeError("You need to pass in the expectation_calculator as an argument (see the Qiskit_helperfunctions package)."))
    allresultlist = []
    #betacounter = 0
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Preparing observable for plotting for K = ' + str(i))
            ansatz = ansatzlist[i]
            O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
            if evalmethod == 'matrix_multiplication':
                print('Evaluating Observable Matrix classically')
                Omat = O_matrix_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            elif evalmethod == "qiskit_circuits":
                print("Evaluating Observable Matrix with Qiskit circuits")
                Omat = O_matrix_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
            result = 0
            #for a in range(len(betamatrixlist[i-1])):
            #    for b in range(len(betamatrixlist[i-1])):
            #        result = result + betamatrixlist[i-1][a][b]*Omat[b][a]
            result = result + np.trace(betamatrixlist[i-1]@Omat)
            allresultlist.append(result)
            #betacounter = betacounter+1
    return allresultlist
    

def QS_plotter_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state, evalmethod='matrix_multiplication', expectation_calculator = None):
    """
    evalmethod can either be "qiskit_circuits" or "matrix_multiplication"
    """
    if evalmethod == "qiskit_circuits" and expectation_calculator == None:
        raise(RuntimeError("You need to pass in the expectation_calculator as an argument (see the Qiskit_helperfunctions package)."))
    if qstype == 'TQS':
        name = 'TQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Preparing observable for plotting for K = ' + str(i))
            ansatz = ansatzlist[i]
            O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
            if evalmethod == 'matrix_multiplication':
                print('Evaluating Observable Matrix classically')
                Omat = O_matrix_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            elif evalmethod == "qiskit_circuits":
                print("Evaluating Observable Matrix with Qiskit circuits")
                Omat = O_matrix_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
            result_pauli_string, result_alphas = ansatz.get_alphas() 
            result_alphas = list(zip(*result_alphas))
            observable_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx] 
                alpha = np.array(alpha)
                observable_value = alpha.conj().T @ Omat @ alpha
                observable_value = observable_value.real 
                observable_vals.append(observable_value)
            lab = name + ' K=' + str(i)
            plt.plot(times, observable_vals,label=lab)
def get_data_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state, evalmethod='matrix_multiplication', expectation_calculator = None):
    """
    evalmethod can either be "qiskit_circuits" or "matrix_multiplication"
    """
    if evalmethod == "qiskit_circuits" and expectation_calculator == None:
        raise(RuntimeError("You need to pass in the expectation_calculator as an argument (see the Qiskit_helperfunctions package)."))
    if qstype == 'TQS':
        name = 'TQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    finaldict = {}
    finaldict['times'] = list(times.real)
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Preparing observable for plotting for K = ' + str(i))
            ansatz = ansatzlist[i]
            O_matrix_uneval = mcp.unevaluatedmatrix(num_qubits, ansatz, observable, "O")
            if evalmethod == 'matrix_multiplication':
                print('Evaluating Observable Matrix classically')
                Omat = O_matrix_uneval.evaluate_matrix_by_matrix_multiplicaton(initial_state)
            elif evalmethod == "qiskit_circuits":
                print("Evaluating Observable Matrix with Qiskit circuits")
                Omat = O_matrix_uneval.evaluate_matrix_with_qiskit_circuits(expectation_calculator)
            result_pauli_string, result_alphas = ansatz.get_alphas() 
            result_alphas = list(zip(*result_alphas))
            observable_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx] 
                alpha = np.array(alpha)
                observable_value = alpha.conj().T @ Omat @ alpha
                observable_value = observable_value.real 
                observable_vals.append(observable_value)
            lab = name + ' K=' + str(i)
            finaldict[str(i)] = list(observable_vals)
            #plt.plot(times, observable_vals,label=lab)
    return finaldict

def CS_plotter_forobservable(times,classicalresult):
    plt.plot(times,classicalresult,label='Classical')

def show_plot():
    plt.legend()
    plt.show()

def print_plot(location,bboxtight=False):
    plt.legend()
    if bboxtight == "tight":
        plt.savefig(location,bbox_inches="tight")
    else:
        plt.savefig(location)
    plt.close()

def QS_plotter_for_fidelity(num_qubits, ansatzlist, times,
     whatKs, qstype, hamiltonian, initial_state):
    initial_statevector = initial_state.get_statevector()
    if qstype == 'TQS':
        name = 'TQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Plotting fidelity for K = ' + str(i))
            ansatz = ansatzlist[i]
            result_pauli_string, result_alphas = ansatz.get_alphas()
            result_alphas = list(zip(*result_alphas)) #transpose it so that each entry is a time value 
            fidelity_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx]
                alpha = np.array(alpha)
                state = np.zeros(2**num_qubits) + 1j*np.zeros(2**num_qubits)
                for j in range(len(alpha)):
                    # print("I'm here", result_pauli_string[j])
                    result_pauli_string_obj = pcp.paulistring(num_qubits, result_pauli_string[j], 1)
                    # print(result_pauli_string_obj)
                    result_pauli_string_matrix = result_pauli_string_obj.get_matrixform()
                    # print("i am here", len(initial_statevector))
                    # print("i am here two", len(result_pauli_string_matrix))
                    # print("i am here three", alpha[j])
                    state += alpha[j] * result_pauli_string_matrix @initial_statevector
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                fidelity_vals.append(fidelity)
            lab = name + "Fidelity K=" + str(i)
            plt.plot(times, fidelity_vals, label = lab)

def get_data_for_fidelity(num_qubits, ansatzlist, times,whatKs, qstype, hamiltonian, initial_state):
    finaldict = {}
    finaldict['times'] = list(times.real)
    initial_statevector = initial_state.get_statevector()
    if qstype == 'TQS':
        name = 'TQS'
    if qstype == 'QAS':
        name = 'QAS'
    if qstype == 'CQFF':
        name = 'CQFF'
    for i in range(len(ansatzlist)):
        if i in whatKs:
            #print('Plotting fidelity for K = ' + str(i))
            ansatz = ansatzlist[i]
            result_pauli_string, result_alphas = ansatz.get_alphas()
            result_alphas = list(zip(*result_alphas)) #transpose it so that each entry is a time value 
            fidelity_vals = []
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx]
                alpha = np.array(alpha)
                state = np.zeros(2**num_qubits) + 1j*np.zeros(2**num_qubits)
                for j in range(len(alpha)):
                    # print("I'm here", result_pauli_string[j])
                    result_pauli_string_obj = pcp.paulistring(num_qubits, result_pauli_string[j], 1)
                    # print(result_pauli_string_obj)
                    result_pauli_string_matrix = result_pauli_string_obj.get_matrixform()
                    # print("i am here", len(initial_statevector))
                    # print("i am here two", len(result_pauli_string_matrix))
                    # print("i am here three", alpha[j])
                    state += alpha[j] * result_pauli_string_matrix @initial_statevector
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                fidelity_vals.append(fidelity)
            lab = name + " K=" + str(i)
            #plt.plot(times, fidelity_vals, label = lab)
            finaldict[str(i)] = list(fidelity_vals)
    return finaldict


def get_fidelity_results(num_qubits, ansatzlist, times,
    whatKs, hamiltonian, initial_state):
    """
    Returns a dictionary of dictionaries. The keys are the values of K that we
    are considering, the values are the dictionaries for that K value. For a
    particular K value, the corresponding dictionary is such that the keys are
    the time values, and the values are the fidelities at that particular
    time
    """
    initial_statevector = initial_state.get_statevector()
    # if qstype == 'TTQS':
    #     name = 'TTQS'
    # if qstype == 'QAS':
    #     name = 'QAS'
    # if qstype == 'CQFF':
    #     name = 'CQFF'
    collated_fidelity_vals = dict()
    for i in range(len(ansatzlist)):
        if i in whatKs:
            print('Calculating and saving fidelity for K = ' + str(i))
            ansatz = ansatzlist[i]
            result_pauli_string, result_alphas = ansatz.get_alphas()
            result_alphas = list(zip(*result_alphas)) #transpose it so that each entry is a time value 
            fidelity_vals = dict()
            for time_idx in range(len(times)):
                time = times[time_idx]
                alpha = result_alphas[time_idx]
                alpha = np.array(alpha)
                state = np.zeros(2**num_qubits) + 1j*np.zeros(2**num_qubits)
                for j in range(len(alpha)):
                    # print("I'm here", result_pauli_string[j])
                    result_pauli_string_obj = pcp.paulistring(num_qubits, result_pauli_string[j], 1)
                    # print(result_pauli_string_obj)
                    result_pauli_string_matrix = result_pauli_string_obj.get_matrixform()
                    # print("i am here", len(initial_statevector))
                    # print("i am here two", len(result_pauli_string_matrix))
                    # print("i am here three", alpha[j])
                    state += alpha[j] * result_pauli_string_matrix @initial_statevector
                
                hamiltonian_matrix = hamiltonian.to_matrixform()
                theoretical_state = expm(-1j * hamiltonian_matrix * time) @ initial_statevector
                # theoretical_state = final_results_from_classical_simulator[time_idx]
                fidelity = np.abs(np.vdot(theoretical_state, state))
                # if time == 60:
                #     print("actual_state_is", state)
                #     print("theoretical_state_is", theoretical_state)
                # fidelity_vals.append(fidelity)
                fidelity_vals[time] = fidelity
            collated_fidelity_vals[i] = fidelity_vals
    return collated_fidelity_vals

def qutip_comparison_with_k_plot_expectation_values(num_qubits,results, theoretical_curves, which_ks,random_selection_new,num_of_csk_states,specify_names=False,observable_names=None):
    x_vals = list(results.keys())
    observable_expectation_results = [list(i[0].items()) for i in list(results.values())]
    observable_expectation_results_transposed = list(zip(*observable_expectation_results))
    # don't need to know the details of what the heck this chunk does, but this
    # chunk is such that the key is the k value, and for each k, we have
    # [(observable one results against g), (observable 2 results against g),
    # (observable 3 results against g)]
    observable_expectation_results_transposed_dict = {k+1:list(zip(*[j[1] for j in observable_expectation_results_transposed[k]])) for k in range(len(observable_expectation_results_transposed))}

    if specify_names==False:
    # which_observables = [0,1,2]
        for k,observable_results in observable_expectation_results_transposed_dict.items():
            if k not in which_ks:
                continue
            for index in range(len(observable_results)):
                # if index not in which_observables:
                #     continue
                observable_result = observable_results[index]
                if random_selection_new:
                    plt.plot(x_vals, observable_result, "o", label = str(num_of_csk_states(k)) + " csk states" + " observable" + str(index + 1))
                else:
                    plt.plot(x_vals, observable_result, "o", label = "k=" + str(k) + " observable" + str(index + 1))
        plt.plot(theoretical_curves[0], theoretical_curves[1][0], label = "theoretical_observable1")
        plt.plot(theoretical_curves[0], theoretical_curves[1][1],
        label = "theoretical_observable2")
        plt.plot(theoretical_curves[0], theoretical_curves[1][2],
        label = "theoretical_observable3")

        plt.xlabel("delta")
        plt.ylabel("expectation_vals")
        plt.title(str(num_qubits) + " qubits expectation values")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    elif specify_names==True:
    # which_observables = [0,1,2]
        for k,observable_results in observable_expectation_results_transposed_dict.items():
            if k not in which_ks:
                continue
            for index in range(len(observable_results)):
                # if index not in which_observables:
                #     continue
                observable_result = observable_results[index]
                if random_selection_new:
                    plt.plot(x_vals, observable_result, "o", label = str(num_of_csk_states(k)) + " csk states" + observable_names[index])
                else:
                    plt.plot(x_vals, observable_result, "o", label = "k=" + str(k) + observable_names[index])
        plt.plot(theoretical_curves[0], theoretical_curves[1][0], label = observable_names[0])
        plt.plot(theoretical_curves[0], theoretical_curves[1][1],
        label = observable_names[1])
        plt.plot(theoretical_curves[0], theoretical_curves[1][2],
        label = observable_names[2])

        plt.xlabel("delta")
        plt.ylabel("expectation_vals")
        plt.title(str(num_qubits) + " qubits expectation values")
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()        
def plot_fidelities(num_qubits,results,random_selection_new,num_of_csk_states):
    x_vals = list(results.keys())
    y_vals_all_k = [list(i[2].values()) for i in list(results.values())]
    y_vals_all_k_transposed = list(zip(*y_vals_all_k))
    y_vals_all_k_transposed_dict = {k+1:y_vals_all_k_transposed[k] for k in range(len(y_vals_all_k_transposed))}

    for k,fidelities in y_vals_all_k_transposed_dict.items():
        print(k)
        if random_selection_new:
            plt.plot(x_vals, fidelities, label=str(num_of_csk_states(k)) + " csk states")
        else:
            plt.plot(x_vals, fidelities, label="k=" + str(k))

    plt.xlabel("delta")
    plt.ylabel("fidelity")
    plt.title(str(num_qubits) + " qubits fidelity graph")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.savefig(savefile,bbox_inches="tight")
    #plt.close()
    #plt.show()
