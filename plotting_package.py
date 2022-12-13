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

def plotter_fortrotter(trotvalues,times,linewidths=2):
    plt.plot(times,trotvalues,label='Trotter',linewidth=linewidths)

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
    

def QS_plotter_forobservable(num_qubits,ansatzlist,times,whatKs,qstype,observable,initial_state, evalmethod='matrix_multiplication', expectation_calculator = None,line_styles=[],linewidths=2):
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
    lscount=0
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
            if len(line_styles)!=0:
                plt.plot(times, observable_vals,label=lab,linestyle=line_styles[lscount],linewidth=linewidths)
                lscount=lscount+1
            else:
                plt.plot(times, observable_vals,label=lab,linewidth=linewidths)
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

def CS_plotter_forobservable(times,classicalresult,line_style=None,linewidths=2):
    if line_style==None:
        plt.plot(times,classicalresult,label='Classical',linewidth=linewidths)
    else:
        plt.plot(times,classicalresult,label='Classical',linestyle=line_style,linewidth=linewidths)

def show_plot():
    plt.legend()
    plt.show()

def print_plot(location,bboxtight=False,legendsize=10):
    plt.legend(prop={'size': legendsize})
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

def qutip_comparison_with_k_plot_expectation_values(num_qubits,results, theoretical_curves, which_ks,random_selection_new,num_of_csk_states,specify_names=False,observable_names=None,plot_title=None,x_axis=None,y_axis=None, location=None, bboxtight=None,k_dot_styles=['o'],line_styles=['-','--','-.']):
    x_vals = list(results.keys())
    observable_expectation_results = [list(i[0].items()) for i in list(results.values())]
    observable_expectation_results_transposed = list(zip(*observable_expectation_results))
    # don't need to know the details of what the heck this chunk does, but this
    # chunk is such that the key is the k value, and for each k, we have
    # [(observable one results against g), (observable 2 results against g),
    # (observable 3 results against g)]
    observable_expectation_results_transposed_diqutip_comparison_with_k_plot_expectation_valuesct = {k+1:list(zip(*[j[1] for j in observable_expectation_results_transposed[k]])) for k in range(len(observable_expectation_results_transposed))}
    observable_expectation_results_transposed_dict = observable_expectation_results_transposed_diqutip_comparison_with_k_plot_expectation_valuesct
    k_dot_style_counter=0
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
                    plt.plot(x_vals, observable_result, k_dot_styles[k_dot_style_counter], label = str(num_of_csk_states(k))+ " "+ r'$\mathbb{CS}_{K}$'+" states" + " observable" + str(index + 1))
                else:
                    plt.plot(x_vals, observable_result, k_dot_styles[k_dot_style_counter], label = "k=" + str(k) + " observable" + str(index + 1))
                if k_dot_style_counter+1<len(k_dot_styles):
                    k_dot_style_counter = k_dot_style_counter+1
        plt.plot(theoretical_curves[0], theoretical_curves[1][0],linestyle=line_styles[0], label = "theoretical_observable1")
        plt.plot(theoretical_curves[0], theoretical_curves[1][1],linestyle=line_styles[1],label = "theoretical_observable2")
        plt.plot(theoretical_curves[0], theoretical_curves[1][2],linestyle=line_styles[2],label = "theoretical_observable3")
        
        if x_axis!=None:
            plt.xlabel(x_axis)
        if y_axis!=None:
            plt.ylabel(y_axis)
        plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
        if plot_title!=None:
            plt.title(plot_title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    elif specify_names==True:
    # which_observables = [0,1,2]
        plt.plot(theoretical_curves[0], theoretical_curves[1][0],linestyle=line_styles[0], label = 'Theoretical'+ observable_names[0])
        plt.plot(theoretical_curves[0], theoretical_curves[1][1],linestyle=line_styles[1],
        label ='Theoretical'+ observable_names[1])
        plt.plot(theoretical_curves[0], theoretical_curves[1][2],linestyle=line_styles[2],
        label ='Theoretical'+ observable_names[2])
        for k,observable_results in observable_expectation_results_transposed_dict.items():
            if k not in which_ks:
                continue
            for index in range(len(observable_results)):
                # if index not in which_observables:
                #     continue
                observable_result = observable_results[index]
                if random_selection_new:
                    plt.plot(x_vals, observable_result, k_dot_styles[k_dot_style_counter], label = str(num_of_csk_states[k]) + " "+ r'$\mathbb{CS}_{K}$'+" states" + observable_names[index])
                else:
                    plt.plot(x_vals, observable_result, k_dot_styles[k_dot_style_counter], label = r'$\mathbb{CS}_{K}=$' + str(k)+',' + observable_names[index])
                if k_dot_style_counter+1<len(k_dot_styles):
                    k_dot_style_counter = k_dot_style_counter+1
        # plt.plot(theoretical_curves[0], theoretical_curves[1][0], label = 'Theoretical'+ observable_names[0])
        # plt.plot(theoretical_curves[0], theoretical_curves[1][1],
        # label ='Theoretical'+ observable_names[1])
        # plt.plot(theoretical_curves[0], theoretical_curves[1][2],
        # label ='Theoretical'+ observable_names[2])

        if x_axis!=None:
            plt.xlabel(x_axis, fontsize=16)
        if y_axis!=None:
            plt.ylabel(y_axis, fontsize = 16)
        plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
        if plot_title!=None:
            plt.title(plot_title)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()        
        if location:
            if bboxtight == "tight":
                plt.savefig(location,bbox_inches="tight")
            else:
                plt.savefig(location)
        else:
            plt.show()
        plt.close()

def plot_fidelities(num_qubits,results,random_selection_new,num_of_csk_states=None,plot_title=None,x_axis=None,y_axis=None, location=None, bboxtight=None,plotlog=False,k_dot_styles=None):
    x_vals = list(results.keys())
    y_vals_all_k = [list(i[2].values()) for i in list(results.values())]
    y_vals_all_k_transposed = list(zip(*y_vals_all_k))
    y_vals_all_k_transposed_dict = {k+1:y_vals_all_k_transposed[k] for k in range(len(y_vals_all_k_transposed))}

    k_dot_style_counter=0
    for k,fidelities in y_vals_all_k_transposed_dict.items():
        print(k)
        if plotlog==True:
            fidelities = np.log(np.array(fidelities))
        if random_selection_new and num_of_csk_states!=None:
            plt.plot(x_vals, fidelities, k_dot_styles[k_dot_style_counter], label=str(num_of_csk_states[k]) +" "+r'$\mathbb{CS}_{K}$'+" states")
        else:
            plt.plot(x_vals, fidelities, k_dot_styles[k_dot_style_counter], label="k=" + str(k))
        if k_dot_styles!=None:
            if k_dot_style_counter+1<len(k_dot_styles):
                k_dot_style_counter = k_dot_style_counter+1
    if x_axis!=None:
        plt.xlabel(x_axis, fontsize=16)
    if y_axis!=None:
        plt.ylabel(y_axis, fontsize=16)
    plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
    if plot_title!=None:
        plt.title(plot_title)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    #plt.savefig(savefile,bbox_inches="tight")
    #plt.close()
    #plt.show()
    if location:
        if bboxtight == "tight":
            plt.savefig(location,bbox_inches="tight")
        else:
            plt.savefig(location)
    else:
        plt.show()
    plt.close()



# def plotter_for_paper():#For reference, not in use
#     fig = plt.figure(figsize=(14,8), constrained_layout=False)
#     gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 2], hspace=0.25, wspace=0.25)
#     ax1 = fig.add_subplot(gs[0,0])
#     ax2 = fig.add_subplot(gs[0,1])
#     ax3 = fig.add_subplot(gs[1,0])
#     ax4 = fig.add_subplot(gs[1,1])
#     # --------------
#     # SUBLOT c)
#     # --------------
#     ax1.plot(time[0:399],jz_expA[1][0:399]/(N_list[1]/2), lw=2, ls='--', alpha=0.75, label=r'$N_{A/B}=20$')
#     ax1.plot(time[0:399],jz_expA[4][0:399]/(N_list[4]/2), lw=2, ls='--', alpha=0.75, label=r'$N_{A/B}=50$')
#     ax1.plot(time[0:399],jz_expA[6][0:399]/(N_list[6]/2), lw=2, ls='--', alpha=0.75, label=r'$N_{A/B}=100$')
#     ax1.plot(time[0:399], msol1[:,2][0:399], 'k', lw=2, alpha=0.75, label=r'$N_{A/B}\rightarrow\infty$')
#     ax1.set_xlabel(r'Time $\kappa t$', fontsize=20)
#     ax1.set_ylabel(r'$\langle \hat{m}^z_A \rangle$', fontsize=20)
#     ax1.set_ylim([-1.1, 1.8])
#     ax1.set_xticks([0, 10, 20, 30, 40])
#     ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
#     ax1.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
#     ax1.legend(loc='best', fontsize=14, ncol=2)
#     ax1.text(-1, 1.4, '(c)', fontsize=20)
#     # --------------
#     # SUBLOT d)
#     # --------------
#     ax2.plot(time[0:399],jz_expB[1][0:399]/(N_list[1]/2), lw=2, ls='--', alpha=0.75)
#     ax2.plot(time[0:399],jz_expB[4][0:399]/(N_list[4]/2), lw=2, ls='--', alpha=0.75)
#     ax2.plot(time[0:399],jz_expB[6][0:399]/(N_list[6]/2), lw=2, ls='--', alpha=0.75)
#     ax2.plot(time[0:399], msol1[:,5][0:399], 'k', lw=2, alpha=0.75)
#     ax2.set_xlabel(r'Time $\kappa t$', fontsize=20)
#     ax2.set_ylabel(r'$\langle \hat{m}^z_B \rangle$', fontsize=20)
#     ax2.set_ylim([-1.1, 1.5])
#     ax2.set_xticks([0, 10, 20, 30, 40])
#     ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
#     ax2.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
#     ax2.text(-1, 1.15, '(d)', fontsize=20)
#     ax2ins = ax2.inset_axes([0.25, 0.4, 0.7, 0.55])
#     ax2ins.plot(time[59:399],jz_expB[1][59:399]/(N_list[1]/2), lw=2, ls='--', alpha=0.75)
#     ax2ins.plot(time[59:399],jz_expB[4][59:399]/(N_list[4]/2), lw=2, ls='--', alpha=0.75)
#     ax2ins.plot(time[59:399],jz_expB[6][59:399]/(N_list[6]/2), lw=2, ls='--', alpha=0.75)
#     ax2ins.plot(time[59:399], msol1[:,5][59:399], 'k', lw=2, alpha=0.75)
#     ax2ins.set_xticklabels([])
#     ax2ins.set_yticklabels([])
#     ax2ins.tick_params(which='both', direction='in')
#     ax2.indicate_inset_zoom(ax2ins, edgecolor="black")
#     # --------------
#     # SUBLOT e)
#     # --------------
#     ax3.plot([0.905, 0.905], [1e-4, 1] , '--k', alpha=0.75)
#     # ax[0].semilogy(-0.905*a,b,'--k')
#     ax3.semilogy(freqs, mz_A_fft_full, lw=3, alpha=0.75, label='A')
#     ax3.semilogy(freqs, mz_B_fft_full, lw=3, alpha=0.75, label='B')
#     # ax[0].semilogy(fftshift(2*np.pi*freq), fftshift(np.abs(signalA)/np.amax(np.abs(signalA))), label='A', lw=3, alpha=0.75)
#     # ax[0].semilogy(fftshift(2*np.pi*freq), fftshift(np.abs(signalB)/np.amax(np.abs(signalB))), label='B', lw=3, alpha=0.75)
#     ax3.set_ylabel(r'$\mathcal{F}$ $\left[\langle \hat{m}^z_{\alpha} \rangle\right]$', fontsize=20)
#     ax3.set_xlabel(r'Frequency $ω$', fontsize=20)
#     ax3.set_xlim([-2, 2])
#     ax3.set_ylim([1e-4, 2])
#     ax3.set_xticks([-1.8, -0.9, 0, 0.9, 1.8])
#     ax3.legend(fontsize=14, ncol=2)
#     ax3.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
#     ax3.text(-1.9, 0.7, '(e)', fontsize=20)
#     # --------------
#     # SUBLOT f)
#     # --------------
#     ax4.plot(np.real(eig_few[0]), np.imag(eig_few[0]), 'd', color='C0', alpha=0.2, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[1]), np.imag(eig_few[1]), 'd', color='C0', alpha=0.3, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[2]), np.imag(eig_few[2]), 'd', color='C0', alpha=0.4, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[3]), np.imag(eig_few[3]), 'd', color='C0', alpha=0.5, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[4]), np.imag(eig_few[4]), 'd', color='C0', alpha=0.6, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[5]), np.imag(eig_few[5]), 'd', color='C0', alpha=0.7, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[6]), np.imag(eig_few[6]), 'd', color='C0', alpha=0.8, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[7]), np.imag(eig_few[7]), 'd', color='C0', alpha=0.9, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few[8]), np.imag(eig_few[8]), 'd', color='C0', markersize=10, markeredgecolor='black', label=rf'$\Gamma/\kappa=0.1$')
#     ax4.plot(np.real(eig_few_crit[0]), np.imag(eig_few_crit[0]), 'o', color='C1', alpha=0.2, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[1]), np.imag(eig_few_crit[1]), 'o', color='C1', alpha=0.3, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[2]), np.imag(eig_few_crit[2]), 'o', color='C1', alpha=0.4, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[3]), np.imag(eig_few_crit[3]), 'o', color='C1', alpha=0.5, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[4]), np.imag(eig_few_crit[4]), 'o', color='C1', alpha=0.6, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[5]), np.imag(eig_few_crit[5]), 'o', color='C1', alpha=0.7, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[6]), np.imag(eig_few_crit[6]), 'o', color='C1', alpha=0.8, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[7]), np.imag(eig_few_crit[7]), 'o', color='C1', alpha=0.9, markersize=10, markeredgecolor='black')
#     ax4.plot(np.real(eig_few_crit[8]), np.imag(eig_few_crit[8]), 'o', color='C1', markersize=10, markeredgecolor='black', label=r'$\Gamma=\Gamma_{\mathrm{crit}}$')
#     ax4.arrow(-0.07, 0.89, 0.04, 0, head_width=0.02, head_length=0.02, color='black', fc='grey', lw=1.5)
#     ax4.text(-0.075, 0.92, '0.89', fontsize=14)
#     ax4.set_ylim(0.35, 1.1)
#     ax4.set_xlim(-0.73, 0)
#     ax4.set_xlabel(r'Re($\lambda$)', fontsize=20)
#     ax4.set_ylabel(r'Im($\lambda$)', fontsize=20)
#     ax4.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
#     ax4.legend(loc='center left', bbox_to_anchor=(0.01, 0.65), ncol=1, fontsize=14)
#     ax4.text(-0.71, 1.01, '(f)', fontsize=20)
#     ax4ins = ax4.inset_axes([0.5, 0.2, 0.45, 0.45])
#     ax4ins.plot(new_inv, i1_fit,'--', color='black', alpha=0.75, label='Fitted')
#     ax4ins.plot(N_list_spec, i1_ev, 'd', color='grey', markeredgecolor='black', label='From data')
#     ax4ins.set_xlabel(r'$N$', fontsize=14)
#     ax4ins.set_ylabel(r'Im($\lambda$)', fontsize=14)
#     ax4ins.legend(loc='upper right', fontsize=12)
#     ax4ins.set_ylim(0.87, 1)
#     ax4ins.set_xticks([10, 30, 50])
#     ax4ins.tick_params(which='both', direction='in', labelsize=14)
#     plt.savefig('figure1_c-f.pdf', transparent=True, bbox_inches='tight')
#     plt.show()

def plot_observable_support(gvals,results,qubits,observable_names,save_folder_name):
    for g in gvals:
        for i in range(len(observable_names)):
            array = results[g][3][i]@results[g][4]
        fig, ax = plt.subplots()
        img = ax.imshow(np.abs(array),cmap='RdYlGn', interpolation='nearest')
        clb = plt.colorbar(img)
        # plt.show()
        clb.ax.tick_params(labelsize=8) 
        clb.ax.set_title('Imag g=%s'%(g),fontsize=8)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('%s/Observablesupport_qubits%s_g%s_obs%s.png'%(save_folder_name,qubits,g,observable_names[i]))
        plt.close()   

