import numpy as np 
import ansatz_class_package as acp 
import pauli_class_package as pcp 
import hamiltonian_class_package as hcp 
import matrix_class_package as mcp 
import scipy
#import qutip as qt
import optimizers as opt_package
from abc import ABC, abstractmethod
from scipy.integrate import ode 
import scipy as scp
import qutip 
''' This is not necessary (But dont delete yet : Jonathan)
#This is stuff to get observables for classicalSimulator
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1j], [1j, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
P = np.array([[0.5, 0.5], [0.5, 0.5]])
II = np.array([[1.0, 0.0], [0.0, 1.0]])

# Function: Perform tensor product, output: I x I x I ....I x A x I
# A: An operator (e.g. Pauli operators)
# k: The position of particle
# N : Total number of particles
def Qtensor(A, k, N_total):
    return np.kron(np.identity(2**(k-1)), np.kron(A, np.identity(2**(N_total-k))))'''

def expectation_val(Qtensormatrix,psi):
    return psi.conj().dot(Qtensormatrix.dot(psi)).real

class classicalSimulator(object):
    """
    I.e, just does e^{-i H t}|\psi(\alpha(t = 0))> for time simulation

    Or just diagonalizes matrix to find ground state

    """
    def __init__(self,N,initialstate,hamiltonian,method='matrix_multiplication'):
        self.N = N
        self.initialstate = initialstate
        self.hamiltonian = hamiltonian
        self.times = None
        self.steps = None 
        self.endtime = None 
        self.results = None
        self.method = method
    def define_endtime(self,endtime):
        self.endtime = endtime
    def numberstep(self,steps):
        self.steps = steps
    
    def find_ground_state(self):
        H = self.hamiltonian.to_matrixform()
        values,vectors = scipy.linalg.eigh(H)
        return (values[0],vectors[:,0])



    def evaluate(self):
        if self.method == 'matrix_multiplication':
            startingstate = self.initialstate.get_statevector()
            self.times = np.linspace(0,self.endtime,num=self.steps)
            H = self.hamiltonian.to_matrixform()
            result = []
            for t in self.times:
                unitary = scipy.linalg.expm(-1j*t*H)
                evolvedstate = unitary@startingstate
                result.append(evolvedstate)
            self.results = result
    
    def get_results(self):#This returns a list of state vectors, the state vectors are the exact results for each timestep
        return self.results

    def get_expectations_observables(self,observablematrix):#This returns a list of the expecations values for each timestep
        expectations = []
        for vector in self.results:
            expectations.append(expectation_val(observablematrix,vector))
        return expectations



class IQAE(object):
    def __init__(self, N, D_matrix, E_matrix):
        """
        These are the evaluated matrices
        """
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.eigh_invcond = None
        self.optimizer = None
        self.ground_state_energy = None 
        self.ground_state_alphas = None
    
    def define_optimizer(self, optimizer, eigh_invcond = 10**(-12)):
        """
        Optimiser can either be eigh or qcqp or sdp or...
        Here we also take in the various invconds as arguments
        """
        self.optimizer = optimizer
        if optimizer == "eigh":
            self.set_eigh_invcond(eigh_invcond)
            
    
    def set_eigh_invcond(self, eigh_invcond):
        self.eigh_invcond = eigh_invcond


    def evaluate(self):
        if self.optimizer == None:
            raise(RuntimeError("run the define optimizer_function first"))

        elif self.optimizer == "eigh":
            qae_energies, qae_eigvecs = opt_package.diag_routine(self.D, self.E, inv_cond=self.eigh_invcond)
            min_energy = qae_energies[0]
            self.ground_state_energy = min_energy
            self.ground_state_alphas = qae_eigvecs[:,0]
        elif self.optimizer == 'qcqp':
            updatedalphas = opt_package.qcqp_IQAE_routine(self.D, self.E)
            self.ground_state_alphas = updatedalphas
            self.ground_state_energy = np.dot(np.conjugate(np.transpose(updatedalphas)),np.dot(self.D,updatedalphas))[0,0]



    
    def get_results(self):
        return (self.ground_state_energy, self.ground_state_alphas)
class IQAE_Lindblad(object):
    def __init__(self, N, D_matrix, E_matrix,R_matrices = [],F_matrices = [],gammas = []):
        """
        These are the evaluated matrices
        """
        if len(R_matrices)!=len(F_matrices) or len(R_matrices)!=len(gammas) or len(F_matrices)!=len(gammas):
            raise(RuntimeError("Number of R_matrices, F_matrices and gammas are not equal"))
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.R_matrices = R_matrices
        self.F_matrices = F_matrices
        self.gammas = gammas
        self.eigh_invcond = None
        self.eig_invcond = None
        self.optimizer = None
        self.sdp_tolerance_bound = 0
        self.ground_state_energy = None 
        self.ground_state_alphas = None
        self.all_energies = None
        self.all_alphas = None
        self.degeneracy_tol = None
        self.density_matrix = None
        self.evaluated_alpha = False
        self.evaluated_denmat = False
        self.addbetaconstraints = None
        self.betainitialpoint = None
    def define_beta_initialpoint(self,initialpoint):
        self.betainitialpoint = initialpoint
    
    def define_additional_constraints_for_feasibility_sdp(self,constraints):
        self.addbetaconstraints = constraints

    def define_optimizer(self, optimizer, eigh_invcond = 10**(-12),eig_invcond = 10**(-12),degeneracy_tol = 5,sdp_tolerance_bound = 0):
        """
        Optimiser can either be eigh or qcqp or sdp or...
        Here we also take in the various invconds as arguments
        """
        self.optimizer = optimizer
        if optimizer == "eigh":
            self.set_eigh_invcond(eigh_invcond)
        elif optimizer == "eig":
            self.set_eig_invcond(eig_invcond)
            self.set_degeneracy_tol(degeneracy_tol)
        elif optimizer == "qcqp":
            pass 
        elif optimizer == 'sdp':
            pass
        elif optimizer == 'feasibility_sdp':
            self.set_sdp_tolerance_bound(sdp_tolerance_bound)
    
    def set_eigh_invcond(self, eigh_invcond):
        self.eigh_invcond = eigh_invcond
    def set_eig_invcond(self,eig_invcond):
        self.eig_invcond = eig_invcond
    def set_sdp_tolerance_bound(self,bound):
        self.sdp_tolerance_bound = bound

    def set_degeneracy_tol(self,degeneracy_tol):
        self.degeneracy_tol = degeneracy_tol

    def evaluate(self):
        if self.optimizer == None:
            raise(RuntimeError("run the define optimizer_function first"))
        elif self.optimizer == 'feasibility_sdp':
            densitymat,minvalue = opt_package.cvxpy_density_matrix_feasibility_sdp_routine(self.D,self.E,self.R_matrices,self.F_matrices,self.gammas,self.sdp_tolerance_bound,additionalbetaconstraints=self.addbetaconstraints,betainitialpoint=self.betainitialpoint)
            self.density_matrix = densitymat
            self.ground_state_energy = minvalue
            self.evaluated_denmat = True
        elif self.optimizer == 'sdp':
            densitymat,minvalue = opt_package.cvxpy_density_matrix_routine(self.D,self.E)
            self.density_matrix = densitymat
            self.ground_state_energy = minvalue
            self.evaluated_denmat = True
        elif self.optimizer == "eigh":
            qae_energies, qae_eigvecs = opt_package.diag_routine(self.D, self.E, inv_cond=self.eigh_invcond)
            min_energy = qae_energies[0]
            self.all_energies = qae_energies
            gsa = []
            for i in range(len(self.all_energies)):
                gsa.append(qae_eigvecs[:,i])
            gsa = np.array(gsa)
            self.all_alphas = gsa
            self.ground_state_energy = min_energy
            self.ground_state_alphas = qae_eigvecs[:,0]
            self.evaluated_alpha = True
        elif self.optimizer == "eig":
            qae_energies, qae_eigvecs = opt_package.eig_diag_routine(self.D, self.E, inv_cond=self.eig_invcond,degeneracy_tol=self.degeneracy_tol)
            #min_energy = qae_energies[0]
            self.all_energies = qae_energies
            gsa = []
            for i in range(len(self.all_energies)):
                gsa.append(qae_eigvecs[:,i])
            gsa = np.array(gsa)
            self.all_alphas = gsa
            #need to sort
            sortinglist = []
            for i in range(len(self.all_energies)):
                sortinglist.append([self.all_energies[i],self.all_alphas[i]])
            sortinglist.sort(key=lambda x: np.abs(x[0]))
            a = []
            b = []
            for i in range(len(sortinglist)):
                a.append(sortinglist[i][0])
                b.append(sortinglist[i][1])
            self.all_energies = np.array(a)
            self.all_alphas = np.array(b)
            self.evaluated_alpha = True
            #return (np.array(a),np.array(b))

    def check_if_hermitian(self, cutoff = 10**(-10)):
        return np.all(np.abs(self.density_matrix - self.density_matrix.conj().T) < cutoff)

    def get_results_all(self):
        if self.evaluated_alpha == False:
            raise(RuntimeError("You did not run the proper optimizer. Remember, if you ran SDP, the results are obtained in the form of density matrix and should use get_density_matrix_results instead"))
        else:
            return (self.all_energies, self.all_alphas)
    def check_if_valid_density_matrix(self,cutoff = 10**(-10)):
        if self.evaluated_denmat == False:
            raise(RuntimeError("You did not run the proper optimizer. Remember, if you ran eigh or eig, the results are obtained in the form of alpha vectors and should use get_results_all instead"))
        else:
            #checking the the density matrix is hermitian
            is_hermitian = self.check_if_hermitian(cutoff = cutoff)
            if is_hermitian:
                #since density matrix is hermitian, we can use eigh
                denmat_values,denmat_vects = scp.linalg.eigh(self.density_matrix)
            else:
                print("density_matrix is not hermitian up to cutoff: "+str(cutoff))
                denmat_values,denmat_vects = scp.linalg.eig(self.density_matrix)
            #print(denmat_values)
            imagvalues = np.imag(denmat_values)
            allreal = True
            for i in imagvalues:
                if i>cutoff:
                    allreal = False
            if allreal == True:
                print('All eigenvalues of beta density matrix are real up to cutoff\n',cutoff)
            else:
                print('Some eigenvalues of beta density matrix are not real, with the cutoff\n',cutoff)
                print('The imaginary parts are: ')
                print(imagvalues)

    def get_density_matrix_results(self):
        if self.evaluated_denmat == False:
            raise(RuntimeError("You did not run the proper optimizer. Remember, if you ran eigh or eig, the results are obtained in the form of alpha vectors and should use get_results_all instead"))
        else:
            return (self.density_matrix,self.ground_state_energy)


#This is an abstract class
class quantumSimulators(ABC):
    @staticmethod
    def helper_getcurrentalphas(alphas):
        result = []
        for al in alphas:
            result.append(al[len(al)-1])
        return np.array(result)

    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        """
        D_matrix and E_matrix here are evaluated matrices
        """
        self.N = N
        self.D = D_matrix
        self.E = E_matrix
        self.startingalphas = startingalphas
        self.steps = None 
        self.endtime = None 
        self.backend = None 
        self.has_it_been_evaluated = False
        self.finishedalphas = None

        #This was required in the past for debugging, but we don't need this now
        print('Ensuring E and D matrices are Hermitian by adding cc')
        self.E = (self.E + self.E.conj().T)/2
        self.D = (self.D + self.D.conj().T)/2

    def numberstep(self, steps):
        self.steps = steps 

    def define_endtime(self,endtime):
        self.endtime = endtime
    
    def define_backend(self,backend):
        self.backend = backend
    
    def get_results(self):
        if self.has_it_been_evaluated == False:
            print("This has not been evaluated yet")
        else:
            return self.finishedalphas

    @abstractmethod
    def define_optimizer(self,optimizer):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class CQFF(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas, method = "diagonalise_H"):
        """
        Method can be either diagonalise_H (which is the main method in https://arxiv.org/pdf/2104.01931.pdf)

        or it can be U_dt (which is the proposed tweak in the discussion, in the same paper)
        """
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.eigh_invcond = None
        self.times = None 
        self.method = method

    def define_optimizer(self, optimizer):
        self.optimizer = optimizer

    def define_eigh_invcond(self, eigh_invcond):
        self.eigh_invcond = eigh_invcond

    def get_times(self):
        return self.times 

    def evaluate(self):

        times=np.linspace(0,self.endtime,num=self.steps, endpoint=True)
        self.times = times
        alphas = self.startingalphas
        initial_alpha = super().helper_getcurrentalphas(alphas)

        if self.optimizer == "eigh":
            eigvals, eigvecs = opt_package.diag_routine(self.D, self.E, inv_cond=self.eigh_invcond)
            # print("eigvals are", eigvals)

            if self.method == "diagonalise_H":
                for t in times:
                    if t == 0:
                        continue
                    U_big_Delta_t_diagterms = np.exp(-1j*eigvals*t)
                    U_big_Delta_t = eigvecs @ np.diag(U_big_Delta_t_diagterms) @ eigvecs.conj().T @ self.E 
                    newalpha = U_big_Delta_t @ initial_alpha
                    for i in range(len(newalpha)):
                        alphas[i].append(newalpha[i])

            elif self.method == "U_dt":
                delta_t = times[1] - times[0]
                no_of_reps = len(times)
                for rep in range(1,no_of_reps + 1):
                    U_big_Delta_t_diagterms = (1 - 1j*eigvals*delta_t)**rep
                    U_big_Delta_t = eigvecs @ np.diag(U_big_Delta_t_diagterms) @ eigvecs.conj().T @ self.E 
                    newalpha = U_big_Delta_t @ initial_alpha
                    for i in range(len(newalpha)):
                        alphas[i].append(newalpha[i])

            self.has_it_been_evaluated = True
            self.finishedalphas = alphas


class QAS(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.p_invcond = 10**(-6) #Default threshold for pseudo-inverse
        self.times = None

    def adot_vector(self, t, avec):
        return -1j*np.linalg.pinv(self.E, rcond = self.p_invcond)@self.D @ avec

    def define_optimizer(self, optimizer):
        self.optimizer = optimizer

    def define_p_invcond(self, p_invcond):
        self.p_invcond = p_invcond

    def get_times(self):
        return self.times 

    def evaluate(self):
        
        times=np.linspace(0,self.endtime,num=self.steps, endpoint=True)
        self.times = times
        alphas = self.startingalphas
        initial_alpha = super().helper_getcurrentalphas(alphas)
        solver = ode(self.adot_vector).set_integrator(self.optimizer)
        solver.set_initial_value(initial_alpha, 0)

        if self.optimizer == "zvode":
            previous_time = None 
            for t in times:
                if t == 0:
                    # newalpha = initial_alpha
                    previous_time = t
                    continue
                else:
                    time_advance_interval = t - previous_time 
                    previous_time = t 
                    newalpha = solver.integrate(solver.t + time_advance_interval)

                for i in range(len(newalpha)):
                    alphas[i].append(newalpha[i])

            self.has_it_been_evaluated = True
            self.finishedalphas = alphas

class TTQS(quantumSimulators):
    def __init__(self, N, D_matrix, E_matrix, startingalphas):
        super().__init__(N, D_matrix, E_matrix, startingalphas)
        self.optimizer = None
        self.invcond = 10**(-6) #Default
        self.times = None 

    def define_optimizer(self,optimizer):
        self.optimizer = optimizer

    def define_invcond(self,ic):
        self.invcond = ic

    def get_times(self):
        return self.times 

    def evaluate(self):

        deltat = self.endtime/(self.steps-1)
        times=np.linspace(0,self.endtime,num=self.steps)
        self.times = times[:] 
        self.G = self.E -1j*deltat*self.D
        alphas = self.startingalphas
        

        for t_idx,t in enumerate(times[:]): #why exclude the endpoint ah (Yeah, this was my mistake, should not be excluding endpoint: Jonathan)
            thisalpha = super().helper_getcurrentalphas(alphas)
            Wtop = np.outer(thisalpha,np.transpose(np.conjugate(thisalpha)))
            Wtop = np.matmul(self.G,Wtop)
            Wtop = np.matmul(Wtop,np.transpose(np.conjugate(self.G)))
            Wbot = np.matmul(np.transpose(np.conjugate(thisalpha)),self.E)
            Wbot = np.matmul(Wbot,thisalpha)
            W_matrix = Wtop/Wbot

            if self.optimizer == 'eigh':
                newalpha = opt_package.eigh_method_for_TTQS(self.E,W_matrix,thisalpha,self.invcond)
                for i in range(len(newalpha)):
                    alphas[i].append(newalpha[i])
            elif self.optimizer == 'qcqp':
                newalpha = opt_package.qcqp_for_TTQS(self.E,W_matrix,thisalpha)
                for i in range(len(newalpha)):
                    alphas[i].append(newalpha[i])

        self.has_it_been_evaluated = True
        self.finishedalphas = alphas

# class TTQS(object):
#     def __init__(self,N,D_matrix,E_matrix,startingalphas):#These are the EVALUATED matrices
#         self.N = N
#         self.D = D_matrix
#         self.E = E_matrix
#         self.invcond = 10**(-6)
#         self.startingalphas = startingalphas
#         self.has_it_been_evaluated = False
    
#     def numberstep(self,steps):
#         self.steps = steps
    
#     def define_endtime(self,endtime):
#         self.endtime = endtime
    
#     def define_optimizer(self,optimizer):
#         self.optimizer = optimizer

#     def define_backend(self,backend):
#         self.backend = backend
    
#     def define_invcond(self,ic):
#         self.invcond = ic

#     def evaluate(self):
        

#         deltat = self.endtime/(self.steps-1)
#         times=np.linspace(0,self.endtime,num=self.steps)
#         self.G = self.E -1j*deltat*self.D
#         alphas = self.startingalphas
        

#         for t_idx,t in enumerate(times[:-1]):
#             thisalpha = helper_getcurrentalphas(alphas)
#             Wtop = np.outer(thisalpha,np.transpose(np.conjugate(thisalpha)))
#             Wtop = np.matmul(self.G,Wtop)
#             Wtop = np.matmul(Wtop,np.transpose(np.conjugate(self.G)))
#             Wbot = np.matmul(np.transpose(np.conjugate(thisalpha)),self.E)
#             Wbot = np.matmul(Wbot,thisalpha)
#             W_matrix = Wtop/Wbot

#             if self.optimizer == 'eigh':
#                 newalpha = opt_package.eigh_method_for_TTQS(self.E,W_matrix,thisalpha,self.invcond)
#                 for i in range(len(newalpha)):
#                     alphas[i].append(newalpha[i])
#         self.has_it_been_evaluated = True
#         self.finishedalphas = alphas

#     def get_results(self):
#         if self.has_it_been_evaluated == False:
#             print("This has not been evaluated yet")
#         else:
#             return self.finishedalphas

def evaluate_rho_dot(rho, hamiltonian_class_object, gammas, L_terms):
    hamiltonian_mat = hamiltonian_class_object.to_matrixform()
    coherent_evo = -1j * (hamiltonian_mat @ rho - rho @ hamiltonian_mat)
    quantum_jumps_total = 0 + 0*1j
    for i in range(len(gammas)):
        gamma_i = gammas[i]
        L_i_mat = L_terms[i].to_matrixform()
        L_i_dag_L_i = L_i_mat.conj().T @ L_i_mat
        anti_commutator = L_i_dag_L_i @ rho + rho @ L_i_dag_L_i
        jump_term = L_i_mat @ rho @ L_i_mat.conj().T
        quantum_jumps_total += gamma_i * (jump_term - 0.5*anti_commutator)
    return coherent_evo + quantum_jumps_total

def analyze_density_matrix(num_qubits,initial_state,IQAE_instance,E_mat_evaluated,ansatz,hamiltonian,gammas,L_terms,qtp_rho_ss,O_matrices_evaluated):
    density_mat,groundstateenergy = IQAE_instance.get_density_matrix_results()
    if type(density_mat) == type(None):
        print('SDP failed for this run, probably due to not high enough K')
    else:
        result_dictionary = {}
        IQAE_instance.check_if_valid_density_matrix()
        #print(all_energies)
        #print(all_states)
        print("the trace of the beta matrix is", np.trace(density_mat @ E_mat_evaluated))
        result_dictionary['trace_beta']=np.trace(density_mat @ E_mat_evaluated)
        print('The ground state energy is\n',groundstateenergy)
        result_dictionary['ground_state_energy']=groundstateenergy
        #print('The density matrix is\n',density_mat)
        result_dictionary["beta"] = density_mat
        if IQAE_instance.check_if_hermitian() == True:
            denmat_values,denmat_vects = scp.linalg.eigh(density_mat)
        else:
            denmat_values,denmat_vects = scp.linalg.eig(density_mat)
        denmat_values = np.real(np.round(denmat_values,10))
        #print(np.imag(denmat_values))
        print("the sorted density matrix (beta matrix) eigenvalues are\n",np.sort(denmat_values))
        result_dictionary['sorted_beta_eigenvalues'] = np.sort(denmat_values)
        p_string_matrices = [i.get_paulistring().get_matrixform() for i in ansatz.get_moments()]
        ini_statevec_vecform = initial_state.get_statevector()
        csk_states = [i@ini_statevec_vecform for i in p_string_matrices]
        rho = np.zeros(shape=(2**num_qubits,2**num_qubits), dtype = np.complex128)
        trace = 0
        for i in range(len(density_mat)):
            for j in range(len(density_mat)):
                i_j_entry = density_mat[(i,j)]
                i_j_ketbra = np.outer(csk_states[i], csk_states[j].conj().T)
                rho += i_j_entry * i_j_ketbra
                trace += i_j_entry * csk_states[j].conj().T @ csk_states[i]

        rho_eigvals,rho_eigvecs = scipy.linalg.eigh(rho)        
        print('rho_eigvals is: ' + str(rho_eigvals))
        result_dictionary['rho_eigvals'] = rho_eigvals
        print("trace rho is", np.trace(rho))
        result_dictionary['trace_rho'] = np.trace(rho)
        #now, we check if rho (the actual denmat) gives 0 for the linblad master equation
        #save the rho for processing
        result_dictionary["rho"] = rho
        rho_dot = evaluate_rho_dot(rho, hamiltonian, gammas, L_terms) #should be 0
        # print('rho_dot is: ' + str(rho_dot))
        print('Max value rho_dot is: ' + str(np.max(np.max(rho_dot))))
        result_dictionary['max_rho_dot'] = np.max(np.max(rho_dot))
        qtp_rho = qutip.Qobj(rho)
        fidelity = qutip.metrics.fidelity(qtp_rho, qtp_rho_ss)
        print("The fidelity is", fidelity)
        result_dictionary['fidelity'] = fidelity
        result_dictionary['observable_expectation'] = [np.trace(density_mat @ O_mat_eval) for O_mat_eval in O_matrices_evaluated]
        #observable_expectation_results[k] = [np.trace(density_mat @ O_mat_eval) for O_mat_eval in O_matrices_evaluated]
        #fidelity_results[k] = fidelity
        #if round(fidelity, 6) == 1:
        #    print("breaking loop as fidelity = 1 already")
        #    #raise(RuntimeError("Fidelity = 1!"))
        #    break
        return result_dictionary















