import numpy as np
import scipy
from scipy import linalg
import cvxpy as cp
havedoneqcqpimports = False

'''READ THIS IF YOU WANT TO USE QCQP'''
#Need a very specific set of libraries
#pip install  cvxpy==0.4.9
#pip install CVXcanon==0.1.0
#pip install qcqp
#AFTER THIS: Go to the python files in your computer
#For me its C:\Users\jonat\AppData\Local\Programs\Python\Python38\Lib\site-packages\cvxpy\atoms\log_sum_exp
#Change from scipy.misc import logsumexp to from scipy.special import logsumexp
#Also might require you to get a mosek license and install mosek
#pip install Mosek
#Request personal Academic License at https://www.mosek.com/products/academic-licenses/ and follow instructions

'''2nd attempt'''
#pip install gurobipy
'''
import cvxpy as cvx
#import mosek
from qcqp import *
#import qcqp as qcp
import importlib
importlib.reload(qcqp)
importlib.reload(cvx)'''

def cvxpy_density_matrix_feasibility_sdp_routine(D_matrix,E_matrix,R_matrices,F_matrices,gammas):
    numstate = len(D_matrix)
    D_matrix_np = np.array(D_matrix)
    #D_matrix_np = 0.5*(D_matrix_np + np.conjugate(np.transpose(D_matrix_np)))
    E_matrix_np = np.array(E_matrix)
    #E_matrix_np = 0.5*(E_matrix_np + np.conjugate(np.transpose(E_matrix_np)))
    R_matrices_np = []
    F_matrices_np = []
    for r in R_matrices:
        R_matrices_np.append(np.array(r))
    for f in F_matrices:
        F_matrices_np.append(np.array(f))
    beta = cp.Variable((numstate,numstate),complex=True)
    constraints = [beta >> 0]
    constraints += [cp.trace(E_matrix_np @ beta) == 1]
    constraints += [beta.H == beta]

    a = -1j*(D_matrix_np@beta@E_matrix_np-E_matrix_np@beta@D_matrix_np)
    #finalconstraints = []
    for i in range(len(gammas)):
        a = a + gammas[i]*(R_matrices_np[i]@beta@np.transpose(np.conjugate(R_matrices_np[i]))-0.5*F_matrices_np[i]@beta@E_matrix_np-0.5*E_matrix_np@beta@F_matrices_np[i])
    #a = -1j*(D_matrix_np@beta@E_matrix_np-E_matrix_np@beta@D_matrix_np)+gammas[0]*(R_matrices_np[0]@beta@np.transpose(np.conjugate(R_matrices_np[0]))-0.5*F_matrices_np[0]@beta@E_matrix_np-0.5*E_matrix_np@beta@F_matrices_np[0])
    constraints += [a == 0]
    prob = cp.Problem(cp.Minimize(0),constraints)
    prob.solve(solver=cp.MOSEK,verbose=False)
    denmat = beta.value
    #print(denmat)
    if type(denmat) == type(None):
        return None,None
    else:
        minval = np.trace(denmat@D_matrix_np)
        return denmat,minval

def cvxpy_density_matrix_routine(D_matrix,E_matrix):
    numstate = len(D_matrix)
    #print(numstate)
    D_matrix_np = np.array(D_matrix)
    D_matrix_np = 0.5*(D_matrix_np + np.conjugate(np.transpose(D_matrix_np)))
    E_matrix_np = np.array(E_matrix)
    E_matrix_np = 0.5*(E_matrix_np + np.conjugate(np.transpose(E_matrix_np)))
    #Realification of E
    #E_real = np.real(E_matrix_np)
    #E_imag = np.imag(E_matrix_np)
    #E_realified = np.bmat([[E_real,-E_imag],[E_imag,E_real]])
    #Realification of D
    #D_real = np.real(D_matrix_np)
    #D_imag = np.imag(D_matrix_np)
    #D_realified = np.bmat([[D_real,-D_imag],[D_imag,D_real]])
    beta = cp.Variable((numstate,numstate),hermitian=True)
    #print(np.shape(beta))

    # Define and solve the CVXPY problem.
    constraints = [beta >> 0]
    #constraints += [beta == beta.H]
    constraints += [cp.trace(E_matrix_np @ beta) == 1]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(D_matrix_np @ beta))),constraints)
    prob.solve(solver=cp.MOSEK,verbose=False)
    #Return result.
    #Returns an np array of the density matrix and the min eigenvalue
    #Needs to unrealify
    denmat = beta.value
    #denmat_real = denmat[0:numstate,0:numstate]
    #denmat_imag = denmat[numstate:numstate*2,0:numstate]
    #denmat = denmat_real+1j*denmat_imag
    minval = np.trace(denmat@D_matrix_np)
    return denmat,minval



def gram_schmidt(set_of_vectors, psd_matrix):
    """
    computes the gram schmidt decomposition of a set of vectors w.r.t a psd,
    Hermitian matrix E. Here, the vector space is over the complex field.
    """
    new_set = []
    E = psd_matrix
    for i in range(len(set_of_vectors)):
        new_vec = set_of_vectors[i]
        for j in range(len(new_set)):
            already_done = new_set[j]
            new_vec = new_vec - already_done.conj().T @ E @ new_vec * already_done
        new_vec_norm = np.sqrt(new_vec.conj().T @ E @ new_vec)
        new_vec = new_vec / new_vec_norm 
        new_set.append(new_vec)
    return new_set

def eig_diag_routine(D_matrix, E_matrix, inv_cond = 10**(-2), degeneracy_tol = 5):
    #degeneracy tol is the number of decimal places until I consider an eigenvalue non degenerate
    #Here, its like Toby's diag routine but I write it myself for learning purposes
    #Also, with some modifications (return all the eigvecs rather than just the first one)
    #inversion cutoff, e.g. eigenvalues smaller than this are set to zero
    #this value can be adjusted depending on shot noise, try to find value that fits your noise level
    # inv_cond=10**-12 #i use this value for statevector simulation (no shotnoise)
    # inv_cond=10**-2 #i use this value for shotnoise with ~10000 shots

    e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
            
    #get e_matrix eigenvalues inverted, cutoff with inv_cond
    e_vals_inverted=np.array(e_vals)

    for k in range(len(e_vals_inverted)):
        #print(e_vals_inverted[k])
        #print(inv_cond)
        if(e_vals_inverted[k]<inv_cond):
            e_vals_inverted[k]=0
        else:
            e_vals_inverted[k]=1/e_vals_inverted[k]
            
    #get e_matrix eigenvalues conditioned, such that small/negative eigenvalues are set to zero
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0

    #convert generalized eigenvalue problem with a regular eigenvalue problem using paper "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve D\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*D*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens

    # s_matrix = e_vecs@np.diag(np.sqrt(e_vals_cond))
    s_matrix_inv = np.diag(np.sqrt(e_vals_inverted))@e_vecs.conj().T
    toeigmat = s_matrix_inv @ D_matrix @ s_matrix_inv.conj().T

    #S^-1*D*S^-1^\dag matrix might not be Hermitian if D is not Hermitian. SO use
    #eig instead of eigh
    #We still use the term "qae_energy" because traditionally this is where the generalised eigenvalue problem came from
    qae_energy,qae_vectors=scipy.linalg.eig(toeigmat)


    ini_alpha_vecs = qae_vectors
    ini_alpha_vecs = s_matrix_inv.conj().T @ ini_alpha_vecs

    #Note that after the above procedure, since some of the eigenvectors are in the null-space of the E_matrix (recall that we mapped those "wrong" eigenvalues to 0),
    #those eigenvectors are not what we want. So, we need to throw them away.
    #To find those eigenvectors corresponding to these wrong eigenvalues, because they are in the nullspace of E, they correspond to alphavec^\dag E alphavec = 0
    #Here, we might as well normalise the alpha_vecs,too because they might be
    #like wrong, cause we kinda threw values away
    correct_eigvals = []
    first_index = 0
    for j in range(len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            first_index = j 
            break 
    first_vector = ini_alpha_vecs[:,first_index]
    correct_eigvals.append(qae_energy[first_index])
    after_normalisation = first_vector / np.sqrt(first_vector.conj().T @ E_matrix @ first_vector)

    for j in range(first_index + 1, len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            jth_vector = jth_vector/norm
            correct_eigvals.append(qae_energy[j])
            after_normalisation = np.column_stack((after_normalisation,
            jth_vector))
    #Here, we do gram schimidt on the eigenspaces of dimension 2 and above
    unique_eval_indices = dict()
    for index in range(len(correct_eigvals)):
        val = correct_eigvals[index]
        val = round(val.real,degeneracy_tol) + 1j * round(val.imag, degeneracy_tol)
        if val not in unique_eval_indices.keys():
            unique_eval_indices[val] = [index]
        else:
            unique_eval_indices[val].append(index)
    # print(unique_eval_indices)
    if len(unique_eval_indices.keys()) == len(correct_eigvals):
        #all eigenvalues are unique, don't need to do any gram schmit
        return (np.array(correct_eigvals), after_normalisation)
    else:
        #sadly, have to do gram schmidt..
        print("Doing gram schmidt cause there are some degeneracies")
        for eigval in unique_eval_indices.keys():
            eigenvectors = [] 
            for index in unique_eval_indices[eigval]:
                eigenvectors.append(after_normalisation[:,index])
            new_eigenvectors = gram_schmidt(eigenvectors, E_matrix)
            counter = 0
            for index in unique_eval_indices[eigval]:
                after_normalisation[:,index] = new_eigenvectors[counter]
                counter += 1
        return (np.array(correct_eigvals) ,after_normalisation)

def diag_routine(D_matrix, E_matrix, inv_cond = 10**(-6)):
    #Like toby's diag routine, with some modifications (return all the eigvecs rather than just the first one)
    #inversion cutoff, e.g. eigenvalues smaller than this are set to zero
    #this value can be adjusted depending on shot noise, try to find value that fits your noise level
    # inv_cond=10**-12 #i use this value for statevector simulation (no shotnoise)
    # inv_cond=10**-2 #i use this value for shotnoise with ~10000 shots
    #Here, I'll use eigh directly because now I'm always solving the Hermitian version. I.e, D_matrix must be hermitian

    e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
            
    #get e_matrix eigenvalues inverted, cutoff with inv_cond
    e_vals_inverted=np.array(e_vals)

    for k in range(len(e_vals_inverted)):
        if(e_vals_inverted[k]<inv_cond):
            e_vals_inverted[k]=0
        else:
            e_vals_inverted[k]=1/e_vals_inverted[k]
            
    #get e_matrix eigenvalues conditioned, such that small/negative eigenvalues are set to zero
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0

    #convert generalized eigenvalue problem with a regular eigenvalue problem using paper "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve D\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*D*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens

    # s_matrix = e_vecs@np.diag(np.sqrt(e_vals_cond))
    s_matrix_inv = np.diag(np.sqrt(e_vals_inverted))@e_vecs.conj().T
    toeigmat = s_matrix_inv @ D_matrix @ s_matrix_inv.conj().T

    #S^-1*D*S^-1^\dag matrix might not be Hermitian if D is not Hermitian. SO use
    #eig instead of eigh
    #We still use the term "qae_energy" because traditionally this is where the generalised eigenvalue problem came from
    qae_energy,qae_vectors=scipy.linalg.eigh(toeigmat)

    ini_alpha_vecs = qae_vectors
    ini_alpha_vecs = s_matrix_inv.conj().T @ ini_alpha_vecs

    #Note that after the above procedure, since some of the eigenvectors are in the null-space of the E_matrix (recall that we mapped those "wrong" eigenvalues to 0),
    #those eigenvectors are not what we want. So, we need to throw them away.
    #To find those eigenvectors corresponding to these wrong eigenvalues, because they are in the nullspace of E, they correspond to alphavec^\dag E alphavec = 0
    #Here, we might as well normalise the alpha_vecs,too because they might be
    #like wrong, cause we kinda threw values away
    correct_eigvals = []
    first_index = 0
    for j in range(len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            first_index = j 
            break 
        #else:
        #    print('Thrown away 1')
    #print(first_index)
    first_vector = ini_alpha_vecs[:,first_index]
    correct_eigvals.append(qae_energy[first_index])
    after_normalisation = first_vector / np.sqrt(first_vector.conj().T @ E_matrix @ first_vector)

    for j in range(first_index + 1, len(qae_energy)):
        jth_vector = ini_alpha_vecs[:,j]
        norm = np.sqrt(jth_vector.conj().T @ E_matrix @ jth_vector)
        if np.abs(1-norm) < inv_cond:
            #print('Add 1 in')
            jth_vector = jth_vector/norm
            correct_eigvals.append(qae_energy[j])
            after_normalisation = np.column_stack((after_normalisation,
            jth_vector))
        #else:
        #    print('Thrown away 1')

    return (np.array(correct_eigvals), after_normalisation)

def qcqp_IQAE_routine(D_matrix, E_matrix):
    #global havedoneqcqpimports
    '''if havedoneqcqpimports == False:
        import cvxpy as cvx
        import mosek
        #from qcqp import *
        import qcqp as qcp
        import importlib
        importlib.reload(qcqp)
        importlib.reload(cvx)
        havedoneqcqpimports = True'''
    lengthofalpha = D_matrix.shape[0]
    #Realification of E
    E_real = np.real(E_matrix)
    E_imag = np.imag(E_matrix)
    E_realified = np.bmat([[E_real,-E_imag],[E_imag,E_real]])
    #realification of D
    D_real = np.real(D_matrix)
    D_imag = np.imag(D_matrix)
    D_realified = np.bmat([[D_real,-D_imag],[D_imag,D_real]])
    #Create Variable
    x = cvx.Variable(2*lengthofalpha)
    #Define objective and constraints
    objective = cvx.quad_form(x,D_realified)
    constraints = [cvx.quad_form(x,E_realified)==1]
    #Solve
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    qcqp = QCQP(prob)
    #Here could use SDP relaxation to find a starting point for local methods
    #Although here, we just use a random point
    qcqp.suggest(SDR,solver=cvx.MOSEK)
    #Attempt to improve the starting point given by the suggest method
    f_cd, v_cd = qcqp.improve(COORD_DESCENT)
    result = np.array(x.value)
    #unrealify
    newalphareal = np.array(result[:lengthofalpha])
    newalphaimag = np.array(result[lengthofalpha:])
    newalpha = newalphareal + 1j*newalphaimag
    #normalpha = np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(newalpha)),np.dot(E_matrix,newalpha))))
    #newalpha = newalpha/normalpha
    return newalpha


#OPTIMIZERS MUST RETURN THE UPDATED ALPHAS
def eigh_method_for_TTQS(E_matrix,W_matrix,alphas,inv_cond):
    e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
    e_vals_adjusted=np.array(e_vals)
    e_vals_inverted=np.array(e_vals)
    for k in range(len(e_vals_inverted)):
        if(e_vals_inverted[k]<inv_cond):
            e_vals_inverted[k]=0
        else:
            e_vals_inverted[k]=1/e_vals_inverted[k]
    e_vals_cond=np.array(e_vals)
    for k in range(len(e_vals_cond)):
        if(e_vals_cond[k]<inv_cond):
            e_vals_cond[k]=0
    W_matrix = -W_matrix
    #convert generalized eigenvalue problem with a regular eigenvalue problem using paper "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
    #we want to solve W\alpha=\lambda E\alpha
    #turns out this does not work well if E_matrix has near zero eigenvalues
    #instead, we turn this into regular eigenvalue problem which is more behaved
    #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
    #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
    #solve S^-1*W*S^-1^\dag*a=\lambda a
    #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
    #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens
    s_matrix=np.dot(e_vecs,np.diag(np.sqrt(e_vals_cond)))
    s_matrix_inv=np.dot(np.diag(np.sqrt(e_vals_inverted)),np.transpose(np.conjugate(e_vecs)))
    toeigmat=np.dot(s_matrix_inv,np.dot(W_matrix,np.transpose(np.conjugate(s_matrix_inv))))
    energy,vectors=scipy.linalg.eigh(toeigmat)
    #energy,vectors=scipy.linalg.eig(toeigmat)
    #print(energy)
    #smallestindex = 0
    #minimumev = np.real(energy[0])
    #for i in range(len(energy)):
    #    if np.real(energy[i])<minimumev:
    #        smallestindex = i
    #        minimumev = np.real(energy[i])
    #print(smallestindex)
    ini_alpha_vec=vectors[:,0]
    ini_alpha_vec=np.dot(np.transpose(np.conjugate(s_matrix_inv)),ini_alpha_vec)
    norm_ini_alpha=np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(ini_alpha_vec)),np.dot(E_matrix,ini_alpha_vec))))
    newalpha=ini_alpha_vec/norm_ini_alpha
    return newalpha


#havedoneqcqpimports = False

def qcqp_for_TTQS(E_matrix,W_matrix,alphas,bounddiff=10**(-6)):
    #global havedoneqcqpimports
    '''
    if havedoneqcqpimports == False:
        import cvxpy as cvx
        import mosek
        #from qcqp import *
        import qcqp as qcp
        import importlib
        importlib.reload(qcp)
        importlib.reload(cvx)
        havedoneqcqpimports = True'''
    lengthofalpha = len(alphas)
    #Realification of E
    E_real = np.real(E_matrix)
    E_imag = np.imag(E_matrix)
    E_realified = np.bmat([[E_real,-E_imag],[E_imag,E_real]])
    #realification of W
    W_real = np.real(W_matrix)
    W_imag = np.imag(W_matrix)
    W_realified = np.bmat([[W_real,-W_imag],[W_imag,W_real]])
    #Create Variable
    x = cvx.Variable(2*lengthofalpha)
    #x = cp.Variable((2*lengthofalpha,1))
    #X = x@x.T
    #Define objective and constraints
    objective = cvx.quad_form(x,-W_realified)
    #constraints = [cp.trace(X@E_realified)<=1+bounddiff,1-bounddiff<=cp.trace(X@E_realified)]
    constraints = [cvx.quad_form(x,E_realified)==1]
    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    #Solve
    #prob.solve(solver = cp.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time:  100.0,mosek.iparam.intpnt_solve_form:   mosek.solveform.dual},verbose = False)
    #prob.solve(verbose = True)
    qcqp = QCQP(prob)
    
    #Here could use SDP relaxation to find a starting point for local methods
    #Although here, we just use a random point
    #qcqp.suggest(SDR,solver=cvx.MOSEK)
    qcqp.suggest(SDR)
    #qcqp.suggest(RANDOM)

    #Attempt to improve the starting point given by the suggest method
    f_cd, v_cd = qcqp.improve(COORD_DESCENT)
    #print("Coordinate descent: objective %.3f, violation %.3f" % (f_cd, v_cd))
    #print(x.value)

    #print(result)
    result = np.array(x.value)
    #unrealify
    newalphareal = np.array(result[:lengthofalpha])
    newalphaimag = np.array(result[lengthofalpha:])
    newalpha = newalphareal + 1j*newalphaimag
    #normalpha = np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(newalpha)),np.dot(E_matrix,newalpha))))
    #newalpha = newalpha/normalpha
    return newalpha




