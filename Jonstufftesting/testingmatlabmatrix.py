import scipy.io 
import scipy.linalg 

matrix = scipy.io.loadmat('Jonstufftesting/'+'savedmatrixfrommatlab.mat')['betarho']
#print(type(matrix))
denmat_values,denmat_vects = scipy.linalg.eig(matrix)
print(denmat_values)