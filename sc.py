import numpy as np
import scipy.special
from numpy.linalg import matrix_rank

class SC:
    """
    This class contains functions to compute the separation capacity as well as the number of homogeneously linearly separable dichtomies of an arbitrary set of points.
    """

    @staticmethod
    def compute_sc(Phi, s, psi = None, eps = 1e-10):
        """
        Phi (function): real-analytic feature map, input shape (M,)  M = dim of codomain of psi 
        s (int): dimension of domain of psi
        psi (function): real-analytic s-dim parametrisation of the domain of Phi; if None, psi = identity
        eps (float): numerical precision, set values < eps to zero
        output (int): separation capacity of Phi 
        """
        if psi == None:
            K = 2*np.size(Phi(np.zeros((s))))
            F = [np.random.rand(s)+1j*np.random.rand(s) for k in range(K)]
        else:
            K = 2*np.size(Phi(psi(np.zeros(s))))
            F = [psi(np.random.rand(s)+1j*np.random.rand(s)) for k in range(K)]
        X = np.column_stack([np.hstack([Phi(f).real, Phi(f).imag]) for f in F])
        X[np.abs(X) < eps] = 0 
        #singular_values = np.linalg.svd(X,compute_uv=False)
        return 2*matrix_rank(X) #2*np.sum(singular_values > eps)

    @staticmethod
    def C(N,M):
        """ 
        N (int): number of points
        M (int): dimension
        output (int): number of homogeneously linearly separable dichotomies if points are in general position 
        """
        return 2*np.sum([SC.binomial(N-1,k) for k in np.arange(M)])
    
    def binomial(n, k):
        b = scipy.special.binom(n,k) 
        if np.isnan(b):
            return 0
        return b
    
    @staticmethod
    def Cv(vec):
        """
        vec (list): list of np arrays 
        output (int): number of homogeneously linearly separable dichotomies of vec based on (Winder, 1966)
        """
        # reset memorisation lists
        SC.E_memo = []
        SC.O_memo = []
        N = len(vec)
        return 2*sum([SC.E(vec,t)[1] for t in range(N+1)])-2**N
    
    """
    Auxiliary functions for Cv
    """
    E_memo = []
    O_memo = []

    def E(vec, t):
        if t < len(SC.E_memo) and SC.E_memo[t] is not None:
            return SC.E_memo[t]

        M = vec[0].size
        if t == 0:
            result = [[np.zeros((M, 1))]], 1
        else:
            res = []
            for vec_list in SC.E(vec, t - 1)[0]:
                vec_mtx = np.column_stack(vec_list)
                r = matrix_rank(vec_mtx)
                for v in vec:
                    if np.any(np.all(vec_mtx == v, axis=0)): 
                        break
                    vec_list_new = vec_list.copy()
                    vec_list_new.append(v)
                    vec_mtx_new = np.column_stack(vec_list_new)
                    if matrix_rank(vec_mtx_new) != r:
                        res.append(vec_list_new)
            if SC.O(vec, t - 1)[1] == 0:
                result = res, len(res)
            else:
                for vec_list in SC.O(vec, t - 1)[0]:
                    vec_mtx = np.column_stack(vec_list)
                    r = matrix_rank(vec_mtx)
                    for v in vec:
                        if np.any(np.all(vec_mtx == v, axis=0)):
                            break
                        vec_list_new = vec_list.copy()
                        vec_list_new.append(v)
                        vec_mtx_new = np.column_stack(vec_list_new)
                        if matrix_rank(vec_mtx_new) == r:
                            res.append(vec_list_new)
                result = res, len(res)

        SC.E_memo.append(result)
        return result

    def O(vec, t):
        if t < len(SC.O_memo) and SC.O_memo[t] is not None:
            return SC.O_memo[t]

        M = vec[0].size
        if t == 0:
            result = [[np.zeros((M, 1))]], 0
        else:
            res = []
            for vec_list in SC.E(vec, t - 1)[0]:
                vec_mtx = np.column_stack(vec_list)
                r = matrix_rank(vec_mtx)
                for v in vec:
                    if np.any(np.all(vec_mtx == v, axis=0)):
                        break
                    vec_list_new = vec_list.copy()
                    vec_list_new.append(v)
                    vec_mtx_new = np.column_stack(vec_list_new)
                    if matrix_rank(vec_mtx_new) == r:
                        res.append(vec_list_new)
            if SC.O(vec, t - 1)[1] == 0:
                result = res, len(res)
            else:
                for vec_list in SC.O(vec, t - 1)[0]:
                    vec_mtx = np.column_stack(vec_list)
                    r = matrix_rank(vec_mtx)
                    for v in vec:
                        if np.any(np.all(vec_mtx == v, axis=0)):
                            break
                        vec_list_new = vec_list.copy()
                        vec_list_new.append(v)
                        vec_mtx_new = np.column_stack(vec_list_new)
                        if matrix_rank(vec_mtx_new) != r:
                            res.append(vec_list_new)
                result = res, len(res)

        SC.O_memo.append(result)
        return result