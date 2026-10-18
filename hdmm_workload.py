# This file is part of the Ektelo framework 
# For licensing terms please see: https://github.com/ektelo/ektelo
#
# Copyright 2019-2021, Tumult Labs Inc.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hdmm_matrix as matrix
from hdmm_matrix import EkteloMatrix, Identity, Ones, VStack, Kronecker, Sum, Weighted
import collections
from functools import reduce
import itertools
import numpy as np
from scipy.special import binom
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

def Total(n, dtype=np.float64):
    """
    The 1 x n matrix of 1s
    :param n: the domain size
    :return: the query matrix
    """
    return Ones(1,n,dtype)

def IdentityTotal(n, weight=1.0, dtype=np.float64):
    """
    The matrix [I; w*T] where w is the weight on the total query
    :param n: the domain size
    :param weight: the weight on the total query
    :return: the query matrix
    """
    I = Identity(n, dtype)
    T = Total(n, dtype)
    w = dtype(weight)
    return VStack([I, w*T])





class Permuted(EkteloMatrix):
    def __init__(self, base, seed=0):
        self.base = base
        prng = np.random.RandomState(seed)
        self.idx = prng.permutation(base.shape[1])
        self.shape = base.shape
        self.dtype = base.dtype

    @property
    def matrix(self):
        return self.base.dense_matrix()[:,self.idx]
   
    def gram(self):
        WtW = self.base.gram().dense_matrix()
        return EkteloMatrix(WtW[self.idx,:][:,self.idx])

class Marginal(Kronecker):
    def __init__(self, domain, key):
        """
        :param domain: a d-tuple containing the domain size of the d attributes
        :param key: a integer key 0 <= key < 2^d identifying the marginal
        """
        self.domain = tuple(domain)
        self.key = key
        binary = self.binary()
        self._axes = tuple(i for i in range(len(binary)) if binary[i] == 0)
        subs = []
        for i,n in enumerate(domain):
            if binary[i] == 0:
                subs.append(Total(n))
            else:
                subs.append(Identity(n))
        Kronecker.__init__(self, subs)

    def _matmat(self, V):
        tensor = V.reshape(*self.domain, V.shape[1])
        return tensor.sum(axis=self._axes).reshape(-1, V.shape[1])

    def _rmatmat(self, V):  
        newdom = tuple(self.domain[i] if i in self.tuple() else 1 for i in range(len(self.domain)))
        tensor = V.reshape(*newdom, V.shape[1])
        ans = np.broadcast_to(tensor, self.domain + (V.shape[1],))
        return ans.reshape(-1, V.shape[1])

    def _transpose(self):
        ans = Kronecker._transpose(self)
        ans._matmat = self._rmatmat
        return ans

    def binary(self):
        i = self.key
        d = len(self.domain)
        return tuple([int(bool(2**k & i)) for k in range(d)])[::-1]

    def tuple(self):
        binary = self.binary()
        d = len(self.domain)
        return tuple(i for i in range(d) if binary[i] == 1)

    @staticmethod
    def frombinary(domain, binary):
        d = len(domain)
        key = sum(binary[k]*2**(d-k-1) for k in range(d))
        return Marginal(domain, key)

    @staticmethod
    def fromtuple(domain, attrs, columns=None):
        if not columns:
            binary = [1 if i in attrs else 0 for i in range(len(domain))]
        else:
            binary = [1 if i in attrs else 0 for i in columns]
        return Marginal.frombinary(domain, binary)

class Marginals(VStack):
    def __init__(self, domain, weights):
        self.domain = tuple(domain)
        self.weights = weights
        subs = []
        for key, wgt in enumerate(weights):
            if wgt > 0: subs.append(wgt * Marginal(domain, key))
        VStack.__init__(self, subs)   

    def gram(self):
        return MarginalsGram(self.domain, self.weights**2)
   
    def pinv(self):
        # note: this is a generalized inverse, but not necessarily the pseudo inverse
        return self.gram().pinv() * self.T

    def sensitivity(self):
        return self.weights.sum()

    @staticmethod 
    def frombinary(domain, weights):
        vect = np.zeros(2**len(domain))
        for binary, wgt in weights.items():
            M = Marginal.frombinary(domain, binary)
            vect[M.key] = wgt
        return Marginals(domain, vect)

    @staticmethod
    def fromtuples(domain, weights, columns=None):
        vect = np.zeros(2**len(domain))
        for tpl, wgt in weights.items():
            M = Marginal.fromtuple(domain, tpl, columns)
            vect[M.key] = wgt
        return Marginals(domain, vect)
   
    @staticmethod
    def approximate(W):
        """
        Given a Union-of-Kron workload, find a Marginals workload that approximates it.
        
        The guarantee is that for all marginals strategies A, Error(W, A) = Error(M, A) where
        M is the returned marginals approximation of W.
        The other guarantee is that this function is idempotent: approx(approx(W)) = approx(W)
        """
        WtW = MarginalsGram.approximate(W.gram())
        return Marginals(WtW.domain, np.sqrt(WtW.weights))

class MarginalsGram(Sum):
    def __init__(self, domain, weights):
        self.domain = tuple(domain)
        self.weights = weights
        subs = []
        n, d = np.prod(domain), len(domain)
        mult = np.ones(2**d)
        for key, wgt in enumerate(weights):
            Q = Marginal(domain, key)
            mult[key] = n / Q.shape[0]
            if wgt != 0: subs.append(wgt * Q.gram())
        self._mult = mult

        Sum.__init__(self, subs)

    def _matmat(self, V):
        tensor = V.reshape(*self.domain, V.shape[1])
        ans = self.weights[-1] * tensor
        d = len(self.domain)
        for key in range(2**d-1):
            binary = tuple([int(bool(2**k & key)) for k in range(d)])[::-1]
            axes = tuple(i for i in range(len(binary)) if binary[i] == 0)
            if self.weights[key] != 0:
                tmp = tensor.sum(axis=axes, keepdims=True)
                tmp *= self.weights[key]
                ans += tmp
        return ans.reshape(-1, V.shape[1])

    def _Xmatrix(self,vect):
        # the matrix X such that M(u) M(v) = M(X(u) v)
        d = len(self.domain)
        A = np.arange(2**d)
        mult = self._mult

        values = np.zeros(3**d)
        rows = np.zeros(3**d, dtype=int)
        cols = np.zeros(3**d, dtype=int)
        start = 0
        for b in range(2**d):
            #uniq, rev = np.unique(a&B, return_inverse=True) # most of time being spent here
            mask = np.zeros(2**d, dtype=int)
            mask[A&b] = 1
            uniq = np.nonzero(mask)[0]
            step = uniq.size
            mask[uniq] = np.arange(step)
            rev = mask[A&b]
            values[start:start+step] = np.bincount(rev, vect*mult[A|b], step)
            cols[start:start+step] = b
            rows[start:start+step] = uniq
            start += step
        X = sparse.csr_matrix((values, (rows, cols)), (2**d, 2**d))
        XT = sparse.csr_matrix((values, (cols, rows)), (2**d, 2**d))
        return X, XT

    def __mul__(self, other):
        if isinstance(other, MarginalsGram) and self.domain == other.domain:
            X, XT = self._Xmatrix(self.weights)
            vect = X.dot(other.weights)
            return MarginalsGram(self.domain, vect)
        else:
            return EkteloMatrix.__mul__(self, other)

    def inv(self):
        assert self.weights[-1] != 0, 'matrix is not invertible'
        X, _ = self._Xmatrix(self.weights)
        z = np.zeros_like(self.weights)
        z[-1] = 1.0
        phi = spsolve_triangular(X, z, lower=False)
        return MarginalsGram(self.domain, phi)

    def ginv(self):
        w = self.weights
        X, _ = self._Xmatrix(w)
        idx = X.dot(np.ones(w.size)) != 0
        X = X[idx,:][:,idx] # to make solve_triangular work
        phi = spsolve_triangular(X, w[idx], lower=False)
        phi = spsolve_triangular(X, phi, lower=False)
        ans = np.zeros(w.size)
        ans[idx] = phi
        return MarginalsGram(self.domain, ans)

    def pinv(self):
        return self.ginv()

    def trace(self):
        return self.weights.sum() * self.shape[1]

    @staticmethod
    def approximate(WtW):
        """
        Given a Sum-of-Kron matrix, find a MarginalsGram object that approximates it.
        """
        WtW = sum_kron_canonical(WtW)
        dom = tuple(Wi.shape[1] for Wi in WtW.matrices[0].base.matrices)
        weights = np.zeros(2**len(dom))
        for sub in WtW.matrices:
            tmp = []
            for n, piece in zip(dom, sub.base.matrices):
                X = piece.dense_matrix()
                b = float(X.sum() - X.trace()) / (n * (n-1))
                a = float(X.trace()) / n - b
                tmp.append(np.array([b,a]))
            weights += sub.weight * reduce(np.kron, tmp)
        return MarginalsGram(dom, weights)
 
class AllNormK(EkteloMatrix):
    def __init__(self, n, norms, dtype=np.float64):
        """
        All predicate queries that sum k elements of the domain
        :param n: The domain size
        :param norms: the L1 norm (number of 1s) of the queries (int or list of ints)
        """
        self.n = n
        if type(norms) is int:
            norms = [norms]
        self.norms = norms
        self.m = int(sum(binom(n, k) for k in norms))
        self.shape = (self.m, self.n)
        self.dtype = dtype

    @property
    def matrix(self):
        Q = np.zeros((self.m, self.n))
        idx = 0
        for k in self.norms:
            for q in itertools.combinations(range(self.n), k):
                Q[idx, q] = 1.0
                idx += 1
        return Q

    def gram(self):
        # WtW[i,i] = binom(n-1, k-1) (1 for each query having q[i] = 1)
        # WtW[i,j] = binom(n-2, k-2) (1 for each query having q[i] = q[j] = 1)
        n = self.n
        diag = sum(binom(n-1, k-1) for k in self.norms)
        off = sum(binom(n-2, k-2) for k in self.norms)
        return off*Ones(n,n) + (diag-off)*Identity(n)

class Disjuncts(Sum):
    """
    Just like the Kron workload class can represent a cartesian product of predicate counting
    queries where the predicates are conjunctions, this workload class can represent a cartesian
    product of predicate counting queries where the predicates are disjunctions.
    """
    #TODO: check implementation after refactoring
    def __init__(self, workloads):
        # q or r = - (-q and -r)
        # W = 1 x 1 - (Q1 x R1)

        self.A = Kronecker([Ones(*W.shape) for W in workloads]) # totals
        self.B = -1*Kronecker([Ones(*W.shape) - W for W in workloads]) # negations
        Sum.__init__(self, [self.A, self.B])

    def gram(self):
        return Sum([self.A.gram(), self.A.T @ self.B, self.B.T @ self.A, self.B.gram()])

class ExplicitGram:
    # not an Ektelo Matrix, but behaves like it in the sense that it has gram function,
    # meaning strategy optimization is possible
    def __init__(self, matrix, queries=None):
        self.matrix = matrix
        self.shape = (queries, matrix.shape[1])
        self.dtype = matrix.dtype
    
    def gram(self):
        return EkteloMatrix(self.matrix)

def Moments(n, k=3):
    N = np.arange(n)
    K = np.arange(1,k+1)
    W = N[None]**K[:,None]
    return EkteloMatrix(W)
 


def DimKMarginals(domain, dims):
    if type(dims) is int:
        dims = [dims]
    weights = {}
    for key in itertools.product(*[[1,0]]*len(domain)):
        if sum(key) in dims:
            weights[key] = 1.0
    return Marginals.frombinary(domain, weights)



def union_kron_canonical(W):
    if isinstance(W, Kronecker):
        return VStack([1.0 * W])
    elif isinstance(W, Weighted) and isinstance(W.base, Kronecker):
        return VStack([W])
    elif isinstance(W, VStack):
        return VStack([1.0 * X for X in W.matrices])
    elif isinstance(W, Weighted) and isinstance(W.base, VStack):
        c = W.weight
        return VStack([c * X for X in W.base.matrices])
    else:
        raise ValueError('Input format not recognized')

def sum_kron_canonical(WtW):
    if isinstance(WtW, Kronecker):
        return Sum([1.0 * WtW])
    elif isinstance(WtW, Weighted) and isinstance(WtW.base, Kronecker):
        return Sum([WtW])
    elif isinstance(WtW, Sum):
        return Sum([1.0 * X for X in WtW.matrices])
    elif isinstance(WtW, Weighted) and isinstance(WtW.base, Sum):
        c = WtW.weight
        return Sum([c * X for X in WtW.base.matrices])
    else:
        raise ValueError('Input format not recognized')
