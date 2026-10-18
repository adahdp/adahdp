# Copyright (C) 2019-2021, Tumult Labs Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import hdmm_workload as workload
import hdmm_templates as templates

def get_domain(W):
    if isinstance(W, workload.Kronecker):
        return tuple(Q.shape[1] for Q in W.matrices)
    elif isinstance(W, workload.Weighted):
        return get_domain(W.base)
    elif isinstance(W, workload.VStack):
        return get_domain(W.matrices[0])
    else:
        return W.shape[1]

class HDMM:

    def __init__(self, W, x, eps, seed=0):
        self.delta = None
        self.lambda_min = None
        self.domain = get_domain(W)
        self.W = W
        self.x = x
        self.eps = eps
        self.prng = np.random.RandomState(seed)
        self.variance_term = None
        self.average_variance = None

    def compute_statistics(self) -> None:
        """
        Compute privacy-free statistics of the optimized HDMM strategy.
        """

        if not hasattr(self, "strategy"):
            raise RuntimeError("The strategy must be optimized before computing statistics.")

        strategy = self.strategy

        if hasattr(strategy, "dense_matrix"):
            dense_strategy = np.asarray(strategy.dense_matrix(), dtype=float,)
        elif hasattr(strategy, "matrix"):
            dense_strategy = np.asarray(strategy.matrix, dtype=float,)
        else:
            dense_strategy = np.asarray(strategy, dtype=float,)

        if dense_strategy.ndim != 2:
            raise ValueError("The strategy matrix must be two-dimensional.")

        workload = self.W

        if hasattr(workload, "dense_matrix"):
            dense_workload = np.asarray(
                workload.dense_matrix(),
                dtype=float,
            )
        elif hasattr(workload, "matrix"):
            dense_workload = np.asarray(workload.matrix, dtype=float,)
        else:
            dense_workload = np.asarray(workload, dtype=float,)

        if dense_workload.ndim != 2:
            raise ValueError("The workload matrix must be two-dimensional.")

        self.delta = float(strategy.sensitivity())

        gram = (dense_strategy.T @ dense_strategy)
        eigenvalues = np.linalg.eigvalsh(gram)
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]

        if positive_eigenvalues.size == 0:
            raise ValueError("The optimized strategy has no positive Gram eigenvalues.")

        self.lambda_min = float(positive_eigenvalues[0])

        gram_pinv = np.linalg.pinv(gram, rcond=1e-12,)

        self.variance_term = float(
            np.trace(
                dense_workload
                @ gram_pinv
                @ dense_workload.T
            )
        )

        if not np.isfinite(self.variance_term) or self.variance_term < 0.0:
            raise ValueError("The HDMM variance term is invalid.")

    def optimize(self, restarts=1):
        W = self.W
        if type(self.domain) is tuple: # kron or union kron workload
            ns = self.domain

            ps = [max(1, n//16) for n in ns]
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                kron = templates.KronPIdentity(ps, ns)
                lossk = kron.optimize(W)
                if lossk < best_loss:
                    best_loss = lossk
                    best_strat = kron

                marg = templates.Marginals(ns)
                lossm = marg.optimize(W)
                if lossm < best_loss:
                    best_loss = lossm
                    best_strat = marg


            # multiplicative factor puts losses on same scale
            self.strategy = best_strat.strategy()
        else:
            n = self.domain
            best_strat, best_loss = None, np.inf
            for _ in range(restarts):
                pid = templates.PIdentity(max(1, n//16), n)
                loss = pid.optimize(W)
                if loss < best_loss:
                    best_loss = loss
                    best_strat = pid
            self.strategy = best_strat.strategy()
           
    # def run(self):
    #     A = self.strategy
    #     A1 = A.pinv()
    #     delta = self.strategy.sensitivity()
    #     noise = self.prng.laplace(loc=0.0, scale=delta/self.eps, size=A.shape[0])
    #     self.ans = A.dot(self.x) + noise
    #     self.xest = A1.dot(self.ans)
    #     return self.xest

    def run(self):
        if not hasattr(self, "strategy"):
            raise RuntimeError("The HDMM strategy must be optimized before run().")

        if self.delta is None or self.lambda_min is None:
            self.compute_statistics()

        strategy = self.strategy
        strategy_pinv = strategy.pinv()

        noise = self.prng.laplace(
            loc=0.0,
            scale=self.delta / self.eps,
            size=strategy.shape[0],
        )

        self.ans = strategy.dot(self.x) + noise
        self.xest = strategy_pinv.dot(self.ans)

        return self.xest
 
