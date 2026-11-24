import re
import os
import GPy
import numpy as np
import casadi as ca
from typing import Tuple

from utils import GaussianConfig
from utils.matrix_methods import cholesky_add_row

class Normalizer:
    def __init__(self, data: np.ndarray) -> None:
        self.mean = data.mean()
        self.std = data.std()
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def inverse_mean(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean
    
    def inverse_variance(self, data: np.ndarray) -> np.ndarray:
        return data * (self.std ** 2)

class SparseFITCGP:
    def __init__(self, kernel: GPy.kern, gaussian_opts: GaussianConfig) -> None:
        self.kern = kernel
        self.gaussian_opts = gaussian_opts

    def fit(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        U: np.ndarray,
        variance: float = 1e-6,
        lambda_: float = 1.0
    ) -> None:
        
        print(f'[GPR]: Fitting Sparse FITC GP Model with normalized: {self.gaussian_opts.normalize_y}')

        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self.U = np.atleast_2d(U)
        self.lambda_ = lambda_
        self.variance = variance

        if self.gaussian_opts.normalize_y is True:
            self.normalizer = Normalizer(self.Y)
            self.Y = self.normalizer.normalize(self.Y)
        else:
            self.normalizer = None

        self.M = len(self.U)
        self.N = len(self.X)

        # Core kernel matrices
        self.Kuu = self.kern.K(self.U)
        self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry
        self.Kuf = self.kern.K(self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-6 * np.eye(self.M)
        self.Luu = np.linalg.cholesky(self.Kuu + jitter)

        # --- Factorized form ---
        # W = Luu^{-1} Kuf
        self.W = np.linalg.solve(self.Luu, self.Kuf)

        self.Kff_diag = self.kern.Kdiag(self.X)
        # self.Qff_diag = np.diag(self.Kuf.T @ self.Kuu_inv @ self.Kuf)
        self.Qff_diag = np.power(self.W.T, 2).sum(axis=-1)
        self.diag = self.Kff_diag - self.Qff_diag + self.variance
        self.diag_inv = 1.0 / self.diag
        self.W_d_inv_y = (self.W * self.diag_inv) @ self.Y

        # K = I + W D^{-1} W^T
        # WD_half = self.W @ np.sqrt(self.D)
        self.K = (self.W * self.diag_inv) @ self.W.T
        self.K = (self.K + self.K.T) / 2 + np.eye(self.M) # ensure symmetry

        # Cholesky of K
        self.L = np.linalg.cholesky(self.K)

        self.L_updated = True

        self.update_mu_su(from_scratch=True)

    # ------------------------------------------------------------------

    def update_mu_su(
        self,
        from_scratch: bool = False,
        Kun: np.ndarray | None = None,
        Knn: np.ndarray | None = None,
    ) -> None:
        if from_scratch:
            self.update_L()

            # self.Kuu = self.kern.K(self.U)
            # self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry

            # self.Kuu_inv = np.linalg.solve(
            #     self.Luu.T, np.linalg.solve(self.Luu, np.eye(self.M))
            # )

            # Posterior covariance Su
            # Sigma = inv[Kuu + Kuf @ inv(D) @ Kfu]
            #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
            #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
            #   = inv(Luu).T @ inv(K) @ inv(Luu)
            #   = Luu^{-T} L^{-T} L^{-1} Luu^{-1}
            # Su = Kuu @ Sigma @ Kuu = Luu L^{-T} L^{-1} Luu.T = tmp.T @ tmp
            tmp = np.linalg.solve(self.L, self.Luu.T)
            # tmp = np.linalg.solve(self.L, np.linalg.solve(self.Luu, self.Kuu))
            self.Su = tmp.T @ tmp  # symmetric MxM

            # Posterior mean mu
            # mu = Su @ Kuu^{-1} Kuf D^{-1} Y
            # rhs = (self.Kuf * self.diag_inv) @ self.Y
            # self.mu = self.Su @ (self.Kuu_inv @ rhs)
            rhs = np.linalg.solve(self.Luu, (self.Kuf * self.diag_inv) @ self.Y)
            self.mu = np.linalg.solve(self.Luu, self.Su.T).T @ rhs
        else:
            # TODO: if new inducing points are too many, then the update will have some error
            # TODO: if new inducing is close to old ones, numerical issues may happen
            Luu_inv_Kun = np.linalg.solve(self.Luu, Kun)
            Kuu_inv_kun = np.linalg.solve(self.Luu.T, Luu_inv_Kun)
            mu_new = Kuu_inv_kun.T @ self.mu
            self.mu = np.vstack((self.mu, mu_new))

            Sun = self.Su @ Kuu_inv_kun
            Snn = Knn - Luu_inv_Kun.T @ Luu_inv_Kun + Kuu_inv_kun.T @ self.Su @ Kuu_inv_kun
            left = np.vstack((self.Su, Sun.T))
            right = np.vstack((Sun, Snn))
            self.Su = np.hstack((left, right))

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray, method: str = 'cholesky', compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        if method == 'cholesky':
            return self.predict_cholesky(Xs)
        elif method == 'mu_su':
            return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def predict_noiseless(self, Xs: np.ndarray, full_cov: bool, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def save_model(self, id: str) -> None:
        print(f'[GPR]: Saving Sparse FITC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        save_path = self.gaussian_opts.save_path + id + '.npy'
        with open(save_path, 'wb') as file:
            np.save(file, self.N)
            np.save(file, self.M)
            np.save(file, self.variance)
            np.save(file, self.lambda_)
            np.save(file, self.X)
            np.save(file, self.Y)
            np.save(file, self.U)

            if self.gaussian_opts.normalize_y is True:
                np.save(file, self.normalizer.mean)
                np.save(file, self.normalizer.std)
            
            np.save(file, self.Luu)
            np.save(file, self.mu)
            np.save(file, self.Su)
            np.save(file, self.Kuf)
            np.save(file, self.W)
            np.save(file, self.diag)
            np.save(file, self.diag_inv)
            np.save(file, self.W_d_inv_y)
            np.save(file, self.L)

    # ------------------------------------------------------------------

    def load_model(self, id: str) -> None:
        print(f'[GPR]: Loading Sparse FITC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        load_path = self.gaussian_opts.load_path + id + '.npy'
        with open(load_path, 'rb') as file:
            self.N = np.load(file)
            self.M = np.load(file)
            self.variance = np.load(file)
            self.lambda_ = np.load(file)
            self.X = np.load(file)
            self.Y = np.load(file)
            self.U = np.load(file)

            if self.gaussian_opts.normalize_y is True:
                mean = np.load(file)
                std = np.load(file)
                self.normalizer = Normalizer(self.Y)
                self.normalizer.mean = mean
                self.normalizer.std = std
            else:
                self.normalizer = None
            
            self.Luu = np.load(file)
            self.mu = np.load(file)
            self.Su = np.load(file)
            self.Kuf = np.load(file)
            self.W = np.load(file)
            self.diag = np.load(file)
            self.diag_inv = np.load(file)
            self.W_d_inv_y = np.load(file)
            self.L = np.load(file)
        self.L_updated = True

    # ------------------------------------------------------------------
    
    def dk_dx(self, Xs: np.ndarray) -> np.ndarray:
        mean_jac = np.zeros((self.U.shape[0], self.U.shape[1]))
        for i in range(self.U.shape[1]):
            mean_jac[:, i] = self.kern.dK_dX2(self.U, Xs, i).squeeze()
        return mean_jac
    
    # ------------------------------------------------------------------
    
    def dmu_dx(self, Xs: np.ndarray) -> np.ndarray:
        rhs = np.linalg.solve(self.Luu, self.mu)
        dk_dx = self.dk_dx(Xs)
        lhs = np.linalg.solve(self.Luu, dk_dx)
        result = lhs.T @ rhs
        return result
    
    # ------------------------------------------------------------------
    
    def dvar_dx(self, Xs: np.ndarray, Ws: np.ndarray, tmp: np.ndarray) -> np.ndarray:
        dk_dx = self.dk_dx(Xs)
        Luu_inv_dk_dx = np.linalg.solve(self.Luu, dk_dx)
        Kuu_inv_dk_dx = np.linalg.solve(self.Luu.T, Luu_inv_dk_dx)
        result = -2 * Luu_inv_dk_dx.T @ Ws + 2 * Kuu_inv_dk_dx.T @ self.Su @ tmp
        return result
    
    # ------------------------------------------------------------------

    def predict_mu_su(self, Xs: np.ndarray, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        '''
        Predict mean and variance at new test points.

        Parameters
        ----------
        Xs : np.ndarray, shape (N*, D)
            Test inputs.

        Returns
        -------
        mu : np.ndarray, shape (N*,)
            Predictive mean.
        var : np.ndarray, shape (N*,)
            Predictive variance.
        '''
        Xs = np.atleast_2d(Xs)

        # self.update_L()

        Kss = self.kern.K(Xs)
        Ksu = self.kern.K(Xs, self.U)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        tmp = np.linalg.solve(self.Luu.T, Ws)
        
        mu = Ws.T @ np.linalg.solve(self.Luu, self.mu)
        # var = Kss - Ksu @ (self.Kuu_inv - self.Kuu_inv @ self.Su @ self.Kuu_inv) @ Ksu.T
        var = Kss - Ws.T @ Ws + tmp.T @ self.Su @ tmp

        if self.gaussian_opts.normalize_y is True:
            mu = self.normalizer.inverse_mean(mu)
            var = self.normalizer.inverse_variance(var)
        
        if compute_dvar_dx:
            self.dvar_dx_data = self.dvar_dx(Xs, Ws, tmp)

        return mu, np.diag(var)
    
    # ------------------------------------------------------------------

    def predict_cholesky(self, Xs: np.ndarray) -> Tuple[np.ndarray]:
        '''
        Predict mean and variance at new test points using Cholesky factors.
        W = inv(Luu) @ Kuf
        Ws = inv(Luu) @ Kus
        D as in self.model()
        K = I + W @ inv(D) @ W.T = L @ L.T
        S = inv[Kuu + Kuf @ inv(D) @ Kfu]
          = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
          = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
          = inv(Luu).T @ inv(K) @ inv(Luu)
          = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
            = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        Parameters
        ----------
        Xs : np.ndarray, shape (N*, D)
            Test inputs.

        Returns
        -------
        mu : np.ndarray, shape (N*,)
            Predictive mean.
        var : np.ndarray, shape (N*,)
            Predictive variance.
        '''
        Xs = np.atleast_2d(Xs)

        Kss = self.kern.K(Xs)
        Ksu = self.kern.K(Xs, self.U)
        Ws = np.linalg.solve(self.Luu, Ksu.T)

        self.update_L()

        L_inv_Ws = np.linalg.solve(self.L, Ws)
        L_inv_W_d_inv_y = np.linalg.solve(self.L, self.W_d_inv_y)

        mu = L_inv_Ws.T @ L_inv_W_d_inv_y
        var = Kss - Ws.T @ Ws + L_inv_Ws.T @ L_inv_Ws
        
        if self.gaussian_opts.normalize_y is True:
            mu = self.normalizer.inverse_mean(mu)
            var = self.normalizer.inverse_variance(var)
            
        return mu, np.diag(var)

    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, Y_new: np.ndarray, Ksu: np.ndarray | None =None) -> None:
        '''
        Online update with a single new data point.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (1, D)
            New input.
        Y_new : np.ndarray, shape (1,) or (1,1)
            New output.
        '''
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_1d(Y_new).reshape(-1, 1)

        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.update_L()

        if Ksu is None:
            Ksu = self.kern.K(X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = self.kern.Kdiag(X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Φ_k = K(z, U) Kuu^{-1}, shape (M, 1)
        # Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)
        Phi_k = np.linalg.solve(self.Luu.T, Ws)            # (M, 1)

        # Residual: r_k = y - Phi_k^T mu
        r_k = Y_new - Phi_k.T @ self.mu            # scalar

        # G_k = Phi_k^T S_u Phi_k + sigma^2
        V = Phi_k.T @ self.Su
        G_k = V @ Phi_k + self.variance + Ktt_diag - Qtt_diag + 1 - self.lambda_  # scalar

        # L_k = S_u Phi_k / G_k
        A = self.Su @ Phi_k
        L_k = A / G_k                # (M, 1)

        # Update mean and covariance
        self.mu = self.mu + L_k * r_k
        self.Su = (1.0 / self.lambda_) * (self.Su - L_k @ V)

    def update_batch(self, X_new: np.ndarray, Y_new: np.ndarray, Ksu: np.ndarray | None = None) -> None:
        '''
        Online update with a batch of new data points.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (B, D)
            New inputs.
        Y_new : np.ndarray, shape (B, 1)
            New outputs.
        '''
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_2d(Y_new)

        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.update_L()

        if Ksu is None:
            Ksu = self.kern.K(X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = self.kern.Kdiag(X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Phi_b = K(X_new, U) Kuu^{-1}, shape (B, M)
        # Phi_b = K_XU @ self.Kuu_inv              # (B, M)
        Phi_b = np.linalg.solve(self.Luu.T, Ws).T

        # Residuals: R_b = Y_new - Phi_b @ mu
        R_b = Y_new - Phi_b @ self.mu            # (B, 1)

        # G_b = Phi_b S_u Phi_b^T + sigma^2 I
        V = Phi_b @ self.Su
        G_b = V @ Phi_b.T + (self.variance + 1 - self.lambda_) * np.eye(len(X_new)) + np.diag(Ktt_diag - Qtt_diag)  # (B,B)

        # L_b = S_u Phi_b^T G_b^{-1}
        # L_b = self.Su @ Phi_b.T @ np.linalg.inv(G_b)  # (M,B)
        A = self.Su @ Phi_b.T
        L_b = np.linalg.solve(G_b.T, A.T).T # V -> A.T, so can it be reused?

        # Update mean and covariance
        self.mu = self.mu + L_b @ R_b
        self.Su = (1.0 / self.lambda_) * (self.Su - L_b @ V)

    def update_L(self) -> None:
        if not self.L_updated:
            self.K = (self.W * self.diag_inv) @ self.W.T
            self.K = (self.K + self.K.T) / 2 + np.eye(self.M)
            self.L = np.linalg.cholesky(self.K)
            self.L_updated = True

    def add_new_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.N = self.X.shape[0]

        Kuf_new = self.kern.K(self.U, X_new)
        W_new = np.linalg.solve(self.Luu, Kuf_new)
        self.Kuf = np.hstack((self.Kuf, Kuf_new))
        self.W = np.hstack((self.W, W_new))

        Kff_diag_new = self.kern.Kdiag(X_new)
        Qff_diag_new = np.power(W_new.T, 2).sum(axis=-1)
        diag_new = Kff_diag_new - Qff_diag_new + self.variance
        diag_inv_new = 1 / diag_new
        self.diag = np.hstack((self.diag, diag_new))
        self.diag_inv = np.hstack((self.diag_inv, diag_inv_new))

        W_d_inv_y_new = (W_new * diag_inv_new) @ Y_new
        self.W_d_inv_y += W_d_inv_y_new.reshape(self.W_d_inv_y.shape)
        
        self.L_updated = False

    def remove_data(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        if remove_number is None and reserve_indices is None:
            raise ValueError('Either remove_number or reserve_indices must be provided.')

        W_old = self.W[:, :remove_number]
        diag_inv_old = self.diag_inv[:remove_number]
        Y_old = self.Y[:remove_number]

        self.X = self.X[remove_number:]
        self.Y = self.Y[remove_number:]
        self.N = self.X.shape[0]

        self.Kuf = self.Kuf[:, remove_number:]
        self.W = self.W[:, remove_number:]
        self.diag = self.diag[remove_number:]
        self.diag_inv = self.diag_inv[remove_number:]

        self.W_d_inv_y -= (W_old * diag_inv_old) @ Y_old

        # self.L_updated = False # Whether this is needed?

    def check_new_inducing_var(self, U_new: np.ndarray) -> np.ndarray:
        a = self.kern.K(self.U, U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability

        c = np.linalg.solve(self.Luu, a)   # (M, 1)

        return aa - c.T @ c, aa
    
    def add_new_inducing(self, U_new: np.ndarray) -> None:
        a = self.kern.K(self.U, U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Ku_new_f = self.kern.K(U_new, self.X)
        self.Kuf = np.vstack((self.Kuf, Ku_new_f))

        W_new = (Ku_new_f - self.Luu[-1:, :-1] @ self.W) / self.Luu[-1, -1]
        self.W = np.vstack((self.W, W_new))

        Qff_diag_new = np.power(self.W.T[:,-1:], 2).sum(axis=-1)
        self.diag -= Qff_diag_new
        self.diag_inv = 1.0 / self.diag

        self.W_d_inv_y = (self.W * self.diag_inv) @ self.Y
        # print(self.W_d_inv_y.flatten())

        self.L_updated = False
    
    def add_new_inducing_quick(self, U_new: np.ndarray) -> None:
        a = self.kern.K(self.U, U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

    def remove_inducing(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        pass

class SparseDTCGP:
    def __init__(self, kernel: GPy.kern, gaussian_opts: GaussianConfig) -> None:
        self.kern = kernel
        self.gaussian_opts = gaussian_opts

    def fit(self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        U: np.ndarray,
        variance: float = 1e-6,
        lambda_: float = 1.0
    ) -> None:
        
        print(f'[GPR]: Fitting Sparse DTC GP Model with normalized: {self.gaussian_opts.normalize_y}')

        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self.U = np.atleast_2d(U)
        self.lambda_ = lambda_
        self.variance = variance
        
        if self.gaussian_opts.normalize_y is True:
            self.normalizer = Normalizer(self.Y)
            self.Y = self.normalizer.normalize(self.Y)
        else:
            self.normalizer = None

        self.M = len(self.U)
        self.N = len(self.X)

        # Core kernel matrices
        self.Kuu = self.kern.K(self.U)
        self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry
        self.Kuf = self.kern.K(self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-8 * np.eye(self.M)
        self.Luu = np.linalg.cholesky(self.Kuu + jitter)

        # --- Factorized form ---
        # W = Luu^{-1} Kuf
        self.W = np.linalg.solve(self.Luu, self.Kuf)

        self.W_d_inv_y = (1.0 / self.variance) * self.W @ self.Y

        # K = I + W D^{-1} W^T
        # WD_half = self.W @ np.sqrt(self.D)
        self.K = (1.0 / self.variance) * self.W @ self.W.T
        self.K = (self.K + self.K.T) / 2 + np.eye(self.M)  # ensure symmetry
        
        # Cholesky of K
        self.L = np.linalg.cholesky(self.K)

        self.L_updated = True

        self.update_mu_su(from_scratch=True)

    # ------------------------------------------------------------------

    def update_mu_su(self, from_scratch: bool = False, Kun: np.ndarray | None = None, Knn: np.ndarray | None = None) -> None:
        if from_scratch:
            self.update_L()

            # self.Kuu = self.kern.K(self.U)
            # self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry

            # self.Kuu_inv = np.linalg.solve(
            #     self.Luu.T, np.linalg.solve(self.Luu, np.eye(self.M))
            # )

            # Posterior covariance Su
            # Sigma = inv[Kuu + Kuf @ inv(D) @ Kfu]
            #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
            #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
            #   = inv(Luu).T @ inv(K) @ inv(Luu)
            #   = Luu^{-T} L^{-T} L^{-1} Luu^{-1}
            # Su = Kuu @ Sigma @ Kuu = Luu L^{-T} L^{-1} Luu.T = tmp.T @ tmp
            # S = np.linalg.solve(self.L, np.linalg.solve(self.Luu, np.eye(self.M)))
            # self.Sigma = S.T @ S
            tmp = np.linalg.solve(self.L, self.Luu.T)
            # tmp = np.linalg.solve(self.L, np.linalg.solve(self.Luu, self.Kuu))
            self.Su = tmp.T @ tmp  # symmetric MxM

            # Posterior mean mu
            rhs = np.linalg.solve(self.Luu, (self.Kuf / self.variance) @ self.Y)
            self.mu = np.linalg.solve(self.Luu, self.Su.T).T @ rhs
        else:
            # TODO: if new inducing points are too many, then the update will have some error
            # TODO: if new inducing is close to old ones, numerical issues may happen
            Luu_inv_Kun = np.linalg.solve(self.Luu, Kun)
            Kuu_inv_kun = np.linalg.solve(self.Luu.T, Luu_inv_Kun)
            mu_new = Kuu_inv_kun.T @ self.mu
            self.mu = np.vstack((self.mu, mu_new))

            Sun = self.Su @ Kuu_inv_kun
            Snn = Knn - Luu_inv_Kun.T @ Luu_inv_Kun + Kuu_inv_kun.T @ self.Su @ Kuu_inv_kun
            left = np.vstack((self.Su, Sun.T))
            right = np.vstack((Sun, Snn))
            self.Su = np.hstack((left, right))

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray, method: str = 'cholesky', compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        if method == 'cholesky':
            return self.predict_cholesky(Xs)
        elif method == 'mu_su':
            return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def predict_noiseless(self, Xs: np.ndarray, full_cov: bool, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def save_model(self, id: str) -> None:
        print(f'[GPR]: Saving Sparse DTC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        save_path = self.gaussian_opts.save_path + id + '.npy'
        with open(save_path, 'wb') as file:
            np.save(file, self.N)
            np.save(file, self.M)
            np.save(file, self.variance)
            np.save(file, self.lambda_)
            np.save(file, self.X)
            np.save(file, self.Y)
            np.save(file, self.U)

            if self.gaussian_opts.normalize_y is True:
                np.save(file, self.normalizer.mean)
                np.save(file, self.normalizer.std)
            
            np.save(file, self.Luu)
            np.save(file, self.mu)
            np.save(file, self.Su)
            np.save(file, self.Kuf)
            np.save(file, self.W)
            np.save(file, self.W_d_inv_y)
            np.save(file, self.L)

    # ------------------------------------------------------------------

    def load_model(self, id: str) -> None:
        print(f'[GPR]: Loading Sparse DTC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        load_path = self.gaussian_opts.load_path + id + '.npy'
        with open(load_path, 'rb') as file:
            self.N = np.load(file)
            self.M = np.load(file)
            self.variance = np.load(file)
            self.lambda_ = np.load(file)
            self.X = np.load(file)
            self.Y = np.load(file)
            self.U = np.load(file)

            if self.gaussian_opts.normalize_y is True:
                mean = np.load(file)
                std = np.load(file)
                self.normalizer = Normalizer(self.Y)
                self.normalizer.mean = mean
                self.normalizer.std = std
            else:
                self.normalizer = None
                
            self.Luu = np.load(file)
            self.mu = np.load(file)
            self.Su = np.load(file)
            self.Kuf = np.load(file)
            self.W = np.load(file)
            self.W_d_inv_y = np.load(file)
            self.L = np.load(file)
        self.L_updated = True
            
    # ------------------------------------------------------------------
    
    def dk_dx(self, Xs: np.ndarray) -> np.ndarray:
        mean_jac = np.zeros((self.U.shape[0], self.U.shape[1]))
        for i in range(self.U.shape[1]):
            mean_jac[:, i] = self.kern.dK_dX2(self.U, Xs, i).squeeze()
        return mean_jac
    
    # ------------------------------------------------------------------
    
    def dmu_dx(self, Xs: np.ndarray) -> np.ndarray:
        rhs = np.linalg.solve(self.Luu, self.mu)
        dk_dx = self.dk_dx(Xs)
        lhs = np.linalg.solve(self.Luu, dk_dx)
        result = lhs.T @ rhs
        return result
    
    # ------------------------------------------------------------------
    
    def dvar_dx(self, Xs: np.ndarray, Ws: np.ndarray, tmp: np.ndarray) -> np.ndarray:
        dk_dx = self.dk_dx(Xs)
        Luu_inv_dk_dx = np.linalg.solve(self.Luu, dk_dx)
        Kuu_inv_dk_dx = np.linalg.solve(self.Luu.T, Luu_inv_dk_dx)
        result = -2 * Luu_inv_dk_dx.T @ Ws + 2 * Kuu_inv_dk_dx.T @ self.Su @ tmp
        return result
        
    # ------------------------------------------------------------------

    def predict_mu_su(self, Xs: np.ndarray, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        '''
        Predict mean and variance at new test points.

        Parameters
        ----------
        Xs : np.ndarray, shape (N*, D)
            Test inputs.

        Returns
        -------
        mu : np.ndarray, shape (N*,)
            Predictive mean.
        var : np.ndarray, shape (N*,)
            Predictive variance.
        '''
        Xs = np.atleast_2d(Xs)

        self.update_L()

        Kss = self.kern.K(Xs)
        Ksu = self.kern.K(Xs, self.U)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        # tmp = np.linalg.solve(self.L, np.linalg.solve(self.Luu, Ksu.T))
        tmp2 = np.linalg.solve(self.Luu.T, Ws)
        
        mu = Ws.T @ np.linalg.solve(self.Luu, self.mu)
        # var = Kss - Ksu @ (self.Kuu_inv - self.Kuu_inv @ self.Su @ self.Kuu_inv) @ Ksu.T
        # var = Kss - Ws.T @ Ws + tmp.T @ tmp
        var = Kss - Ws.T @ Ws + tmp2.T @ self.Su @ tmp2
        # var = Kss - Ws.T @ Ws + Ksu @ self.Sigma @ Ksu.T
        
        if self.gaussian_opts.normalize_y is True:
            mu = self.normalizer.inverse_mean(mu)
            var = self.normalizer.inverse_variance(var)
        
        if compute_dvar_dx:
            self.dvar_dx_data = self.dvar_dx(Xs, Ws, tmp2)

        return mu, np.diag(var)

    # ------------------------------------------------------------------

    def predict_cholesky(self, Xs: np.ndarray) -> Tuple[np.ndarray]:
        '''
        Predict mean and variance at new test points using Cholesky factors.
        W = inv(Luu) @ Kuf
        Ws = inv(Luu) @ Kus
        D as in self.model()
        K = I + W @ inv(D) @ W.T = L @ L.T
        S = inv[Kuu + Kuf @ inv(D) @ Kfu]
          = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
          = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
          = inv(Luu).T @ inv(K) @ inv(Luu)
          = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
            = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        Parameters
        ----------
        Xs : np.ndarray, shape (N*, D)
            Test inputs.

        Returns
        -------
        mu : np.ndarray, shape (N*,)
            Predictive mean.
        var : np.ndarray, shape (N*,)
            Predictive variance.
        '''
        Xs = np.atleast_2d(Xs)

        Kss = self.kern.K(Xs)
        Ksu = self.kern.K(Xs, self.U)
        Ws = np.linalg.solve(self.Luu, Ksu.T)
        # W_d_inv_y = (1.0 / self.variance) * self.W @ self.Y

        self.update_L()

        L_inv_Ws = np.linalg.solve(self.L, Ws)
        L_inv_W_d_inv_y = np.linalg.solve(self.L, self.W_d_inv_y)

        mu = L_inv_Ws.T @ L_inv_W_d_inv_y
        var = Kss - Ws.T @ Ws + L_inv_Ws.T @ L_inv_Ws
        
        if self.gaussian_opts.normalize_y is True:
            mu = self.normalizer.inverse_mean(mu)
            var = self.normalizer.inverse_variance(var)
            
        return mu, np.diag(var)

    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        '''
        Online update with a single new data point.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (1, D)
            New input.
        Y_new : np.ndarray, shape (1,) or (1,1)
            New output.
        '''
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_1d(Y_new).reshape(-1, 1)

        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.update_L()

        # Φ_k = K(z, U) Kuu^{-1}, shape (M, 1)
        K_zU = self.kern.K(X_new, self.U)        # shape (1, M)
        # Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)
        Phi_k = np.linalg.solve(self.Luu.T, np.linalg.solve(self.Luu, K_zU.T))

        # Residual: r_k = y - Phi_k^T mu
        r_k = Y_new - Phi_k.T @ self.mu            # scalar

        # G_k = Phi_k^T S_u Phi_k + sigma^2
        G_k = Phi_k.T @ self.Su @ Phi_k + self.variance + 1 - self.lambda_  # scalar

        # L_k = S_u Phi_k / G_k
        L_k = self.Su @ Phi_k / G_k                # (M, 1)

        # Update mean and covariance
        self.mu = self.mu + L_k * r_k
        self.Su = (1.0 / self.lambda_) * (self.Su - L_k @ (Phi_k.T @ self.Su))

    def update_batch(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        '''
        Online update with a batch of new data points.
        Not good for the new data that are far from the inducing points as the inducing points are not updated.

        Parameters
        ----------
        X_new : np.ndarray, shape (B, D)
            New inputs.
        Y_new : np.ndarray, shape (B, 1)
            New outputs.
        '''
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_2d(Y_new)

        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.update_L()

        # Phi_b = K(X_new, U) Kuu^{-1}, shape (B, M)
        K_XU = self.kern.K(X_new, self.U)      # (B, M)
        # Phi_b = K_XU @ self.Kuu_inv              # (B, M)
        Phi_b = np.linalg.solve(self.Luu.T, np.linalg.solve(self.Luu, K_XU.T)).T

        # Residuals: R_b = Y_new - Phi_b @ mu
        R_b = Y_new - Phi_b @ self.mu            # (B, 1)

        # G_b = Phi_b S_u Phi_b^T + sigma^2 I
        V = Phi_b @ self.Su
        G_b = V @ Phi_b.T + (self.variance + 1 - self.lambda_) * np.eye(len(X_new))  # (B,B)

        # L_b = S_u Phi_b^T G_b^{-1}
        A = self.Su @ Phi_b.T
        L_b = np.linalg.solve(G_b.T, A.T).T  # (M,B)

        # Update mean and covariance
        self.mu = self.mu + L_b @ R_b
        self.Su = (1.0 / self.lambda_) * (self.Su - L_b @ V)

    def update_L(self) -> None:
        if not self.L_updated:
            self.K = (1 / self.variance) * self.W @ self.W.T
            self.K = (self.K + self.K.T) / 2 + np.eye(self.M)
            self.L = np.linalg.cholesky(self.K)
            self.L_updated = True

    def add_new_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)
            
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.N = self.X.shape[0]

        Kuf_new = self.kern.K(self.U, X_new)
        W_new = np.linalg.solve(self.Luu, Kuf_new)
        self.Kuf = np.hstack((self.Kuf, Kuf_new))
        self.W = np.hstack((self.W, W_new))

        W_d_inv_y_new = 1.0 / self.variance * W_new @ Y_new
        self.W_d_inv_y += W_d_inv_y_new.reshape(self.W_d_inv_y.shape)
        
        self.L_updated = False

    def remove_data(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        if remove_number is None and reserve_indices is None:
            raise ValueError('Either remove_number or reserve_indices must be provided.')

        W_old = self.W[:, :remove_number]
        Y_old = self.Y[:remove_number]

        self.X = self.X[remove_number:]
        self.Y = self.Y[remove_number:]
        self.N = self.X.shape[0]

        self.Kuf = self.Kuf[:, remove_number:]
        self.W = self.W[:, remove_number:]

        self.W_d_inv_y -= 1.0 / self.variance * W_old @ Y_old

        # self.L_updated = False # Whether this is needed?

    def check_new_inducing_var(self, U_new: np.ndarray) -> np.ndarray:
        a = self.kern.K(self.U, U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability

        c = np.linalg.solve(self.Luu, a)   # (M, 1)

        return aa - c.T @ c, aa
    
    def add_new_inducing(self, U_new: np.ndarray) -> None:
        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        a = self.kern.K(self.U[:-1], U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability
        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Kuf_new = self.kern.K(U_new, self.X)
        self.Kuf = np.vstack((self.Kuf, Kuf_new))

        W_new = (Kuf_new - self.Luu[-1:, :-1] @ self.W) / self.Luu[-1, -1]
        self.W = np.vstack((self.W, W_new))

        self.W_d_inv_y = 1.0 / self.variance * self.W @ self.Y

        self.L_updated = False

    def add_new_inducing_quick(self, U_new: np.ndarray) -> None:
        a = self.kern.K(self.U, U_new)
        aa = self.kern.K(U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)
        
    def remove_inducing(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        pass

class GaussianModel:
    def __init__(self, gaussian_opts: GaussianConfig) -> None:
        '''
            The kernel is a rbf kernel with ard lengthscale, 
            so the params are given as [likelihood variance, rbf variance, lengthscale]
        '''
        
        self.gps = []
        self.LINV, self.YAWV = 0, 1 # the index of the params and gps
        self.sparse_method = gaussian_opts.sparse_method
        self.opts = gaussian_opts

        # ----- 1. Read the data and load it to numpy -----
        train_data, inducing_data = self.read_data(gaussian_opts)
        self.feature_size = gaussian_opts.input_window * 4 # for now the feature is [v, w, v_c, w_c] of history

        # ----- 2. Build the GPR for the all states -----
        inducing_data_number = gaussian_opts.inducing_data_number
        params = gaussian_opts.params

        for i in range(len(params)):
            ard_length_scale = np.array(params[i][2:])
            variance = params[i][1]
            kernel = GPy.kern.RBF(input_dim=self.feature_size, variance=variance, lengthscale=ard_length_scale, ARD=True)

            if self.sparse_method == 0 or self.sparse_method == 2:
                gp = SparseDTCGP(kernel, gaussian_opts)
            else:
                gp = SparseFITCGP(kernel, gaussian_opts)

            if gaussian_opts.read_model is True:
                gp.load_model('_' + str(i))
            else:
                gp.fit(
                    train_data[:, :self.feature_size], 
                    train_data[:, self.feature_size + i].reshape(-1, 1),
                    inducing_data[inducing_data_number * i:inducing_data_number * (i+1), :],
                    variance=params[i][0],
                )

            self.gps.append(gp)

    def read_data(self, gaussian_opts: GaussianConfig) -> Tuple[np.ndarray]:
        train_data, inducing_data = [], []
        if os.path.exists(gaussian_opts.path):
            with open(gaussian_opts.path, 'r') as f:
                lines = f.readlines()

            i = 0
            for line in lines:
                if i < gaussian_opts.total_data_number:
                    result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
                    data = [float(d) for d in result]
                    train_data.append(data)
                elif i < gaussian_opts.total_data_number + 2 * gaussian_opts.inducing_data_number:
                    result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
                    data = [float(d) for d in result]
                    inducing_data.append(data)
                i += 1
        else:
            raise NotImplementedError(f'gaussain data {gaussian_opts.path} not exist!')

        return np.array(train_data), np.array(inducing_data)
    
    def dmu_dx(self, Xnew: np.ndarray) -> np.ndarray:
        dmu_dx = []
        for i in range(len(self.gps)):
            gp, normalizer = self.gps[i], self.gps[i].normalizer
            result = gp.dmu_dx(Xnew)
            if normalizer is not None: 
                result = normalizer.inverse_mean(result) - normalizer.inverse_mean(0.0)
            dmu_dx.append(result)

        return dmu_dx

def test_gaussian_model(model):
    gp_nums = len(model.gps)
    for i in range(gp_nums):
        gp = model.gps[i]
        gpy_mu, gpy_var = gp.predict_noiseless(gp.X, full_cov=True)
        if model.opts.normalize_y is True:
            mean, std = gp.normalizer.mean, gp.normalizer.std
            y = gp.normalizer.inverse_mean(gp.Y)
        else:
            mean, std = 0.0, 0.0
            y = gp.Y
        print(f'[GaussianModel]: error is {np.mean(np.abs(gpy_mu - y)):+.6f}, mean and std are {mean:+.6f} {std:+.6f}')
