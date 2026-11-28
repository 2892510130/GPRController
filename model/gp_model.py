import re
import os
import GPy
import numpy as np
from scipy.linalg import solve_triangular
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

_MODEL_FORMAT_VERSION = 2
_NPZ_MAGIC = b"PK\x03\x04"


def _is_rbf_kernel(kern) -> bool:
    return isinstance(kern, GPy.kern.RBF)


def _rbf_parameters(kern):
    variance = float(np.asarray(kern.variance).reshape(-1)[0])
    lengthscales = np.asarray(kern.lengthscale, dtype=float).reshape(-1)
    if lengthscales.size == 1:
        lengthscales = np.full(kern.input_dim, float(lengthscales[0]))
    return variance, lengthscales


def _kern_K(kern, X1, X2=None) -> np.ndarray:
    """RBF kernel matrix (numpy implementation; falls back to GPy otherwise)."""
    if X2 is None:
        X2 = X1
    if not _is_rbf_kernel(kern):
        return kern.K(X1, X2)
    variance, lengthscales = _rbf_parameters(kern)
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    K = np.empty((X1.shape[0], X2.shape[0]))
    block = 256
    for start in range(0, X1.shape[0], block):
        diff = (X1[start:start + block, None, :] - X2[None, :, :]) / lengthscales
        K[start:start + block] = variance * np.exp(-0.5 * np.sum(diff * diff, axis=-1))
    return K


def _kern_Kdiag(kern, X) -> np.ndarray:
    if not _is_rbf_kernel(kern):
        return kern.Kdiag(X)
    variance, _ = _rbf_parameters(kern)
    return np.full(np.atleast_2d(X).shape[0], variance)


def _rbf_gradient(kern, X1, X2) -> np.ndarray:
    """dK(X1, X2)/dX2 for an RBF kernel; shape (len(X1), feature_dim)."""
    variance, lengthscales = _rbf_parameters(kern)
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1 - X2
    coefficient = variance * np.exp(
        -0.5 * np.sum((diff / lengthscales) ** 2, axis=1, keepdims=True)
    )
    return coefficient * diff / (lengthscales ** 2)


def _is_npz_file(path: str) -> bool:
    """The model files keep the historical `.npy` suffix; detect the container."""
    with open(path, "rb") as file:
        return file.read(len(_NPZ_MAGIC)) == _NPZ_MAGIC


def _save_sparse_gp(model, path: str, kind: str) -> None:
    """Persist a sparse GP (mu_su state always, Cholesky state when valid)."""
    if model.cholesky_valid:
        # Make sure L matches the current W/diag before it is written out.
        model.update_L()
    arrays = {
        "format_version": np.array(_MODEL_FORMAT_VERSION),
        "kind": np.array(kind),
        "N": np.array(model.N),
        "M": np.array(model.M),
        "variance": np.array(model.variance),
        "lambda_": np.array(model.lambda_),
        "X": model.X,
        "Y": model.Y,
        "U": model.U,
        "feature_dim": np.array(model.U.shape[1]),
        "kernel_params": np.asarray(model.kern.param_array, dtype=float),
        "normalize_y": np.array(bool(model.gaussian_opts.normalize_y)),
        "cholesky_valid": np.array(bool(model.cholesky_valid)),
        "Luu": model.Luu,
        "mu": model.mu,
        "Su": model.Su,
    }
    if model.normalizer is not None:
        arrays["normalizer_mean"] = np.array(model.normalizer.mean)
        arrays["normalizer_std"] = np.array(model.normalizer.std)

    if model.cholesky_valid:
        arrays["Kuf"] = model.Kuf
        arrays["W"] = model.W
        arrays["W_d_inv_y"] = model.W_d_inv_y
        arrays["L"] = model.L
        if kind == "FITC":
            arrays["diag"] = model.diag
            arrays["diag_inv"] = model.diag_inv

    with open(path, "wb") as file:
        np.savez(file, **arrays)


def _load_legacy_sparse_gp(model, path: str, kind: str) -> None:
    """Read the pre-versioning format (a flat sequence of np.save records)."""
    with open(path, "rb") as file:
        N = int(np.load(file))
        M = int(np.load(file))
        variance = float(np.load(file))
        lambda_ = float(np.load(file))
        X = np.load(file)
        Y = np.load(file)
        U = np.load(file)

        if model.gaussian_opts.normalize_y is True:
            mean = np.load(file)
            std = np.load(file)
            normalizer = Normalizer(Y)
            normalizer.mean, normalizer.std = mean, std
        else:
            normalizer = None

        Luu = np.load(file)
        mu = np.load(file)
        Su = np.load(file)
        Kuf = np.load(file)
        W = np.load(file)
        if kind == "FITC":
            diag = np.load(file)
            diag_inv = np.load(file)
            W_d_inv_y = np.load(file)
            L = np.load(file)
        else:
            W_d_inv_y = np.load(file)
            L = np.load(file)

    model.N, model.M = N, M
    model.variance, model.lambda_ = variance, lambda_
    model.X, model.Y, model.U = X, Y, U
    model.normalizer = normalizer
    model.Luu, model.mu, model.Su = Luu, mu, Su
    model.Kuf, model.W = Kuf, W
    model.W_d_inv_y, model.L = W_d_inv_y, L
    if kind == "FITC":
        model.diag, model.diag_inv = diag, diag_inv
    model.L_updated = True
    model.cholesky_valid = True


def _load_sparse_gp(model, path: str, kind: str) -> None:
    if not _is_npz_file(path):
        _load_legacy_sparse_gp(model, path, kind)
        return

    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"].item())
        if version != _MODEL_FORMAT_VERSION:
            raise ValueError(
                f"[GPR]: unsupported model file version {version} in '{path}' "
                f"(expected {_MODEL_FORMAT_VERSION})"
            )

        saved_kind = str(data["kind"].item())
        if saved_kind != kind:
            raise ValueError(
                f"[GPR]: model file '{path}' was trained with a {saved_kind} GP, but "
                f"the configured sparse_method uses {kind}; retrain or fix sparse_method."
            )

        saved_normalize = bool(data["normalize_y"].item())
        configured_normalize = bool(model.gaussian_opts.normalize_y)
        if saved_normalize != configured_normalize:
            raise ValueError(
                f"[GPR]: model file '{path}' was trained with normalize_y={saved_normalize} "
                f"but the config says {configured_normalize}; retrain or fix the config."
            )

        feature_dim = int(data["feature_dim"].item())
        if feature_dim != model.kern.input_dim:
            raise ValueError(
                f"[GPR]: model file '{path}' has {feature_dim} input features but the "
                f"config builds a kernel for {model.kern.input_dim}; retrain or fix input_window."
            )

        kernel_params = np.asarray(data["kernel_params"], dtype=float)
        config_kernel_params = np.asarray(model.kern.param_array, dtype=float)
        if not np.allclose(kernel_params, config_kernel_params, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"[GPR]: model file '{path}' was trained with different kernel "
                "hyperparameters than config.gaussian.params; update the params or "
                "retrain with read_model: false."
            )

        N = int(data["N"].item())
        M = int(data["M"].item())
        variance = float(data["variance"].item())
        lambda_ = float(data["lambda_"].item())
        X = np.array(data["X"])
        Y = np.array(data["Y"])
        U = np.array(data["U"])
        Luu = np.array(data["Luu"])
        mu = np.array(data["mu"])
        Su = np.array(data["Su"])
        cholesky_valid = bool(data["cholesky_valid"].item())

        normalizer = None
        if saved_normalize:
            normalizer = Normalizer(Y)
            normalizer.mean = float(data["normalizer_mean"].item())
            normalizer.std = float(data["normalizer_std"].item())

        if cholesky_valid:
            Kuf = np.array(data["Kuf"])
            W = np.array(data["W"])
            W_d_inv_y = np.array(data["W_d_inv_y"])
            L = np.array(data["L"])
            if kind == "FITC":
                diag = np.array(data["diag"])
                diag_inv = np.array(data["diag_inv"])

    if X.ndim != 2 or X.shape != (N, feature_dim):
        raise ValueError(f"[GPR]: invalid X shape {X.shape} in '{path}' (expected ({N}, {feature_dim}))")
    if Y.shape != (N, 1):
        raise ValueError(f"[GPR]: invalid Y shape {Y.shape} in '{path}' (expected ({N}, 1))")
    if U.shape != (M, feature_dim):
        raise ValueError(f"[GPR]: invalid U shape {U.shape} in '{path}' (expected ({M}, {feature_dim}))")
    if Luu.shape != (M, M) or mu.shape != (M, 1) or Su.shape != (M, M):
        raise ValueError(
            f"[GPR]: inconsistent mu_su state in '{path}': Luu {Luu.shape}, "
            f"mu {mu.shape}, Su {Su.shape} (expected M={M})"
        )
    if cholesky_valid:
        if Kuf.shape != (M, N) or W.shape != (M, N):
            raise ValueError(f"[GPR]: inconsistent Cholesky state in '{path}' (Kuf/W shapes)")
        if W_d_inv_y.shape != (M, 1) or L.shape != (M, M):
            raise ValueError(f"[GPR]: inconsistent Cholesky state in '{path}' (W_d_inv_y/L shapes)")
        if kind == "FITC" and (diag.shape != (N,) or diag_inv.shape != (N,)):
            raise ValueError(f"[GPR]: inconsistent FITC diag state in '{path}'")

    model.N, model.M = N, M
    model.variance, model.lambda_ = variance, lambda_
    model.X, model.Y, model.U = X, Y, U
    model.normalizer = normalizer
    model.Luu, model.mu, model.Su = Luu, mu, Su
    model.cholesky_valid = cholesky_valid
    if cholesky_valid:
        model.Kuf, model.W = Kuf, W
        model.W_d_inv_y, model.L = W_d_inv_y, L
        if kind == "FITC":
            model.diag, model.diag_inv = diag, diag_inv
        model.L_updated = True
    else:
        model.L_updated = False


class SparseFITCGP:
    def __init__(self, kernel: GPy.kern, gaussian_opts: GaussianConfig) -> None:
        self.kern = kernel
        self.gaussian_opts = gaussian_opts
        self.cholesky_valid = False

    def _solve_luu(self, rhs: np.ndarray) -> np.ndarray:
        return solve_triangular(self.Luu, rhs, lower=True)

    def _solve_luu_t(self, rhs: np.ndarray) -> np.ndarray:
        return solve_triangular(self.Luu.T, rhs, lower=False)

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
        self.Kuu = _kern_K(self.kern, self.U)
        self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry
        self.Kuf = _kern_K(self.kern, self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-6 * np.eye(self.M)
        self.Luu = np.linalg.cholesky(self.Kuu + jitter)

        # --- Factorized form ---
        # W = Luu^{-1} Kuf
        self.W = self._solve_luu(self.Kuf)

        self.Kff_diag = _kern_Kdiag(self.kern, self.X)
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

        self.cholesky_valid = True

    # ------------------------------------------------------------------

    def update_mu_su(
        self,
        from_scratch: bool = False,
        Kun: np.ndarray | None = None,
        Knn: np.ndarray | None = None,
    ) -> None:
        if from_scratch:
            self.update_L()

            # self.Kuu = _kern_K(self.kern, self.U)
            # self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry

            # self.Kuu_inv = np.linalg.solve(
            #     self.Luu.T, self._solve_luu(np.eye(self.M))
            # )

            # Posterior covariance Su
            # Sigma = inv[Kuu + Kuf @ inv(D) @ Kfu]
            #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
            #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
            #   = inv(Luu).T @ inv(K) @ inv(Luu)
            #   = Luu^{-T} L^{-T} L^{-1} Luu^{-1}
            # Su = Kuu @ Sigma @ Kuu = Luu L^{-T} L^{-1} Luu.T = tmp.T @ tmp
            tmp = np.linalg.solve(self.L, self.Luu.T)
            # tmp = np.linalg.solve(self.L, self._solve_luu(self.Kuu))
            self.Su = tmp.T @ tmp  # symmetric MxM

            # Posterior mean mu
            # mu = Su @ Kuu^{-1} Kuf D^{-1} Y
            # rhs = (self.Kuf * self.diag_inv) @ self.Y
            # self.mu = self.Su @ (self.Kuu_inv @ rhs)
            rhs = self._solve_luu((self.Kuf * self.diag_inv) @ self.Y)
            self.mu = self._solve_luu(self.Su.T).T @ rhs
        else:
            # TODO: if new inducing points are too many, then the update will have some error
            # TODO: if new inducing is close to old ones, numerical issues may happen
            Luu_inv_Kun = self._solve_luu(Kun)
            Kuu_inv_kun = self._solve_luu_t(Luu_inv_Kun)
            mu_new = Kuu_inv_kun.T @ self.mu
            self.mu = np.vstack((self.mu, mu_new))

            Sun = self.Su @ Kuu_inv_kun
            Snn = Knn - Luu_inv_Kun.T @ Luu_inv_Kun + Kuu_inv_kun.T @ self.Su @ Kuu_inv_kun
            left = np.vstack((self.Su, Sun.T))
            right = np.vstack((Sun, Snn))
            self.Su = np.hstack((left, right))

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray, method: str = 'mu_su', compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        if method == 'cholesky':
            return self.predict_cholesky(Xs)
        if method == 'mu_su':
            return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
        raise ValueError(
            f"Unknown prediction method '{method}'; expected 'mu_su' or 'cholesky'."
        )
    
    # ------------------------------------------------------------------

    def predict_noiseless(self, Xs: np.ndarray, full_cov: bool, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def save_model(self, id: str) -> None:
        print(f'[GPR]: Saving Sparse FITC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        save_path = self.gaussian_opts.save_path + id + '.npy'
        _save_sparse_gp(self, save_path, "FITC")

    # ------------------------------------------------------------------

    def load_model(self, id: str) -> None:
        print(f'[GPR]: Loading Sparse FITC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        load_path = self.gaussian_opts.load_path + id + '.npy'
        _load_sparse_gp(self, load_path, "FITC")

    # ------------------------------------------------------------------

    def _require_cholesky(self, operation: str) -> None:
        if not self.cholesky_valid:
            raise RuntimeError(
                f"{self.__class__.__name__}.{operation}: the Cholesky-based state is "
                "stale because add_new_inducing_quick() only maintains the mu_su "
                "(mean/covariance) prediction state. Use predict_noiseless() or "
                "predict(..., method='mu_su'), or call sync_cholesky() first."
            )

    # ------------------------------------------------------------------

    def sync_cholesky(self) -> None:
        """Rebuild the Cholesky-based prediction/update state from U, X and Y.

        Note: ``mu``/``Su`` keep the incrementally updated posterior maintained
        by ``add_new_inducing_quick`` (an approximation), so mu_su predictions
        are not expected to match Cholesky predictions exactly.
        """
        self.Kuf = _kern_K(self.kern, self.U, self.X)
        self.W = self._solve_luu(self.Kuf)
        self.Kff_diag = _kern_Kdiag(self.kern, self.X)
        self.Qff_diag = np.power(self.W.T, 2).sum(axis=-1)
        self.diag = self.Kff_diag - self.Qff_diag + self.variance
        if np.any(self.diag <= 0.0):
            raise FloatingPointError(
                f"{self.__class__.__name__}.sync_cholesky: non-positive FITC diagonal; "
                "increase the GP noise variance before syncing."
            )
        self.diag_inv = 1.0 / self.diag
        self.W_d_inv_y = (self.W * self.diag_inv) @ self.Y
        self.L_updated = False
        self.cholesky_valid = True
        try:
            self.update_L()
        except Exception:
            self.cholesky_valid = False
            raise

    # ------------------------------------------------------------------
    
    def dk_dx(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.atleast_2d(Xs)
        if Xs.shape[0] != 1:
            raise ValueError(
                "dk_dx/dmu_dx only support a single test input, got "
                f"{Xs.shape[0]} rows."
            )
        if _is_rbf_kernel(self.kern):
            return _rbf_gradient(self.kern, self.U, Xs)
        mean_jac = np.zeros((self.U.shape[0], self.U.shape[1]))
        for i in range(self.U.shape[1]):
            mean_jac[:, i] = self.kern.dK_dX2(self.U, Xs, i).squeeze()
        return mean_jac
    
    # ------------------------------------------------------------------
    
    def dmu_dx(self, Xs: np.ndarray) -> np.ndarray:
        rhs = self._solve_luu(self.mu)
        dk_dx = self.dk_dx(Xs)
        lhs = self._solve_luu(dk_dx)
        result = lhs.T @ rhs
        return result
    
    # ------------------------------------------------------------------
    
    def dvar_dx(self, Xs: np.ndarray, Ws: np.ndarray, tmp: np.ndarray) -> np.ndarray:
        dk_dx = self.dk_dx(Xs)
        Luu_inv_dk_dx = self._solve_luu(dk_dx)
        Kuu_inv_dk_dx = self._solve_luu_t(Luu_inv_dk_dx)
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

        Kss = _kern_K(self.kern, Xs)
        Ksu = _kern_K(self.kern, Xs, self.U)
        Ws = self._solve_luu(Ksu.T)
        tmp = self._solve_luu_t(Ws)
        
        mu = Ws.T @ self._solve_luu(self.mu)
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

        self._require_cholesky("predict_cholesky")

        Kss = _kern_K(self.kern, Xs)
        Ksu = _kern_K(self.kern, Xs, self.U)
        Ws = self._solve_luu(Ksu.T)

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
            Ksu = _kern_K(self.kern, X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = _kern_Kdiag(self.kern, X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = self._solve_luu(Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Φ_k = K(z, U) Kuu^{-1}, shape (M, 1)
        # Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)
        Phi_k = self._solve_luu_t(Ws)            # (M, 1)

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
            Ksu = _kern_K(self.kern, X_new, self.U)
        # When the new data is far from the original data, Ktt_diag - Qtt_diag will close to 1, make the update not useful
        Ktt_diag = _kern_Kdiag(self.kern, X_new)
        # Qtt_diag = np.diag(Ksu @ self.Kuu_inv @ Ksu.T)
        Ws = self._solve_luu(Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)

        # Phi_b = K(X_new, U) Kuu^{-1}, shape (B, M)
        # Phi_b = K_XU @ self.Kuu_inv              # (B, M)
        Phi_b = self._solve_luu_t(Ws).T

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

    def update_mu_su_bijl(self, X_new: np.ndarray, Y_new: np.ndarray,
                          Ksu: np.ndarray | None = None) -> None:
        """Sequential update of Bijl and van Wingerden (2015).

        Non-variational: the mean/covariance of the inducing variables are
        updated through the matrix inversion lemma on the FITC posterior, with
        no ELBO and no forgetting factor.
        """
        X_new = np.atleast_2d(X_new)
        Y_new = np.atleast_1d(Y_new).reshape(-1, 1)

        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.update_L()

        if Ksu is None:
            Ksu = _kern_K(self.kern, X_new, self.U)

        Ktt_diag = _kern_Kdiag(self.kern, X_new)
        Ws = self._solve_luu(Ksu.T)
        Qtt_diag = np.power(Ws.T, 2).sum(axis=-1)
        diag = Ktt_diag - Qtt_diag + self.variance

        Luu_T_inv_Ws = self._solve_luu_t(Ws)
        B_tmp = (Luu_T_inv_Ws * (1.0 / diag)) @ Luu_T_inv_Ws.T

        Su_B = self.Su @ B_tmp
        denom = 1.0 + np.trace(Su_B)
        self.Su = self.Su - (Su_B @ self.Su) / denom

        Su_B = self.Su @ B_tmp
        denom = 1.0 + np.trace(Su_B)
        self.mu = (np.eye(self.M) - Su_B / denom) @ self.mu \
            + (self.Su @ Luu_T_inv_Ws * (1.0 / diag)) @ Y_new

    def update_L(self) -> None:
        if not self.L_updated:
            if not self.cholesky_valid:
                # Quick updates keep only the mu_su state; there is nothing to
                # rebuild until sync_cholesky() is called.
                return
            self.K = (self.W * self.diag_inv) @ self.W.T
            self.K = (self.K + self.K.T) / 2 + np.eye(self.M)
            self.L = np.linalg.cholesky(self.K)
            self.L_updated = True

    def add_new_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        self._require_cholesky("add_new_data")
        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)

        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.N = self.X.shape[0]

        Kuf_new = _kern_K(self.kern, self.U, X_new)
        W_new = self._solve_luu(Kuf_new)
        self.Kuf = np.hstack((self.Kuf, Kuf_new))
        self.W = np.hstack((self.W, W_new))

        Kff_diag_new = _kern_Kdiag(self.kern, X_new)
        Qff_diag_new = np.power(W_new.T, 2).sum(axis=-1)
        diag_new = Kff_diag_new - Qff_diag_new + self.variance
        diag_inv_new = 1 / diag_new
        self.diag = np.hstack((self.diag, diag_new))
        self.diag_inv = np.hstack((self.diag_inv, diag_inv_new))

        W_d_inv_y_new = (W_new * diag_inv_new) @ Y_new
        self.W_d_inv_y += W_d_inv_y_new.reshape(self.W_d_inv_y.shape)
        
        self.L_updated = False
        # Keep the mu_su posterior consistent with the enlarged training set.
        self.update_mu_su(from_scratch=True)

    def remove_data(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        self._require_cholesky("remove_data")
        if remove_number is None and reserve_indices is None:
            raise ValueError('Either remove_number or reserve_indices must be provided.')

        keep = np.zeros(self.N, dtype=bool)
        if reserve_indices is not None:
            keep[reserve_indices] = True
        else:
            keep[remove_number:] = True

        self.X = self.X[keep]
        self.Y = self.Y[keep]
        self.N = self.X.shape[0]
        self.sync_cholesky()
        self.update_mu_su(from_scratch=True)

    def check_new_inducing_var(self, U_new: np.ndarray) -> np.ndarray:
        a = _kern_K(self.kern, self.U, U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability

        c = self._solve_luu(a)   # (M, 1)

        return aa - c.T @ c, aa
    
    def add_new_inducing(self, U_new: np.ndarray) -> None:
        self._require_cholesky("add_new_inducing")
        a = _kern_K(self.kern, self.U, U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Ku_new_f = _kern_K(self.kern, U_new, self.X)
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
        a = _kern_K(self.kern, self.U, U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        # Quick updates intentionally skip Kuf/W/diag/L: only the mu_su
        # prediction state is kept up to date. Cholesky users must call
        # sync_cholesky() first (or stick to predict_noiseless).
        self.cholesky_valid = False
        self.L_updated = False

    def contributions(self) -> np.ndarray:
        """Marginal contribution of each inducing point.

        Proxy for the decrease of the variational bound when the point is
        dropped: squared norm of the corresponding row of ``W = L_uu^{-1} K_uf``,
        i.e. how much of the projected data energy that point carries.
        """
        return np.power(self.W, 2).sum(axis=1)

    def remove_inducing(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        if remove_number is None and reserve_indices is None:
            raise ValueError('Either remove_number or reserve_indices must be provided.')

        keep = np.ones(self.M, dtype=bool)
        if reserve_indices is not None:
            keep[:] = False
            keep[reserve_indices] = True
        else:
            keep[:remove_number] = False

        U_keep = self.U[keep]
        if U_keep.shape[0] == self.M:
            return
        # Rebuild the posterior for the reduced inducing set; the stored training
        # data is reused, so pruning does not discard observations.
        self.fit(self.X, self.Y, U_keep, variance=self.variance, lambda_=self.lambda_)



class SparseDTCGP:
    def __init__(self, kernel: GPy.kern, gaussian_opts: GaussianConfig) -> None:
        self.kern = kernel
        self.gaussian_opts = gaussian_opts
        self.cholesky_valid = False

    def _solve_luu(self, rhs: np.ndarray) -> np.ndarray:
        return solve_triangular(self.Luu, rhs, lower=True)

    def _solve_luu_t(self, rhs: np.ndarray) -> np.ndarray:
        return solve_triangular(self.Luu.T, rhs, lower=False)

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
        self.Kuu = _kern_K(self.kern, self.U)
        self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry
        self.Kuf = _kern_K(self.kern, self.U, self.X)

        # Precompute inverses (with jitter for stability)
        jitter = 1e-8 * np.eye(self.M)
        self.Luu = np.linalg.cholesky(self.Kuu + jitter)

        # --- Factorized form ---
        # W = Luu^{-1} Kuf
        self.W = self._solve_luu(self.Kuf)

        self.W_d_inv_y = (1.0 / self.variance) * self.W @ self.Y

        # K = I + W D^{-1} W^T
        # WD_half = self.W @ np.sqrt(self.D)
        self.K = (1.0 / self.variance) * self.W @ self.W.T
        self.K = (self.K + self.K.T) / 2 + np.eye(self.M)  # ensure symmetry
        
        # Cholesky of K
        self.L = np.linalg.cholesky(self.K)

        self.L_updated = True

        self.update_mu_su(from_scratch=True)

        self.cholesky_valid = True

    # ------------------------------------------------------------------

    def update_mu_su(self, from_scratch: bool = False, Kun: np.ndarray | None = None, Knn: np.ndarray | None = None) -> None:
        if from_scratch:
            self.update_L()

            # self.Kuu = _kern_K(self.kern, self.U)
            # self.Kuu = (self.Kuu + self.Kuu.T) / 2  # ensure symmetry

            # self.Kuu_inv = np.linalg.solve(
            #     self.Luu.T, self._solve_luu(np.eye(self.M))
            # )

            # Posterior covariance Su
            # Sigma = inv[Kuu + Kuf @ inv(D) @ Kfu]
            #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
            #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
            #   = inv(Luu).T @ inv(K) @ inv(Luu)
            #   = Luu^{-T} L^{-T} L^{-1} Luu^{-1}
            # Su = Kuu @ Sigma @ Kuu = Luu L^{-T} L^{-1} Luu.T = tmp.T @ tmp
            # S = np.linalg.solve(self.L, self._solve_luu(np.eye(self.M)))
            # self.Sigma = S.T @ S
            tmp = np.linalg.solve(self.L, self.Luu.T)
            # tmp = np.linalg.solve(self.L, self._solve_luu(self.Kuu))
            self.Su = tmp.T @ tmp  # symmetric MxM

            # Posterior mean mu
            rhs = self._solve_luu((self.Kuf / self.variance) @ self.Y)
            self.mu = self._solve_luu(self.Su.T).T @ rhs
        else:
            # TODO: if new inducing points are too many, then the update will have some error
            # TODO: if new inducing is close to old ones, numerical issues may happen
            Luu_inv_Kun = self._solve_luu(Kun)
            Kuu_inv_kun = self._solve_luu_t(Luu_inv_Kun)
            mu_new = Kuu_inv_kun.T @ self.mu
            self.mu = np.vstack((self.mu, mu_new))

            Sun = self.Su @ Kuu_inv_kun
            Snn = Knn - Luu_inv_Kun.T @ Luu_inv_Kun + Kuu_inv_kun.T @ self.Su @ Kuu_inv_kun
            left = np.vstack((self.Su, Sun.T))
            right = np.vstack((Sun, Snn))
            self.Su = np.hstack((left, right))

    # ------------------------------------------------------------------

    def predict(self, Xs: np.ndarray, method: str = 'mu_su', compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        if method == 'cholesky':
            return self.predict_cholesky(Xs)
        if method == 'mu_su':
            return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
        raise ValueError(
            f"Unknown prediction method '{method}'; expected 'mu_su' or 'cholesky'."
        )
    
    # ------------------------------------------------------------------

    def predict_noiseless(self, Xs: np.ndarray, full_cov: bool, compute_dvar_dx: bool = False) -> Tuple[np.ndarray]:
        return self.predict_mu_su(Xs, compute_dvar_dx=compute_dvar_dx)
    
    # ------------------------------------------------------------------

    def save_model(self, id: str) -> None:
        print(f'[GPR]: Saving Sparse DTC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        save_path = self.gaussian_opts.save_path + id + '.npy'
        _save_sparse_gp(self, save_path, "DTC")

    # ------------------------------------------------------------------

    def load_model(self, id: str) -> None:
        print(f'[GPR]: Loading Sparse DTC GP Model with normalized: {self.gaussian_opts.normalize_y}')
        load_path = self.gaussian_opts.load_path + id + '.npy'
        _load_sparse_gp(self, load_path, "DTC")

    # ------------------------------------------------------------------

    def _require_cholesky(self, operation: str) -> None:
        if not self.cholesky_valid:
            raise RuntimeError(
                f"{self.__class__.__name__}.{operation}: the Cholesky-based state is "
                "stale because add_new_inducing_quick() only maintains the mu_su "
                "(mean/covariance) prediction state. Use predict_noiseless() or "
                "predict(..., method='mu_su'), or call sync_cholesky() first."
            )

    # ------------------------------------------------------------------

    def sync_cholesky(self) -> None:
        """Rebuild the Cholesky-based prediction/update state from U, X and Y."""
        self.Kuf = _kern_K(self.kern, self.U, self.X)
        self.W = self._solve_luu(self.Kuf)
        self.W_d_inv_y = (1.0 / self.variance) * self.W @ self.Y
        self.L_updated = False
        self.cholesky_valid = True
        try:
            self.update_L()
        except Exception:
            self.cholesky_valid = False
            raise
            
    # ------------------------------------------------------------------
    
    def dk_dx(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.atleast_2d(Xs)
        if Xs.shape[0] != 1:
            raise ValueError(
                "dk_dx/dmu_dx only support a single test input, got "
                f"{Xs.shape[0]} rows."
            )
        if _is_rbf_kernel(self.kern):
            return _rbf_gradient(self.kern, self.U, Xs)
        mean_jac = np.zeros((self.U.shape[0], self.U.shape[1]))
        for i in range(self.U.shape[1]):
            mean_jac[:, i] = self.kern.dK_dX2(self.U, Xs, i).squeeze()
        return mean_jac
    
    # ------------------------------------------------------------------
    
    def dmu_dx(self, Xs: np.ndarray) -> np.ndarray:
        rhs = self._solve_luu(self.mu)
        dk_dx = self.dk_dx(Xs)
        lhs = self._solve_luu(dk_dx)
        result = lhs.T @ rhs
        return result
    
    # ------------------------------------------------------------------
    
    def dvar_dx(self, Xs: np.ndarray, Ws: np.ndarray, tmp: np.ndarray) -> np.ndarray:
        dk_dx = self.dk_dx(Xs)
        Luu_inv_dk_dx = self._solve_luu(dk_dx)
        Kuu_inv_dk_dx = self._solve_luu_t(Luu_inv_dk_dx)
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

        Kss = _kern_K(self.kern, Xs)
        Ksu = _kern_K(self.kern, Xs, self.U)
        Ws = self._solve_luu(Ksu.T)
        # tmp = np.linalg.solve(self.L, self._solve_luu(Ksu.T))
        tmp2 = self._solve_luu_t(Ws)
        
        mu = Ws.T @ self._solve_luu(self.mu)
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

        self._require_cholesky("predict_cholesky")

        Kss = _kern_K(self.kern, Xs)
        Ksu = _kern_K(self.kern, Xs, self.U)
        Ws = self._solve_luu(Ksu.T)
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
        K_zU = _kern_K(self.kern, X_new, self.U)        # shape (1, M)
        # Phi_k = (K_zU @ self.Kuu_inv).T            # (M, 1)
        Phi_k = self._solve_luu_t(self._solve_luu(K_zU.T))

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
        K_XU = _kern_K(self.kern, X_new, self.U)      # (B, M)
        # Phi_b = K_XU @ self.Kuu_inv              # (B, M)
        Phi_b = self._solve_luu_t(self._solve_luu(K_XU.T)).T

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
            if not self.cholesky_valid:
                # Quick updates keep only the mu_su state; there is nothing to
                # rebuild until sync_cholesky() is called.
                return
            self.K = (1 / self.variance) * self.W @ self.W.T
            self.K = (self.K + self.K.T) / 2 + np.eye(self.M)
            self.L = np.linalg.cholesky(self.K)
            self.L_updated = True

    def add_new_data(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        self._require_cholesky("add_new_data")
        if self.gaussian_opts.normalize_y is True:
            Y_new = self.normalizer.normalize(Y_new)
            
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.N = self.X.shape[0]

        Kuf_new = _kern_K(self.kern, self.U, X_new)
        W_new = self._solve_luu(Kuf_new)
        self.Kuf = np.hstack((self.Kuf, Kuf_new))
        self.W = np.hstack((self.W, W_new))

        W_d_inv_y_new = 1.0 / self.variance * W_new @ Y_new
        self.W_d_inv_y += W_d_inv_y_new.reshape(self.W_d_inv_y.shape)
        
        self.L_updated = False
        # Keep the mu_su posterior consistent with the enlarged training set.
        self.update_mu_su(from_scratch=True)

    def remove_data(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        self._require_cholesky("remove_data")
        if remove_number is None and reserve_indices is None:
            raise ValueError('Either remove_number or reserve_indices must be provided.')

        keep = np.zeros(self.N, dtype=bool)
        if reserve_indices is not None:
            keep[reserve_indices] = True
        else:
            keep[remove_number:] = True

        self.X = self.X[keep]
        self.Y = self.Y[keep]
        self.N = self.X.shape[0]
        self.sync_cholesky()
        self.update_mu_su(from_scratch=True)

    def check_new_inducing_var(self, U_new: np.ndarray) -> np.ndarray:
        a = _kern_K(self.kern, self.U, U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability

        c = self._solve_luu(a)   # (M, 1)

        return aa - c.T @ c, aa
    
    def add_new_inducing(self, U_new: np.ndarray) -> None:
        self._require_cholesky("add_new_inducing")
        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        a = _kern_K(self.kern, self.U[:-1], U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability
        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        Kuf_new = _kern_K(self.kern, U_new, self.X)
        self.Kuf = np.vstack((self.Kuf, Kuf_new))

        W_new = (Kuf_new - self.Luu[-1:, :-1] @ self.W) / self.Luu[-1, -1]
        self.W = np.vstack((self.W, W_new))

        self.W_d_inv_y = 1.0 / self.variance * self.W @ self.Y

        self.L_updated = False

    def add_new_inducing_quick(self, U_new: np.ndarray) -> None:
        a = _kern_K(self.kern, self.U, U_new)
        aa = _kern_K(self.kern, U_new) + 1e-8  # jitter for numerical stability

        self.U = np.vstack((self.U, U_new))
        self.M = self.U.shape[0]

        self.update_mu_su(Kun=a, Knn=aa)
        self.Luu = cholesky_add_row(self.Luu, a, aa)

        # Quick updates intentionally skip Kuf/W/diag/L: only the mu_su
        # prediction state is kept up to date. Cholesky users must call
        # sync_cholesky() first (or stick to predict_noiseless).
        self.cholesky_valid = False
        self.L_updated = False
        
    def remove_inducing(self, remove_number: int | None = None, reserve_indices: list[int] | None = None) -> None:
        raise NotImplementedError(
            "remove_inducing is not implemented; rebuild the GP with fewer inducing points instead."
        )

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
        if gaussian_opts.read_model is True:
            # Loading a saved model does not need the training data file.
            train_data, inducing_data = None, None
        else:
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
                    lambda_=float(getattr(gaussian_opts, "lambda_", 1.0)),
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
