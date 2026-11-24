import numpy as np

def block_matrix_inversion(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    A_inv: np.ndarray | None = None
) -> np.ndarray:
    """
    Perform block matrix inversion using the formula for inverting a 2x2 block matrix.

    Given a block matrix:
        M = | A  B |
            | C  D |

    The inverse of M is given by:
        M_inv = | E  F |
                | G  H |

    where:
        E = A_inv + A_inv * B * S_inv * C * A_inv
        F = -A_inv * B * S_inv
        G = -S_inv * C * A_inv
        H = S_inv

    and S = D - C * A_inv * B is the Schur complement of A in M.

    Parameters
    ----------
    A : np.ndarray, shape (m, m)
        Top-left block of the matrix.
    B : np.ndarray, shape (m, n)
        Top-right block of the matrix.
    C : np.ndarray, shape (n, m)
        Bottom-left block of the matrix.
    D : np.ndarray, shape (n, n)
        Bottom-right block of the matrix.

    Returns
    -------
    M_inv : np.ndarray, shape (m+n, m+n)
        Inverse of the block matrix M.
    """
    # Invert A
    if A_inv is None:
        A_inv = np.linalg.inv(A)

    # Compute Schur complement S
    S = D - C @ A_inv @ B
    S_inv = np.linalg.inv(S)

    # Compute blocks of the inverse matrix
    E = A_inv + A_inv @ B @ S_inv @ C @ A_inv
    F = -A_inv @ B @ S_inv
    G = -S_inv @ C @ A_inv
    H = S_inv

    # Combine blocks into full inverse matrix
    top = np.hstack((E, F))
    bottom = np.hstack((G, H))
    M_inv = np.vstack((top, bottom))

    return M_inv

def cholesky_add_row(L: np.ndarray, a: np.ndarray, aa: np.ndarray) -> np.ndarray:
    """
    L:  (M, M)
    a:  (M, M_new)
    aa: (M_new, M_new)
    """
    c = np.linalg.solve(L, a)
    d_sq = aa - c.T @ c
    if np.any(d_sq <= 0.0):
        raise FloatingPointError(
            "cholesky_add_row: the updated matrix is not positive definite; "
            "the new inducing point is probably (nearly) linearly dependent on "
            "the existing ones."
        )
    d = np.sqrt(d_sq)
    left = np.vstack((L, c.T))
    right = np.vstack((np.zeros((L.shape[0], d.shape[1])), d))
    L_new = np.hstack((left, right))
    return L_new