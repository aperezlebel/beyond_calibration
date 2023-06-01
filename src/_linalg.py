import numpy as np


def create_orthonormal_vector(x):
    """Return a vector orthonormal to the given one."""
    if np.allclose(x, np.zeros_like(x)):
        raise ValueError("x is null")
    if len(x) < 2:
        raise ValueError("x must be at least 2 dimensional to find orthonormal vector")
    y = np.zeros_like(x)
    m = np.argmax(x != 0)
    n = (m + 1) % len(x)
    y[n] = x[m]
    y[m] = -x[n]
    y /= np.linalg.norm(y)
    return y


def is_orthogonal_basis(P):
    d = P.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            if not np.allclose(np.vdot(P[:, i], P[:, j]), 0):
                return False
    return True


def is_orthonormal_basis(P):
    return np.allclose(np.linalg.norm(P, axis=0), 1) and is_orthogonal_basis(P)


def projection(x, y):
    """Projection of x onto y."""
    return np.vdot(x, y) / np.square(np.linalg.norm(y)) * y


def gram_schmidt_orthonormalization(P):
    """Turn given basis of R^d into an orthonormal basis."""
    P = P.copy()

    # Make it orthogonal using Gram-Schmidt orthogonalization procedure
    for i in range(1, P.shape[1]):
        P[:, i] -= np.sum([projection(P[:, i], P[:, j]) for j in range(i)], axis=0)

    # Normalize
    P = np.divide(P, np.linalg.norm(P, axis=0, keepdims=True))

    return P
