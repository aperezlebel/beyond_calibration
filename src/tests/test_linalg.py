import numpy as np

from src._linalg import (
    create_orthonormal_vector,
    gram_schmidt_orthonormalization,
    is_orthonormal_basis,
    projection,
)


def test_create_orthonormal_vector():
    rs = np.random.RandomState(0)
    x = rs.uniform(-100, 100, size=10)
    y = create_orthonormal_vector(x)
    assert not np.allclose(y, np.zeros_like(y))
    assert np.allclose(np.vdot(x, y), 0)
    assert np.allclose(np.linalg.norm(y), 1)


def test_projection():
    d = 5
    rs = np.random.RandomState(0)
    x = rs.uniform(size=d)
    y = rs.uniform(size=d)
    z = projection(x, y)
    assert np.allclose(z, projection(z, y))


def test_gram_schmidt_orthonormalization():
    d = 5
    rs = np.random.RandomState(0)
    P = rs.uniform(size=(d, d))
    P = gram_schmidt_orthonormalization(P)

    assert is_orthonormal_basis(P)
