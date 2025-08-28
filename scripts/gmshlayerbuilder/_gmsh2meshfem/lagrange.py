import numpy as np
import numpy.typing as npt


def build_lagrange_polys(knots: npt.ArrayLike) -> np.ndarray:
    """Generates the Lagrange interpolating polynomial coefficients
    `L` of shape `(N,N)`, where $L_i(x) = \\sum_{k=0}^{N-1} L[i,k] x^k$.

    Args:
        knots (ArrayLike): set of points for which
            $L_i(knots[j]) = \\delta_{ij}$.
            This must be shape `(N,)`.

    Returns:
        np.ndarray: The coefficient array.
    """
    knots = np.array(knots)
    if len(knots.shape) != 1:
        raise ValueError(
            f"knots must be ArrayLike with shape (N,). Found {knots.shape}."
        )
    N = knots.shape[0]
    L = np.zeros((N, N))
    L[:, 0] = 1
    for k in range(N):
        kfilter = (np.arange(N) != k)
        tmp = L[kfilter, :-1].copy()
        fac = 1 / (knots[kfilter] - knots[k])[:, None]
        L[kfilter, :-1] = -knots[k] * fac * tmp
        L[kfilter, 1:] += fac * tmp
    return L


def differentiate_polys(L: np.ndarray) -> np.ndarray:
    """Generates the first derivative of the polynomials given as `L`.
    `L` is expected to be of shape `(N,M)`, where `M` is the polynomial degree.
    The polynomials are given by $L_i(x) = \\sum_{k=0}^{M-1} L[i,k] x^k$.

    Args:
        L (np.ndarray): The coefficient array of L

    Returns:
        np.ndarray: The coefficient array of L', with shape `(N,M-1)`.
    """
    return L[:, 1:] * np.arange(1, L.shape[1])[None, :]


def lagrange_deriv_at_knots(knots: npt.ArrayLike) -> np.ndarray:
    """For the given collection `knots[i]`, computes $L_i'(knots[j])$,
    storing them into `DL[i,j]`.

    Args:
        knots (ArrayLike): set of points for which
            $L_i(knots[j]) = \\delta_{ij}$.
            This must be shape `(N,)`.

    Returns:
        np.ndarray: The evaluated derivatives
    """
    knots = np.array(knots)
    if len(knots.shape) != 1:
        raise ValueError(
            f"knots must be ArrayLike with shape (N,). Found {knots.shape}."
        )
    N = knots.shape[0]
    # use product rule

    #collect products here:
    DL = np.ones((N, N))
    DL_diag = np.zeros(N)
    for j in range(N):
        jfilter = (np.arange(N) != j)
        DL_diag[jfilter] += 1/(knots[jfilter] - knots[j])
        knotdiff = knots[:] - knots[j]
        knotdiff[j] = 1
        DL[jfilter,:] *= knotdiff/(knots[jfilter,None] - knots[j])

    DL[np.arange(N),np.arange(N)] = DL_diag
    return DL
