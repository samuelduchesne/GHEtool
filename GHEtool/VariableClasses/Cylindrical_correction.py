"""
This document contains the code to add the cylindrical correction to the pygfunction package,
as was documented here: https://github.com/MassimoCimmino/pygfunction/issues/269.
Note that this is a temporary solution, until issue #44 of pygfunction is solved.
"""

import pygfunction as gt
import numpy as np

from pygfunction.boreholes import Borehole, _EquivalentBorehole, find_duplicates
from pygfunction.heat_transfer import finite_line_source, finite_line_source_vectorized, \
    finite_line_source_equivalent_boreholes_vectorized
from pygfunction.heat_transfer import _finite_line_source_integrand as _finite_line_source_integrand_pygf
from pygfunction.heat_transfer import \
    _finite_line_source_equivalent_boreholes_integrand as _finite_line_source_equivalent_boreholes_integrand_pygf
from pygfunction.networks import network_thermal_resistance

from scipy.integrate import quad_vec
from scipy.special import erf, j0, j1, y0, y1

from scipy.interpolate import interp1d as interp1d
from time import perf_counter

# Gauss-Kronrod 21 nodes and weights (same tables as scipy.integrate.quad_vec)
_GK21_NODES = np.array([
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
    0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
    0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
    0.294392862701460198131126603103866, 0.148874338981631210884826001129720,
    0.,
    -0.148874338981631210884826001129720, -0.294392862701460198131126603103866,
    -0.433395394129247190799265943165784, -0.562757134668604683339000099272694,
    -0.679409568299024406234327365114874, -0.780817726586416897063717578345042,
    -0.865063366688984510732096688423493, -0.930157491355708226001207180059508,
    -0.973906528517171720077964012084452, -0.995657163025808080735527280689003])
_GK21_WEIGHTS_GAUSS = np.array([
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
    0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
    0.295524224714752870173892994651338, 0.295524224714752870173892994651338,
    0.269266719309996355091226921569469, 0.219086362515982043995534934228163,
    0.149451349150580593145776339657697, 0.066671344308688137593568809893332])
_GK21_WEIGHTS_KRONROD = np.array([
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
    0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
    0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
    0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
    0.147739104901338491374841515972068, 0.142775938577060080797094273138717,
    0.134709217311473325928054001771707, 0.123491976262065851077958109831074,
    0.109387158802297641899210590325805, 0.093125454583697605535065465083366,
    0.075039674810919952767043140916190, 0.054755896574351996031381300244580,
    0.032558162307964727478818972459390, 0.011694638867371874278064396062192])


def _erfint(x):
    """Integral of the error function (identical to pygfunction.utilities.erfint)."""
    return x * erf(x) - 1.0 / np.sqrt(np.pi) * (1.0 - np.exp(-x ** 2))


def _gk21_evaluate(f_batched, x1, x2):
    """
    Evaluate the Gauss-Kronrod 21 rule on a batch of subintervals in a single
    vectorized integrand call. Uses the same nodes, weights and error model as
    scipy.integrate.quad_vec's _quadrature_gk21.

    Parameters
    ----------
    f_batched : callable
        Function taking a 1D array of quadrature points (shape (n,)) and returning
        the integrand evaluated at all points, with the point axis first
        (shape (n, ...)).
    x1, x2 : np.ndarray
        1D arrays with the subinterval bounds.

    Returns
    -------
    integrals : np.ndarray
        Integral estimates, shape (len(x1), ...).
    err : np.ndarray
        Error estimates (2-norm over the payload), shape (len(x1),).
    """
    eps = np.finfo(np.float64).eps
    m = len(x1)
    c = 0.5 * (x1 + x2)
    h = 0.5 * (x2 - x1)
    nodes = c[:, None] + h[:, None] * _GK21_NODES[None, :]
    ff = f_batched(nodes.reshape(-1))
    payload_shape = ff.shape[1:]
    ff = ff.reshape((m, 21) + payload_shape)
    payload_axes = tuple(range(1, 1 + len(payload_shape)))

    s_k = np.einsum('j,mj...->m...', _GK21_WEIGHTS_KRONROD, ff)
    s_g = np.einsum('j,mj...->m...', _GK21_WEIGHTS_GAUSS, ff[:, 1::2])
    s_k_abs = np.einsum('j,mj...->m...', _GK21_WEIGHTS_KRONROD, np.abs(ff))
    y0 = s_k / 2.0
    s_k_dabs = np.einsum('j,mj...->m...', _GK21_WEIGHTS_KRONROD,
                         np.abs(ff - y0[:, None]))

    def norm(x):
        return np.sqrt(np.sum(np.abs(x) ** 2, axis=payload_axes))

    h_abs = np.abs(h)
    err = norm(s_k - s_g) * h_abs
    dabs = norm(s_k_dabs) * h_abs
    mask = (dabs != 0) & (err != 0)
    err[mask] = dabs[mask] * np.minimum(1.0, (200 * err[mask] / dabs[mask]) ** 1.5)
    round_err = norm(50 * eps * s_k_abs) * h_abs
    err = np.where(round_err > np.finfo(np.float64).tiny, np.maximum(err, round_err), err)
    integrals = h[(slice(None),) + (None,) * len(payload_shape)] * s_k
    return integrals, err


def _adaptive_gk21_batched(f_batched, a, b, epsabs=1e-200, epsrel=1e-8, max_rounds=60):
    """
    Adaptive Gauss-Kronrod 21 integration of a vector-valued integrand over many
    independent finite intervals at once.

    This applies the same GK21 rule, error model, tolerances and worst-first
    bisection strategy as ``scipy.integrate.quad_vec`` (the integrator pygfunction
    uses for the finite line source solution), but evaluates the integrand for the
    quadrature nodes of all intervals that still need refinement in a single
    vectorized call per round. This removes the per-node Python and numpy-dispatch
    overhead without changing the quadrature rule, the error estimate or the
    convergence criterion, so the achieved integration accuracy is the same as with
    quad_vec.

    Parameters
    ----------
    f_batched : callable
        Function taking a 1D array of quadrature points s (shape (n,)) and returning
        the integrand evaluated at all points, with the point axis first
        (shape (n, ...)).
    a, b : np.ndarray
        1D arrays with the (finite) integration bounds per interval.
    epsabs : float
        Absolute tolerance (same default as scipy.integrate.quad_vec).
    epsrel : float
        Relative tolerance (same default as scipy.integrate.quad_vec).
    max_rounds : int
        Maximum number of subdivision rounds.

    Returns
    -------
    result : np.ndarray
        Array of shape (len(a), ...) with the integrals for every interval.
    """
    n_intervals = len(a)

    # state of all current subintervals
    idx = np.arange(n_intervals)
    x1 = np.asarray(a, dtype=np.float64).copy()
    x2 = np.asarray(b, dtype=np.float64).copy()
    integrals, err = _gk21_evaluate(f_batched, x1, x2)
    payload_shape = integrals.shape[1:]
    payload_axes = tuple(range(1, 1 + len(payload_shape)))

    for _ in range(max_rounds):
        # aggregate per original interval
        total = np.zeros((n_intervals,) + payload_shape)
        np.add.at(total, idx, integrals)
        err_total = np.zeros(n_intervals)
        np.add.at(err_total, idx, err)

        tol = np.maximum(epsabs, epsrel * np.sqrt(np.sum(total ** 2, axis=payload_axes)))
        interval_converged = err_total <= tol
        pending = ~interval_converged[idx]
        if not np.any(pending):
            return total

        # per unconverged interval, bisect its worst subinterval (worst-first,
        # like scipy's quad_vec with workers=1)
        idx_p = idx[pending]
        err_p = err[pending]
        order = np.lexsort((-err_p, idx_p))
        idx_sorted = idx_p[order]
        first_of_group = np.ones(len(idx_sorted), dtype=bool)
        first_of_group[1:] = idx_sorted[1:] != idx_sorted[:-1]
        worst_local = order[first_of_group]
        worst = np.where(pending)[0][worst_local]

        x1_w, x2_w = x1[worst], x2[worst]
        mid = 0.5 * (x1_w + x2_w)
        splittable = (mid > x1_w) & (mid < x2_w)
        if not np.any(splittable):
            return total
        worst = worst[splittable]
        x1_w, x2_w, mid = x1_w[splittable], x2_w[splittable], mid[splittable]

        # evaluate the two halves of every bisected subinterval
        new_x1 = np.concatenate([x1_w, mid])
        new_x2 = np.concatenate([mid, x2_w])
        new_idx = np.concatenate([idx[worst], idx[worst]])
        new_integrals, new_err = _gk21_evaluate(f_batched, new_x1, new_x2)

        # replace the bisected subintervals with their halves
        keep = np.ones(len(idx), dtype=bool)
        keep[worst] = False
        idx = np.concatenate([idx[keep], new_idx])
        x1 = np.concatenate([x1[keep], new_x1])
        x2 = np.concatenate([x2[keep], new_x2])
        integrals = np.concatenate([integrals[keep], new_integrals])
        err = np.concatenate([err[keep], new_err])

    total = np.zeros((n_intervals,) + payload_shape)
    np.add.at(total, idx, integrals)
    return total


def _finite_line_source_vectorized_batched(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True,
        approximation=False, N=10):
    """
    Evaluate the Finite Line Source (FLS) solution for an array of time values.

    This is numerically equivalent to pygfunction's
    :func:`finite_line_source_vectorized`: it evaluates the same one-integral form
    of the FLS solution with the same adaptive Gauss-Kronrod quadrature rule and
    tolerances, but batches the integrand evaluations over the quadrature nodes of
    all time intervals at once, which is considerably faster. It falls back to the
    pygfunction implementation whenever the inputs are not of the expected form.
    """
    q_dim = np.ndim(D2 - D1 + H2) + 1
    if (approximation or not (reaSource and imgSource) or np.ndim(time) == 0
            or isinstance(time, (np.floating, float)) or len(np.atleast_1d(time)) < 2
            or np.ndim(dis) > q_dim - 1):
        return finite_line_source_vectorized(
            time, alpha, dis, H1, D1, H2, D2, reaSource=reaSource,
            imgSource=imgSource, approximation=approximation, N=N)

    p = np.array([1., -1., 1., -1., 1., -1., 1., -1.])
    q = np.stack([D2 - D1 + H2,
                  D2 - D1,
                  D2 - D1 - H1,
                  D2 - D1 + H2 - H1,
                  D2 + D1 + H2,
                  D2 + D1,
                  D2 + D1 + H1,
                  D2 + D1 + H2 + H1],
                 axis=-1)
    dis_arr = np.asarray(dis, dtype=np.float64)
    trailing = (1,) * (q.ndim - 1)
    # many of the q values coincide (segments share dimensions), so erfint is only
    # evaluated at the unique values and the result is scattered back afterwards.
    # This yields exactly the same values, since erfint works pointwise.
    q_unique, q_inverse = np.unique(q.reshape(-1), return_inverse=True)

    def f_batched(s):
        n = len(s)
        s_col = s.reshape((n,) + trailing)
        erfint_q = _erfint(np.multiply.outer(s, q_unique))[:, q_inverse].reshape((n,) + q.shape)
        inner = np.einsum('k,...k->...', p, erfint_q)
        exp_term = np.exp(-dis_arr ** 2 * s_col ** 2) if dis_arr.ndim == 0 else \
            np.exp(-dis_arr[None, ...] ** 2 * s.reshape((n,) + (1,) * dis_arr.ndim) ** 2
                   ).reshape((n,) + (1,) * (q.ndim - 1 - dis_arr.ndim) + dis_arr.shape)
        return s_col ** -2 * exp_term * inner

    # Lower bounds of integration
    a = 1.0 / np.sqrt(4.0 * alpha * np.asarray(time, dtype=np.float64))
    # first time value: integral over the semi-infinite interval, evaluated with
    # scipy's quad_vec exactly as pygfunction does
    f = _finite_line_source_integrand_pygf(dis, H1, D1, H2, D2, reaSource, imgSource)
    h_first = 0.5 / H2 * quad_vec(f, a[0], np.inf)[0]
    # remaining time values: batched adaptive quadrature over the finite intervals
    pieces = _adaptive_gk21_batched(f_batched, a[1:], a[:-1])
    pieces = np.asarray(0.5 / H2)[..., None] * np.moveaxis(pieces, 0, -1)
    h = np.cumsum(np.concatenate([h_first[..., None], pieces], axis=-1), axis=-1)
    return h


def _finite_line_source_equivalent_boreholes_vectorized_batched(
        time, alpha, dis, wDis, H1, D1, H2, D2, N2, reaSource=True, imgSource=True):
    """
    Evaluate the equivalent Finite Line Source (FLS) solution for an array of time
    values.

    This is numerically equivalent to pygfunction's
    :func:`finite_line_source_equivalent_boreholes_vectorized` (same integral, same
    adaptive Gauss-Kronrod rule and tolerances), but batches the integrand
    evaluations over the quadrature nodes of all time intervals at once. It falls
    back to the pygfunction implementation whenever the inputs are not of the
    expected form.
    """
    dis_arr = np.asarray(dis, dtype=np.float64)
    wDis_arr = np.asarray(wDis, dtype=np.float64)
    if (not (reaSource and imgSource) or np.ndim(time) == 0
            or isinstance(time, (np.floating, float)) or len(np.atleast_1d(time)) < 2
            or dis_arr.ndim != 2 or wDis_arr.ndim != 2):
        return finite_line_source_equivalent_boreholes_vectorized(
            time, alpha, dis, wDis, H1, D1, H2, D2, N2,
            reaSource=reaSource, imgSource=imgSource)

    p = np.array([1., -1., 1., -1., 1., -1., 1., -1.])
    q = np.stack([D2 - D1 + H2,
                  D2 - D1,
                  D2 - D1 - H1,
                  D2 - D1 + H2 - H1,
                  D2 + D1 + H2,
                  D2 + D1,
                  D2 + D1 + H1,
                  D2 + D1 + H2 + H1],
                 axis=-1)

    # many of the q values coincide (segments share dimensions), so erfint is only
    # evaluated at the unique values and the result is scattered back afterwards.
    # This yields exactly the same values, since erfint works pointwise.
    q_unique, q_inverse = np.unique(q.reshape(-1), return_inverse=True)

    def f_batched(s):
        n = len(s)
        s2 = (s ** 2).reshape((n, 1, 1))
        erfint_q = _erfint(np.multiply.outer(s, q_unique))[:, q_inverse].reshape((n,) + q.shape)
        inner = np.einsum('k,...k->...', p, erfint_q)
        # ( exp(-dis²s²) @ wDis ).T per quadrature point
        mm = np.swapaxes(np.exp(-dis_arr[None, ...] ** 2 * s2) @ wDis_arr, -1, -2)
        return s2 ** -1 * mm * inner

    a = 1.0 / np.sqrt(4.0 * alpha * np.asarray(time, dtype=np.float64))
    f = _finite_line_source_equivalent_boreholes_integrand_pygf(
        dis, wDis, H1, D1, H2, D2, N2, reaSource, imgSource)
    h_first = 0.5 / (N2 * H2) * quad_vec(f, a[0], np.inf)[0]
    pieces = _adaptive_gk21_batched(f_batched, a[1:], a[:-1])
    pieces = np.asarray(0.5 / (N2 * H2))[..., None] * np.moveaxis(pieces, 0, -1)
    h = np.cumsum(np.concatenate([h_first[..., None], pieces], axis=-1), axis=-1)
    return h


# cache for the cylindrical heat source correction: during an iterative sizing the
# correction is requested over and over with identical (time, alpha, r, r_b) inputs
_CHS_CACHE: dict = {}
_CHS_CACHE_MAX_SIZE: int = 32


# update pygfunction
def cylindrical_heat_source(
        time, alpha, r, r_b):
    """
    Evaluate the Cylindrical Heat Source (CHS) solution.
    This function uses a numerical quadrature to evaluate the CHS solution, as
    proposed by Carslaw and Jaeger [#CarslawJaeger1946]_. The CHS solution
    is given by:
        .. math::
            G(r,t) =
            \\frac{1}{\pi^2}
            \\int_{0}^{\\infty}
            \\frac{1}{s^2}
            \\frac{e^{-Fo s^2} - 1}{J_1^2(s) + Y_1^2(s)}
            [J_0(ps)Y_1(s) - J_1(s)Y_0(ps)]ds
    Parameters
    ----------
    time : float
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    r : float
        Radial distance from the borehole axis (in m).
    r_b : float
        Borehole radius (in m).
    Returns
    -------
    G : float
        Value of the CHS solution. The temperature at a distance r from
        borehole is:
        .. math:: \\Delta T(r,t) = T_g - \\frac{Q}{k_s H} G(r,t)
    Examples
    --------
    >>> G = gt.heat_transfer.cylindrical_heat_source(4*168*3600., 1.0e-6, 0.1, 0.075)
    G =
    References
    ----------
    .. [#CarslawJaeger1946] Carslaw, H.S., & Jaeger, J.C. (1946). The Laplace
       transformation: Problems on the cylinder and sphere, in: OU Press (Ed.),
       Conduction of heat in solids, Oxford University, Oxford, pp. 327-352.
    """
    # def _CHS(u, Fo, p):
    #     # Function to integrate
    #     CHS_integrand = ( 1. / (u**2 * np.pi**2) * (np.exp(-u**2 * Fo) - 1.0)
    #         / (j1(u)**2 + y1(u)**2) * (j0(p * u) * y1(u) - j1(u) * y0(p * u)) )
    #     return CHS_integrand
    CHS_integrand = lambda u: (1. / (u ** 2 * np.pi ** 2) * (np.exp(-u ** 2 * Fo) - 1.0)
                               / (j1(u) ** 2 + y1(u) ** 2) * (j0(p * u) * y1(u) - j1(u) * y0(p * u)))

    # the CHS solution only depends on (time, alpha, r, r_b); during an iterative
    # sizing it is requested many times with identical inputs, so cache the result
    time_arr = np.asarray(time, dtype=np.float64)
    key = (float(alpha), float(r), float(r_b), time_arr.shape, time_arr.tobytes())
    cached = _CHS_CACHE.get(key)
    if cached is not None:
        return cached.copy() if isinstance(cached, np.ndarray) else cached

    # Fourier number
    Fo = alpha * time / r_b ** 2
    # Normalized distance from borehole axis
    p = r / r_b
    # Lower bound of integration
    a = 0.
    # Upper bound of integration
    b = np.inf
    # Evaluate integral using Gauss-Kronrod
    G = quad_vec(CHS_integrand, a, b)[0]

    if len(_CHS_CACHE) >= _CHS_CACHE_MAX_SIZE:
        _CHS_CACHE.clear()
    _CHS_CACHE[key] = G.copy() if isinstance(G, np.ndarray) else G
    return G


def infinite_line_source(
        time, alpha, r):
    """
    Evaluate the Infinit Line Source (ILS) solution.
    This function uses the exponential integral to evaluate the ILS solution.
    The ILS solution is given by:
        .. math::
            I(r,t) = E_1(\\frac{r^2}{4 \\alpha t})
    Parameters
    ----------
    time : float
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    r : float
        Radial distance from the borehole axis (in m).
    borehole : Borehole object
        Borehole object of the borehole extracting heat.
    Returns
    -------
    I : float
        Value of the ILS solution. The temperature at a distance r from
        borehole is:
        .. math:: \\Delta T(r,t) = T_g - \\frac{Q}{4 \\pi k_s H} I(r,t)
    Examples
    --------
    >>> b = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> G = gt.heat_transfer.infinite_line_source(4*168*3600., 1.0e-6, 0.1, b)
    I =
    """
    I = gt.utilities.exp1(r ** 2 / (4 * alpha * time))

    return I


def thermal_response_factors(self, time, alpha, kind='linear'):
    """
    Evaluate the segment-to-segment thermal response factors for all pairs
    of segments in the borefield at all time steps using the finite line
    source solution.

    This method returns a scipy.interpolate.interp1d object of the matrix
    of thermal response factors, containing a copy of the matrix accessible
    by h_ij.y[:nSources,:nSources,:nt+1]. The first index along the
    third axis corresponds to time t=0. The interp1d object can be used to
    obtain thermal response factors at any intermediate time by
    h_ij(t)[:nSources,:nSources].

    Parameters
    ----------
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.

    Returns
    -------
    h_ij : interp1d
        interp1d object (scipy.interpolate) of the matrix of
        segment-to-segment thermal response factors.

    """
    if self.disp:
        print('Calculating segment to segment response factors ...',
              end='')
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize chrono
    tic = perf_counter()
    # Initialize segment-to-segment response factors
    h_ij = np.zeros((self.nSources, self.nSources, nt + 1), dtype=self.dtype)
    segment_lengths = self.segment_lengths()

    # ---------------------------------------------------------------------
    # Segment-to-segment thermal response factors for borehole-to-borehole
    # thermal interactions
    # ---------------------------------------------------------------------
    # Groups correspond to unique pairs of borehole dimensions
    for pairs in self.borehole_to_borehole:
        i, j = pairs[0]
        # Prepare inputs to the FLS function
        dis, wDis = self._find_unique_distances(self.dis, pairs)
        H1, D1, H2, D2, i_pair, j_pair, k_pair = \
            self._map_axial_segment_pairs(i, j)
        H1 = H1.reshape(1, -1)
        H2 = H2.reshape(1, -1)
        D1 = D1.reshape(1, -1)
        D2 = D2.reshape(1, -1)
        N2 = np.array(
            [[self.boreholes[j].nBoreholes for (i, j) in pairs]]).T
        # Evaluate FLS at all time steps
        h = _finite_line_source_equivalent_boreholes_vectorized_batched(
            time, alpha, dis, wDis, H1, D1, H2, D2, N2)
        # Broadcast values to h_ij matrix
        for k, (i, j) in enumerate(pairs):
            i_segment = self._i0Segments[i] + i_pair
            j_segment = self._i0Segments[j] + j_pair
            h_ij[j_segment, i_segment, 1:] = h[k, k_pair, :]
            if not i == j:
                h_ij[i_segment, j_segment, 1:] = (h[k, k_pair, :].T \
                                                  * segment_lengths[j_segment] / segment_lengths[i_segment]).T

    # ---------------------------------------------------------------------
    # Segment-to-segment thermal response factors for same-borehole thermal
    # interactions
    # ---------------------------------------------------------------------
    # Groups correspond to unique borehole dimensions
    for group in self.borehole_to_self:
        # Index of first borehole in group
        i = group[0]
        # Find segment-to-segment similarities
        H1, D1, H2, D2, i_pair, j_pair, k_pair = \
            self._map_axial_segment_pairs(i, i)
        # Evaluate FLS at all time steps
        H1 = H1.reshape(1, -1)
        H2 = H2.reshape(1, -1)
        D1 = D1.reshape(1, -1)
        D2 = D2.reshape(1, -1)
        if self.cylindrical_correction:
            dis = 0.0005 * self.boreholes[i].H
        else:
            dis = self.boreholes[i].r_b
        h = _finite_line_source_vectorized_batched(
            time, alpha, dis, H1, D1, H2, D2,
            approximation=self.approximate_FLS, N=self.nFLS)
        # the correction terms only depend on the group (not the individual
        # borehole), so they are computed once outside the loop
        if self.cylindrical_correction:
            r_b = self.boreholes[group[0]].r_b
            h_ils = infinite_line_source(time, alpha, dis)
            h_chs = cylindrical_heat_source(time, alpha, r_b, r_b)
            correction = 2 * np.pi * h_chs - 0.5 * h_ils
        # Broadcast values to h_ij matrix
        for i in group:
            i_segment = self._i0Segments[i] + i_pair
            j_segment = self._i0Segments[i] + j_pair
            h_ij[j_segment, i_segment, 1:] = \
                h_ij[j_segment, i_segment, 1:] + h[0, k_pair, :]

            if self.cylindrical_correction:
                ii_segment = j_segment[j_segment == i_segment]
                h_ij[ii_segment, ii_segment, 1:] = (
                        h_ij[ii_segment, ii_segment, 1:] + correction)

    # Return 2d array if time is a scalar
    if np.isscalar(time):
        h_ij = h_ij[:, :, 1]

    # Interp1d object for thermal response factors
    h_ij = interp1d(np.hstack((0., time)), h_ij,
                    kind=kind, copy=True, axis=2)
    toc = perf_counter()
    if self.disp: print(f' {toc - tic:.3f} sec')

    return h_ij


def solve(self, time, alpha):
    """
    Build and solve the system of equations.

    Parameters
    ----------
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    Returns
    -------
    gFunc : float or array
        Values of the g-function
    """
    # Number of time values
    self.time = time
    nt = len(self.time)
    # Evaluate threshold time for g-function linearization
    if self.linear_threshold is None:
        if self.cylindrical_correction:
            time_threshold = 0.
        else:
            time_threshold = self.r_b_max ** 2 / (25 * alpha)
    else:
        time_threshold = self.linear_threshold
    # Find the number of g-function values to be linearized
    p_long = np.searchsorted(self.time, time_threshold, side='right')
    if p_long > 0:
        time_long = np.concatenate([[time_threshold], self.time[p_long:]])
    else:
        time_long = self.time
    nt_long = len(time_long)
    # Initialize g-function
    gFunc = np.zeros(nt)
    # Initialize segment heat extraction rates
    if self.boundary_condition == 'UHTR':
        Q_b = 1
    else:
        Q_b = np.zeros((self.nSources, nt), dtype=self.dtype)
    if self.boundary_condition == 'UBWT':
        T_b = np.zeros(nt, dtype=self.dtype)
    else:
        T_b = np.zeros((self.nSources, nt), dtype=self.dtype)
    # Calculate segment to segment thermal response factors
    h_ij = self.thermal_response_factors(time_long, alpha, kind=self.kind)
    # Segment lengths
    H_b = self.segment_lengths()
    if self.boundary_condition == 'MIFT':
        Hb_individual = np.array([b.H for b in self.boreSegments], dtype=self.dtype)
    H_tot = np.sum(H_b)
    if self.disp: print('Building and solving the system of equations ...',
                        end='')
    # Initialize chrono
    tic = perf_counter()
    # Build and solve the system of equations at all times
    p0 = max(0, p_long - 1)
    for p in range(nt_long):
        if self.boundary_condition == 'UHTR':
            # Evaluate the g-function with uniform heat extraction along
            # boreholes
            # Thermal response factors evaluated at time t[p]
            h_dt = h_ij.y[:, :, p + 1]
            # Borehole wall temperatures are calculated by the sum of
            # contributions of all segments
            T_b[:, p + p0] = np.sum(h_dt, axis=1)
            # The g-function is the average of all borehole wall
            # temperatures
            gFunc[p + p0] = np.sum(T_b[:, p + p0] * H_b) / H_tot
        else:
            # Current thermal response factor matrix
            if p > 0:
                dt = time_long[p] - time_long[p - 1]
            else:
                dt = time_long[p]
            # Thermal response factors evaluated at t=dt
            h_dt = h_ij(dt)
            # Reconstructed load history
            Q_reconstructed = self.load_history_reconstruction(
                time_long[0:p + 1], Q_b[:, p0:p + p0 + 1])
            # Borehole wall temperature for zero heat extraction at
            # current step
            T_b0 = self.temporal_superposition(
                h_ij.y[:, :, 1:], Q_reconstructed)
            if self.boundary_condition == 'UBWT':
                # Evaluate the g-function with uniform borehole wall
                # temperature
                # ---------------------------------------------------------
                # Build a system of equation [A]*[X] = [B] for the
                # evaluation of the g-function. [A] is a coefficient
                # matrix, [X] = [Q_b,T_b] is a state space vector of the
                # borehole heat extraction rates and borehole wall
                # temperature (equal for all segments), [B] is a
                # coefficient vector.
                #
                # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                # Energy conservation: sum([Q_b*Hb]) = sum([Hb])
                # ---------------------------------------------------------
                A = np.block([[h_dt, -np.ones((self.nSources, 1),
                                              dtype=self.dtype)],
                              [H_b, 0.]])
                B = np.hstack((-T_b0, H_tot))
                # Solve the system of equations
                X = np.linalg.solve(A, B)
                # Store calculated heat extraction rates
                Q_b[:, p + p0] = X[0:self.nSources]
                # The borehole wall temperatures are equal for all segments
                T_b[p + p0] = X[-1]
                gFunc[p + p0] = T_b[p + p0]
            elif self.boundary_condition == 'MIFT':
                # Evaluate the g-function with mixed inlet fluid
                # temperatures
                # ---------------------------------------------------------
                # Build a system of equation [A]*[X] = [B] for the
                # evaluation of the g-function. [A] is a coefficient
                # matrix, [X] = [Q_b,T_b,Tf_in] is a state space vector of
                # the borehole heat extraction rates, borehole wall
                # temperatures and inlet fluid temperature (into the bore
                # field), [B] is a coefficient vector.
                #
                # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                # Heat transfer inside boreholes:
                # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
                # Energy conservation: sum([Q_b*H_b]) = sum([H_b])
                # ---------------------------------------------------------
                a_in, a_b = self.network.coefficients_borehole_heat_extraction_rate(
                    self.network.m_flow_network,
                    self.network.cp_f,
                    self.nBoreSegments,
                    segment_ratios=self.segment_ratios)
                k_s = self.network.p[0].k_s
                A = np.block(
                    [[h_dt,
                      -np.eye(self.nSources, dtype=self.dtype),
                      np.zeros((self.nSources, 1), dtype=self.dtype)],
                     [np.eye(self.nSources, dtype=self.dtype),
                      a_b / (2.0 * np.pi * k_s * np.atleast_2d(Hb_individual).T),
                      a_in / (2.0 * np.pi * k_s * np.atleast_2d(Hb_individual).T)],
                     [H_b, np.zeros(self.nSources + 1, dtype=self.dtype)]])
                B = np.hstack(
                    (-T_b0,
                     np.zeros(self.nSources, dtype=self.dtype),
                     H_tot))
                # Solve the system of equations
                X = np.linalg.solve(A, B)
                # Store calculated heat extraction rates
                Q_b[:, p + p0] = X[0:self.nSources]
                T_b[:, p + p0] = X[self.nSources:2 * self.nSources]
                T_f_in = X[-1]
                # The gFunction is equal to the effective borehole wall
                # temperature
                # Outlet fluid temperature
                T_f_out = T_f_in - 2 * np.pi * self.network.p[0].k_s * H_tot / (
                    np.sum(self.network.m_flow_network * self.network.cp_f))
                # Average fluid temperature
                T_f = 0.5 * (T_f_in + T_f_out)
                # Borefield thermal resistance
                R_field = network_thermal_resistance(
                    self.network, self.network.m_flow_network,
                    self.network.cp_f)
                # Effective borehole wall temperature
                T_b_eff = T_f - 2 * np.pi * self.network.p[0].k_s * R_field
                gFunc[p + p0] = T_b_eff
    # Linearize g-function for times under threshold
    if p_long > 0:
        gFunc[:p_long] = gFunc[p_long - 1] * self.time[:p_long] / time_threshold
        if not self.boundary_condition == 'UHTR':
            Q_b[:, :p_long] = 1 + (Q_b[:, p_long - 1:p_long] - 1) * self.time[:p_long] / time_threshold
        if self.boundary_condition == 'UBWT':
            T_b[:p_long] = T_b[p_long - 1] * self.time[:p_long] / time_threshold
        else:
            T_b[:, :p_long] = T_b[:, p_long - 1:p_long] * self.time[:p_long] / time_threshold
    # Store temperature and heat extraction rate profiles
    if self.profiles:
        self.Q_b = Q_b
        self.T_b = T_b
    toc = perf_counter()
    if self.disp: print(f' {toc - tic:.3f} sec')
    return gFunc


def __init__(self, boreholes, network, time, boundary_condition,
             m_flow_borehole=None, m_flow_network=None, cp_f=None,
             nSegments=8, segment_ratios=gt.utilities.segment_ratios,
             approximate_FLS=False, mQuad=11, nFLS=10,
             linear_threshold=None, disp=False, profiles=False,
             kind='linear', dtype=np.double, cylindrical_correction=False, **other_options):
    self.boreholes = boreholes
    self.network = network
    # Convert time to a 1d array
    self.time = np.atleast_1d(time).flatten()
    self.linear_threshold = linear_threshold
    self.cylindrical_correction = cylindrical_correction
    self.r_b_max = np.max([b.r_b for b in self.boreholes])
    self.boundary_condition = boundary_condition
    nBoreholes = len(self.boreholes)
    # Format number of segments and segment ratios
    if type(nSegments) is int:
        self.nBoreSegments = [nSegments] * nBoreholes
    else:
        self.nBoreSegments = nSegments
    if isinstance(segment_ratios, np.ndarray):
        segment_ratios = [segment_ratios] * nBoreholes
    elif segment_ratios is None:
        segment_ratios = [np.full(n, 1. / n) for n in self.nBoreSegments]
    elif callable(segment_ratios):
        segment_ratios = [segment_ratios(n) for n in self.nBoreSegments]
    self.segment_ratios = segment_ratios
    # Shortcut for segment_ratios comparisons
    self._equal_segment_ratios = \
        (np.all(np.array(self.nBoreSegments, dtype=np.uint) == self.nBoreSegments[0])
         and np.all([np.allclose(segment_ratios, self.segment_ratios[0]) for segment_ratios in self.segment_ratios]))
    # Boreholes with a uniform discretization
    self._uniform_segment_ratios = [
        np.allclose(segment_ratios,
                    segment_ratios[0:1],
                    rtol=1e-6)
        for segment_ratios in self.segment_ratios]
    # Find indices of first and last segments along boreholes
    self._i0Segments = [sum(self.nBoreSegments[0:i])
                        for i in range(nBoreholes)]
    self._i1Segments = [sum(self.nBoreSegments[0:(i + 1)])
                        for i in range(nBoreholes)]
    self.nMassFlow = 0
    self.m_flow_borehole = m_flow_borehole
    if self.m_flow_borehole is not None:
        if not self.m_flow_borehole.ndim == 1:
            self.nMassFlow = np.size(self.m_flow_borehole, axis=0)
        self.m_flow_borehole = np.atleast_2d(self.m_flow_borehole)
        self.m_flow = self.m_flow_borehole
    self.m_flow_network = m_flow_network
    if self.m_flow_network is not None:
        if not isinstance(self.m_flow_network, (np.floating, float)):
            self.nMassFlow = len(self.m_flow_network)
        self.m_flow_network = np.atleast_1d(self.m_flow_network)
        self.m_flow = self.m_flow_network
    self.cp_f = cp_f
    self.approximate_FLS = approximate_FLS
    self.mQuad = mQuad
    self.nFLS = nFLS
    self.disp = disp
    self.profiles = profiles
    self.kind = kind
    self.dtype = dtype
    # Check the validity of inputs
    self._check_inputs()
    # Initialize the solver with solver-specific options
    self.nSources = self.initialize(**other_options)

    return


def update_pygfunction() -> None:
    """
    This function updates pygfunction by adding the cylindrical correction methods to it.

    Returns
    -------
    None
    """
    gt.heat_transfer.cylindrical_heat_source = cylindrical_heat_source
    gt.heat_transfer.infinite_line_source = infinite_line_source
    gt.solvers.Equivalent.thermal_response_factors = thermal_response_factors
    gt.solvers._BaseSolver.solve = solve
    gt.solvers._BaseSolver.__init__ = __init__
