"""
This file contains a hybrid dispatch optimisation formulated as a linear program.

For a fixed borefield, the fluid temperature is a linear function of the dispatched
ground load (a convolution with the g-function increments). The question 'which part
of the building load should the borefield serve, hour by hour' therefore has a
polyhedral feasible set, and the optimal hybrid dispatch is a linear program that can
be solved globally and all at once, instead of with a load-duration-curve clipping
iteration:

    max   total energy served by the borefield        (objective='energy')
    min   external (backup) peak capacity needed      (objective='power')
    s.t.  Tf_min <= Tf(t) <= Tf_max     for every hour of every simulated year
          0 <= served(t) <= demand(t)   for every hour

Only a handful of the temperature constraints bind at the optimum (the hours in which
the ground is exhausted), so the LP is solved with constraint-row generation: solve,
find the violated hours with an exact temperature evaluation, add those rows, repeat.
The multi-year temperature response of the periodic load is evaluated exactly with a
year-folded convolution kernel.

The 'power' objective is solved lexicographically: first the minimal backup capacity
is found, then, with that capacity fixed, the served energy is maximised, so the
returned dispatch is not needlessly conservative.

The returned solution is certified: the temperatures of the resulting borefield load
are recalculated with the regular hourly temperature calculation of GHEtool and
checked against the temperature limits.

The dual variables (shadow prices) of the binding temperature constraints are
returned as well: they quantify the marginal value of relaxing the temperature band,
and, through the 1/length scaling of the temperature response, the marginal value of
additional borehole length. This is the coupling quantity for a joint
configuration-and-dispatch (two-stage) design optimisation.

The LP is solved through an incremental *sparse* model: capacity rows (two
nonzeros each) enter up front, temperature rows are constraint-generated, and,
when the optional ``highspy`` package is installed, every re-solve (including
the lexicographic second phase) hot-starts from the previous HiGHS basis
instead of re-factorising a dense matrix from scratch.
"""
import copy

import numpy as np

from scipy import sparse
from scipy.optimize import linprog
from scipy.signal import fftconvolve

from GHEtool.VariableClasses import SCOP, SEER
from GHEtool.VariableClasses.LoadData import HourlyBuildingLoad

try:  # optional accelerator: incremental rows + hot-started re-solves
    import highspy
    _HAS_HIGHSPY = True
except ImportError:  # pragma: no cover - highspy is an optional extra
    _HAS_HIGHSPY = False

__all__ = ['optimise_load_profile_lp']


class _IncrementalSparseLP:
    """Incremental sparse LP: min c @ x  s.t.  A_ub @ x <= b_ub, lb <= x <= ub.

    The constraint matrix is stored sparse: a capacity row has two nonzeros and
    a temperature row at most 2 x 8760, so the LP that used to be handed to the
    solver as a ~9000 x 17522 *dense* array (99.97 % zeros) shrinks to a few
    hundred thousand nonzeros.

    With ``highspy`` installed the model lives inside a single HiGHS instance:
    rows are added incrementally between solves and every re-solve hot-starts
    from the previous basis, so each constraint-generation round only pays for
    the marginal work. Without it, ``scipy.optimize.linprog`` is called on the
    sparse matrix each round - identical solution, just cold-started.

    Duals returned by :meth:`solve` follow scipy's ``ineqlin.marginals``
    convention: non-positive for a binding ``<=`` row of a minimisation
    (dObjective / dRHS).
    """

    def __init__(self, c, lb, ub, time_limit=None):
        self.n = len(c)
        self.n_rows = 0
        self.c = np.asarray(c, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.time_limit = time_limit
        self._rows_idx: list = []
        self._rows_val: list = []
        self._rhs: list = []
        self._h = None
        if _HAS_HIGHSPY:
            self._h = highspy.Highs()
            self._h.setOptionValue('output_flag', False)
            if time_limit is not None:
                self._h.setOptionValue('time_limit', float(time_limit))
            upper = np.where(np.isinf(self.ub), highspy.kHighsInf, self.ub)
            self._h.addCols(self.n, self.c, self.lb, upper,
                            0, np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                            np.array([], dtype=np.float64))

    def add_rows(self, rows_idx, rows_val, rhs):
        """Add ``<=`` rows given as (column-index array, value array) pairs."""
        if not len(rhs):
            return
        self.n_rows += len(rhs)
        if self._h is not None:
            # HiGHS owns the rows: keeping a Python mirror as well doubles the
            # memory of an already large model for nothing.
            lengths = np.array([len(i) for i in rows_idx], dtype=np.int32)
            starts = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int32)
            indices = np.concatenate(rows_idx).astype(np.int32)
            values = np.concatenate(rows_val).astype(np.float64)
            self._h.addRows(len(rhs), np.full(len(rhs), -highspy.kHighsInf),
                            np.asarray(rhs, dtype=np.float64),
                            len(indices), starts, indices, values)
            return
        self._rows_idx.extend(rows_idx)
        self._rows_val.extend(rows_val)
        self._rhs.extend(float(r) for r in rhs)

    def set_costs(self, c):
        self.c = np.asarray(c, dtype=np.float64)
        if self._h is not None:
            self._h.changeColsCost(self.n, np.arange(self.n, dtype=np.int32), self.c)

    def set_bounds(self, j, lower, upper):
        self.lb[j], self.ub[j] = lower, upper
        if self._h is not None:
            self._h.changeColBounds(j, lower,
                                    upper if np.isfinite(upper) else highspy.kHighsInf)

    def solve(self):
        """Solve and return ``(x, row_duals)``; raises ValueError when not optimal.

        A configured ``time_limit`` makes a pathological solve (heavily
        degenerate deep-undersizing cases) fail fast with ValueError instead of
        hanging the worker; the caller treats that candidate as infeasible.
        """
        if self._h is not None:
            self._h.run()
            status = self._h.getModelStatus()
            if status != highspy.HighsModelStatus.kOptimal:
                raise ValueError('The LP dispatch optimisation failed: '
                                 f'{self._h.modelStatusToString(status)}')
            solution = self._h.getSolution()
            duals = np.asarray(solution.row_dual) if self.n_rows else np.array([])
            return np.asarray(solution.col_value), duals
        a_ub = None
        if self._rhs:
            indptr = np.concatenate(([0], np.cumsum([len(i) for i in self._rows_idx])))
            a_ub = sparse.csr_matrix(
                (np.concatenate(self._rows_val), np.concatenate(self._rows_idx), indptr),
                shape=(len(self._rhs), self.n))
        options = {'time_limit': float(self.time_limit)} if self.time_limit is not None else None
        res = linprog(self.c, A_ub=a_ub, b_ub=np.asarray(self._rhs) if self._rhs else None,
                      bounds=np.column_stack((self.lb, self.ub)), method='highs',
                      options=options)
        if not res.success:
            raise ValueError(f'The LP dispatch optimisation failed: {res.message}')
        duals = res.ineqlin.marginals if self._rhs else np.array([])
        return res.x, duals


def optimise_load_profile_lp(
        borefield,
        building_load: HourlyBuildingLoad,
        objective: str = 'energy',
        temperature_threshold: float = 0.025,
        max_lp_rounds: int = 40,
        return_shadow_prices: bool = False,
        time_limit: float = 600.):
    """
    This function optimises the hybrid dispatch (the split of the building load between
    the borefield and an external system) as a single linear program, which is globally
    optimal for the chosen objective. See the module documentation for the formulation.

    Parameters
    ----------
    borefield : Borefield
        Borefield object (with ground data and temperature limits set).
    building_load : HourlyBuildingLoad
        Building load to be split between the borefield and the external system.
        Constant efficiencies (SCOP/SEER) are required and the load should start in
        January without a DHW profile.
    objective : str
        'energy' to maximise the total energy served by the borefield, 'power' to
        minimise the external (backup) peak capacity (lexicographically followed by
        an energy maximisation at that capacity).
    temperature_threshold : float
        Maximum allowed violation of the temperature limits in the certification of
        the result [K].
    max_lp_rounds : int
        Maximum number of constraint-generation rounds.
    return_shadow_prices : bool
        True if a dictionary with the shadow prices of the binding temperature
        constraints (and the derived marginal value of borehole length) should be
        returned as a third element.
    time_limit : float
        Wall-clock budget for the LP solver [s]. Heavily undersized fields can
        make the solve pathologically degenerate; on exceeding the budget a
        ValueError is raised instead of hanging the process.

    Returns
    -------
    tuple(HourlyBuildingLoad, HourlyBuildingLoad) or tuple(..., ..., dict)
        Borefield load, external load (and optionally the shadow price information).

    Raises
    ------
    ValueError
        When the load is not an HourlyBuildingLoad with constant efficiencies starting
        in January, or when the certification of the result fails.
    """
    if not isinstance(building_load, HourlyBuildingLoad):
        raise ValueError('The LP dispatch optimisation requires an HourlyBuildingLoad.')
    if not (isinstance(building_load.cop, SCOP) and isinstance(building_load.eer, SEER)):
        raise ValueError('The LP dispatch optimisation requires constant efficiencies (SCOP/SEER). '
                         'For temperature-dependent efficiencies, please use the optimise_load_profile_power '
                         'or _energy methods, or iterate this method with updated efficiencies.')
    if building_load.start_month != 1:
        raise ValueError('The LP dispatch optimisation requires a load starting in January.')
    if building_load._hourly_dhw_load is not None and np.any(building_load._hourly_dhw_load):
        raise ValueError('The LP dispatch optimisation does not support DHW profiles.')
    if objective not in ('energy', 'power'):
        raise ValueError("The objective should be either 'energy' or 'power'.")

    borefield = copy.deepcopy(borefield)
    borefield.load = copy.deepcopy(building_load)

    P = 8760
    years = building_load.simulation_period
    dem_h = building_load.hourly_heating_load.copy()
    dem_c = building_load.hourly_cooling_load.copy()

    # building -> ground conversion factors (constant, since SCOP/SEER)
    f_h = 1. - 1. / building_load.cop.get_COP(0)
    f_c = 1. + 1. / building_load.eer.get_EER(0)

    # exact temperature response of the borefield
    H = borefield.H
    depth = borefield.calculate_depth(H, borefield.D)
    L_tot = borefield.number_of_boreholes * H
    k_s = borefield.ground_data.k_s(depth, borefield.D)
    Tg = borefield._Tg(H)
    Tf_min, Tf_max = borefield.Tf_min, borefield.Tf_max
    Rb = borefield.Rb

    g = borefield.gfunction(building_load.time_L4, H)
    dg = np.diff(g, prepend=0)
    scale = 1000. / (2 * np.pi * k_s * L_tot)   # K per kW of load in the convolution
    rb_term = 1000. * Rb / L_tot                # K per kW of load at the same hour

    # Year-folded convolution kernels: K_j gives the exact temperature in year j
    # of a periodic load (j = 1 is the first year, j = years the steady drift of
    # the last year). Constraints are generated for *every* year: when the
    # dispatch balances the field, the first-order year-over-year drift
    # vanishes and the seasonal second-order term can make an intermediate year
    # (typically year 2) the binding one, which first/last-year-only
    # constraints would miss.
    _kernels: dict = {}

    def year_kernel(j):
        """Folded kernel of year ``j`` (1-based), on a 2P-1 support."""
        if j not in _kernels:
            K = np.zeros(2 * P - 1)
            for i in range(j):
                lo, hi = i * P - (P - 1), i * P + P
                src = dg[max(lo, 0):min(hi, years * P)]
                K[max(lo, 0) - lo: max(lo, 0) - lo + len(src)] += src
            _kernels[j] = K
        return _kernels[j]

    def temperatures_all_years(q):
        """Fluid temperature at every hour of the whole horizon for periodic q [kW]."""
        q_full = np.tile(q, years)
        return Tg + scale * fftconvolve(q_full, dg)[:years * P] + rb_term * q_full

    def temperature_row(tau, year):
        """row a such that Tf(tau of year ``year``) = Tg + a . q"""
        a = scale * year_kernel(year)[tau - np.arange(P) + P - 1]
        a[tau] += rb_term
        return a

    # ------------------------------------------------------------------
    # LP with constraint-row generation
    # variables: x = [served heating (P), served cooling (P), (Cap_h, Cap_c)]
    # ------------------------------------------------------------------
    n_extra = 2 if objective == 'power' else 0
    nvar = 2 * P + n_extra
    lb = np.zeros(nvar)
    ub = np.concatenate((dem_h, dem_c, np.full(n_extra, np.inf)))
    if objective == 'energy':
        c = np.concatenate((-np.ones(2 * P), np.zeros(n_extra)))
    else:
        c = np.zeros(nvar)
        c[-2:] = 1.

    lp = _IncrementalSparseLP(c, lb, ub, time_limit=time_limit)
    row_info = []
    added_T = set()

    if objective == 'power':
        # All capacity rows (dem - x <= Cap  ->  -x - Cap <= -dem) are known a
        # priori and have two nonzeros each: add them up front in one batch
        # instead of discovering thousands of them one round at a time.
        rows_idx, rows_val, rhs = [], [], []
        for u in np.flatnonzero(dem_h > 0):
            rows_idx.append(np.array([u, nvar - 2]))
            rows_val.append(np.array([-1., -1.]))
            rhs.append(-dem_h[u])
            row_info.append(None)
        for u in np.flatnonzero(dem_c > 0):
            rows_idx.append(np.array([P + u, nvar - 1]))
            rows_val.append(np.array([-1., -1.]))
            rhs.append(-dem_c[u])
            row_info.append(None)
        lp.add_rows(rows_idx, rows_val, rhs)

    def generate_temperature_rows(x):
        """Add the temperature rows violated by dispatch ``x``; return the count.

        The exact temperature of every hour of every year is scanned, so the
        generated constraint set covers intermediate years as well; at
        convergence the LP dispatch respects the band over the whole horizon.
        """
        q = f_c * x[P:2 * P] - f_h * x[:P]
        T = temperatures_all_years(q).reshape(years, P)
        rows_idx, rows_val, rhs = [], [], []
        for sign in (1., -1.):
            v = sign * T - (Tf_max if sign > 0 else -Tf_min)
            # Per hour, only the worst year can bind the pointwise maximum; the
            # per-year kernels converge geometrically in the year index, so
            # adding every violated year would flood the LP with near-duplicate
            # rows. If a different year later becomes the worst for an hour, it
            # is added in a subsequent round.
            worst_year = np.argmax(v, axis=0)
            worst_v = v[worst_year, np.arange(P)]
            for tau in np.argsort(worst_v)[-160:]:
                if worst_v[tau] > 1e-7:
                    tau, year = int(tau), int(worst_year[tau]) + 1
                    if (tau, year, sign) in added_T:
                        continue
                    a = temperature_row(tau, year)
                    coeff = np.concatenate((sign * (-f_h) * a, sign * f_c * a))
                    nz = np.flatnonzero(coeff)
                    rows_idx.append(nz)
                    rows_val.append(coeff[nz])
                    rhs.append((Tf_max - Tg) if sign > 0 else (Tg - Tf_min))
                    row_info.append((tau, year, sign))
                    added_T.add((tau, year, sign))
        lp.add_rows(rows_idx, rows_val, rhs)
        return len(rhs)

    x, duals = None, np.array([])
    for _ in range(max_lp_rounds):
        x, duals = lp.solve()
        if generate_temperature_rows(x) == 0:
            break
    else:
        raise ValueError('The LP dispatch optimisation did not converge within the allowed number of rounds.')

    if objective == 'power':
        # lexicographic refinement: fix the optimal capacities, maximise the
        # served energy. The model (and its basis, with highspy) is reused: only
        # the objective and the two capacity bounds change.
        cap_h, cap_c = x[-2], x[-1]
        lp.set_costs(np.concatenate((-np.ones(2 * P), np.zeros(2))))
        lp.set_bounds(nvar - 2, 0., cap_h + 1e-9)
        lp.set_bounds(nvar - 1, 0., cap_c + 1e-9)
        try:
            for _ in range(max_lp_rounds):
                x, duals = lp.solve()
                if generate_temperature_rows(x) == 0:
                    break
        except ValueError:  # pragma: no cover - keep the phase-1 dispatch
            pass

    served_h, served_c = x[:P], x[P:2 * P]

    # ------------------------------------------------------------------
    # certification with the regular GHEtool temperature calculation
    # ------------------------------------------------------------------
    borefield_load = copy.deepcopy(building_load)
    borefield_load.hourly_heating_load = served_h
    borefield_load.hourly_cooling_load = served_c
    external_load = copy.deepcopy(building_load)
    external_load.hourly_heating_load = np.maximum(dem_h - served_h, 0.)
    external_load.hourly_cooling_load = np.maximum(dem_c - served_c, 0.)

    borefield.load = copy.deepcopy(borefield_load)
    borefield.calculate_temperatures(hourly=True)
    T_max_cert = float(np.max(borefield.results.peak_injection))
    T_min_cert = float(np.min(borefield.results.peak_injection))
    if T_max_cert > Tf_max + temperature_threshold or T_min_cert < Tf_min - temperature_threshold:
        raise ValueError(  # pragma: no cover
            f'The certification of the LP dispatch failed: the fluid temperature spans '
            f'[{T_min_cert:.3f}, {T_max_cert:.3f}] degC for limits [{Tf_min}, {Tf_max}] degC.')

    if not return_shadow_prices:
        return borefield_load, external_load

    # shadow prices of the binding temperature constraints; the marginal objective
    # value of extra borehole length follows from the 1/length scaling of the
    # temperature response: dTf/dL = -(Tf - Tg)/L at the binding hours
    marginals = duals if row_info else np.array([])
    shadow = []
    marginal_value_length = 0.
    # row_info can outrun the duals: constraint generation appends rows after the
    # last successful solve when a re-solve fails (phase 2 swallows that failure
    # and keeps the previous dispatch). Rows without a dual carry no shadow price.
    for i, info in enumerate(row_info[:len(marginals)]):
        if info is None or abs(marginals[i]) < 1e-12:
            continue
        tau, year, sign = info
        lam = -marginals[i]  # positive shadow price of tightening the constraint
        limit = Tf_max if sign > 0 else Tf_min
        shadow.append({'hour': tau, 'year': year,
                       'limit': limit, 'shadow_price': lam})
        marginal_value_length += lam * abs(limit - Tg) / L_tot
    info = {'temperature_constraints': shadow,
            'marginal_objective_value_per_meter': marginal_value_length,
            'certified_temperature_range': (T_min_cert, T_max_cert)}
    return borefield_load, external_load, info
