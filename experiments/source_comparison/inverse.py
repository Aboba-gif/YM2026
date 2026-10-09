"""Матрицы регуляризации и численная проверка условий ККТ."""
import numpy as np

from adrkit.inverse.regularization import p1_matrices
from .arrays import finite_real_array


def penalty_grams(basis, q_reference, *, tau_hours):
    r"""Построить матрицы штрафов L² и H¹ для профиля P1.

    Parameters
    ----------
    basis : P1Basis
        Кусочно-линейный базис с временными узлами в часах.
    q_reference : float
        Положительный масштаб интенсивности в C·км²/ч; коэффициенты профиля
        безразмерны.
    tau_hours : float
        Положительный временной масштаб производной в H¹, в часах.

    Returns
    -------
    grams : dict of str to ndarray
        Матрицы L2 и H1 формы (basis.size, basis.size), в C²·км⁴/ч; C —
        единица концентрации.

    Notes
    -----
    Пусть :math:`M` — матрица интегралов произведений функций P1,
    :math:`K` — матрица интегралов произведений их производных по времени.
    При :math:`Q_{\rm ref}=` q_reference и :math:`\tau=` tau_hours:

    .. math::

        G_{L^2}=Q_{\rm ref}^2 M, \qquad
        G_{H^1}=Q_{\rm ref}^2(M+\tau^2 K).

    H¹ включает как профиль, так и его производную.
    """
    if basis.time_unit != "h" or not np.isfinite(q_reference) or q_reference <= 0:
        raise ValueError("hour-domain P1 and positive Qref required")
    if not np.isfinite(tau_hours) or tau_hours <= 0:
        raise ValueError("positive temporal H1 scale required")
    mass, stiffness = p1_matrices(basis)
    return {"L2": q_reference**2*mass,
            "H1": q_reference**2*(mass+tau_hours**2*stiffness)}


def kkt_certificate(a, data_gradient, penalty_gradient, *, data_term, penalty_term,
                    optimizer_success, forward_residual, forward_tolerance):
    """Вычислить численные показатели условий ККТ для источника.

    Parameters
    ----------
    a : array_like, shape (n_coefficients,)
        Оценка безразмерных узловых коэффициентов с ограничением
        неотрицательности.
    data_gradient, penalty_gradient : array_like, shape (n_coefficients,)
        Ковекторы невязки и штрафа в координатах коэффициентов.
    data_term, penalty_term : float
        Значения невязки и штрафа при той же оценке.
    optimizer_success : bool
        Признак успешного завершения оптимизатора.
    forward_residual : float
        Максимальная масштабированная невязка прямого решения.
    forward_tolerance : float
        Допуск для этой невязки.

    Returns
    -------
    certificate : dict
        Масштабированные показатели допустимости, стационарности и
        комплементарности, множители ограничений и численный признак
        accepted.

    Notes
    -----
    Проверка ККТ с численными допусками не удостоверяет глобальный минимум
    полулинейной обратной задачи.
    """
    a = finite_real_array(a, "coefficients")
    gd = finite_real_array(data_gradient, "data covector")
    gp = finite_real_array(penalty_gradient, "penalty covector")
    if a.ndim != 1 or len(a) == 0 or gd.shape != a.shape or gp.shape != a.shape:
        raise ValueError("matching nonempty coefficient/covector vectors required")
    g = gd+gp
    multipliers = np.where(a <= 1e-10, np.maximum(g, 0.), 0.)
    inf = lambda x: float(np.max(np.abs(x), initial=0.))
    sg = max(1., inf(gd), inf(gp), inf(multipliers))
    sj = max(1., abs(data_term)+abs(penalty_term))
    norms = dict(primal=inf(np.minimum(a, 0.)),
                 dual=inf(np.minimum(multipliers, 0.))/sg,
                 stationarity=inf(g-multipliers)/sg,
                 complementarity=inf(a*multipliers)/sj,
                 free_coordinate=inf(g[a > 1e-10])/sg)
    finite = bool(np.isfinite([data_term, penalty_term, forward_residual,
                               sg, sj, *norms.values()]).all())
    accepted = bool(finite and optimizer_success and 0 <= forward_residual <= forward_tolerance
                    and norms["primal"] <= 1e-8 and norms["dual"] <= 1e-8
                    and all(norms[k] <= 1e-6 for k in
                            ("stationarity", "complementarity", "free_coordinate")))
    return dict(accepted=accepted, finite=finite, a=a.tolist(), gD=gd.tolist(),
                gP=gp.tolist(), multipliers=multipliers.tolist(), sg=sg, sJ=sj,
                norms=norms, forward_residual=float(forward_residual),
                forward_tolerance=float(forward_tolerance), optimizer_success=bool(optimizer_success))
