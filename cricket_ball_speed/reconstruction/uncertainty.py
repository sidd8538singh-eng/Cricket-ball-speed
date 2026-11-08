"""
Uncertainty Estimator: Computes covariance matrices and confidence bounds.
Propagates measurement variances into velocity, position, and final speed uncertainties.
"""

from typing import Dict, Any, Tuple
import numpy as np


class UncertaintyEstimator:
    """
    Computes formal covariance and confidence intervals from non-linear least squares Jacobians.
    """

    @staticmethod
    def compute_parameter_covariance(
        jacobian: np.ndarray, residuals: np.ndarray
    ) -> np.ndarray:
        """
        Computes parameter covariance matrix: Sigma = (J^T J)^-1 * s2
        where s2 is the residual variance.
        """
        n_obs = len(residuals)
        n_params = jacobian.shape[1]
        dof = max(1, n_obs - n_params)
        s2 = float(np.sum(residuals ** 2) / dof)

        try:
            jt_j = jacobian.T @ jacobian
            cov = np.linalg.inv(jt_j) * s2
        except np.linalg.LinAlgError:
            cov = np.eye(n_params) * 1.0

        return cov

    @staticmethod
    def compute_speed_uncertainty(
        vx: float, vy: float, vz: float,
        sigma_vx: float, sigma_vy: float, sigma_vz: float
    ) -> float:
        """
        Propagates velocity standard errors into speed uncertainty in km/h:
        Speed = 3.6 * sqrt(vx^2 + vy^2 + vz^2)
        dSpeed/dv_i = 3.6 * v_i / Speed
        sigma_Speed = 3.6 * sqrt((vx*s_vx)^2 + (vy*s_vy)^2 + (vz*s_vz)^2) / Speed_m_s
        """
        speed_m_s = np.hypot(vx, np.hypot(vy, vz))
        if speed_m_s < 1e-3:
            return 10.0

        num = (vx * sigma_vx) ** 2 + (vy * sigma_vy) ** 2 + (vz * sigma_vz) ** 2
        sigma_speed_m_s = np.sqrt(num) / speed_m_s
        sigma_speed_kmh = float(sigma_speed_m_s * 3.6)
        return sigma_speed_kmh
