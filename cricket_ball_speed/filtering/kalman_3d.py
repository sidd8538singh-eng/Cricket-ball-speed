r"""
3D World-Space Kalman Filter.
State vector: [X, Y, Z, VX, VY, VZ]^T (positions in metres, velocities in m/s).

Mathematical Model:
-------------------
Continuous-time ballistic process under gravity:
    \dot{X} = VX
    \dot{Y} = VY
    \dot{Z} = VZ - g
    \dot{VX} = w_x
    \dot{VY} = w_y
    \dot{VZ} = w_z
    where g = 9.80665 m/s^2 acts downward along -Z,
    and w ~ N(0, q_std^2 * I_3) represents unmodeled aerodynamic perturbations.

Discrete-time state transition for nonuniform intervals Delta t = t_k - t_{k-1}:
    F(Delta t) = [[1, 0, 0, Delta t, 0,       0      ],
                  [0, 1, 0, 0,       Delta t, 0      ],
                  [0, 0, 1, 0,       0,       Delta t],
                  [0, 0, 0, 1,       0,       0      ],
                  [0, 0, 0, 0,       1,       0      ],
                  [0, 0, 0, 0,       0,       1      ]]

Control vector for gravity:
    u(Delta t) = [0, 0, -0.5 * g * Delta t^2, 0, 0, -g * Delta t]^T

Discrete process noise covariance Q(Delta t):
    Block diagonal per axis (X, Y, Z):
    Q_i = q_std^2 * [[Delta t^4 / 4, Delta t^3 / 2],
                     [Delta t^3 / 2, Delta t^2    ]]

Measurement model:
    z_k = H x_k + v_k, where H = [I_3 | 0_3x3]
    R_k = diag([R_X, R_Y, R_Z]) reflects 3D reconstruction ray intersection uncertainty.

Speed uncertainty propagation:
    speed = sqrt(VX^2 + VY^2 + VZ^2)
    sigma_speed = sqrt( (grad_v)^T * P_vv * (grad_v) )
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np


class Kalman3D:
    """
    3D Kalman filter with physical ballistic model incorporating gravity and aerodynamic drag covariance.
    Tracks ball 3D position, 3D velocity vector, speed in km/h, and associated uncertainties.
    """

    def __init__(
        self,
        initial_xyz: np.ndarray,
        initial_vxyz: Optional[np.ndarray] = None,
        process_noise_std: float = 1.0,
        gravity_m_s2: float = 9.80665,
        initial_pos_std: float = 0.05,
        initial_vel_std: float = 2.0
    ):
        """
        Args:
            initial_xyz: Initial 3D position [X, Y, Z] in metres.
            initial_vxyz: Initial 3D velocity [VX, VY, VZ] in m/s (optional, defaults to zero).
            process_noise_std: Aerodynamic acceleration noise in m/s^2.
            gravity_m_s2: Gravitational acceleration (default 9.80665 m/s^2).
        """
        init_pos = np.asarray(initial_xyz, dtype=np.float64).reshape(3)
        init_vel = np.asarray(initial_vxyz, dtype=np.float64).reshape(3) if initial_vxyz is not None else np.zeros(3)

        # State: [X, Y, Z, VX, VY, VZ]^T
        self.x = np.concatenate([init_pos, init_vel]).astype(np.float64)

        # Initial covariance matrix P
        p_pos = float(initial_pos_std ** 2)
        p_vel = float(initial_vel_std ** 2)
        self.P = np.diag([p_pos, p_pos, p_pos, p_vel, p_vel, p_vel]).astype(np.float64)

        self.q_std = float(process_noise_std)
        self.g = float(gravity_m_s2)

        # 3D Measurement matrix H (observing X, Y, Z)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Diagnostics cache
        self.last_dt: float = 0.0
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None
        self.last_kalman_gain: Optional[np.ndarray] = None
        self.last_mahalanobis: float = 0.0

    def predict(self, dt: float) -> np.ndarray:
        """
        Advances the 3D ballistic state incorporating gravity downward in Z.
        Handles nonuniform dt > 0.
        """
        dt = max(1e-4, float(dt))
        self.last_dt = dt

        # State transition matrix F
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Gravity control vector
        control = np.zeros(6, dtype=np.float64)
        control[2] = -0.5 * self.g * (dt ** 2)
        control[5] = -self.g * dt

        # Continuous White Noise Acceleration Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q_var = self.q_std ** 2

        Q_block = q_var * np.array([
            [dt4, dt3],
            [dt3, dt2]
        ], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[np.ix_([i, i + 3], [i, i + 3])] = Q_block

        # Extrapolate state and covariance
        self.x = F @ self.x + control
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def get_measurement_covariance(self, measurement_cov_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """Constructs 3x3 measurement covariance matrix R."""
        if measurement_cov_diag is not None:
            diag_vals = np.asarray(measurement_cov_diag, dtype=np.float64).reshape(3)
            return np.diag(np.maximum(1e-6, diag_vals)).astype(np.float64)
        # Default baseline anisotropic noise: depth (Y) typically has higher reconstruction uncertainty
        return np.diag([0.02, 0.08, 0.02]).astype(np.float64)

    def compute_mahalanobis(
        self,
        observed_xyz: np.ndarray,
        measurement_cov_diag: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes 3D Mahalanobis distance without modifying state.
        Returns: (mahalanobis_distance, innovation, innovation_covariance_S)
        """
        z = np.asarray(observed_xyz, dtype=np.float64).reshape(3)
        R = self.get_measurement_covariance(measurement_cov_diag)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        try:
            S_inv = np.linalg.inv(S)
            d_sq = float(y.T @ S_inv @ y)
            d_m = float(np.sqrt(max(0.0, d_sq)))
        except np.linalg.LinAlgError:
            d_m = float(np.linalg.norm(y) / 0.1)

        return d_m, y, S

    def is_within_gate(
        self,
        observed_xyz: np.ndarray,
        measurement_cov_diag: Optional[np.ndarray] = None,
        gate_threshold: float = 3.8
    ) -> Tuple[bool, float, float]:
        """
        Evaluates whether 3D observation falls within the ellipsoidal gate.
        Returns: (is_accepted, mahalanobis_distance, gate_threshold)
        """
        d_m, _, _ = self.compute_mahalanobis(observed_xyz, measurement_cov_diag)
        return (d_m <= gate_threshold), d_m, gate_threshold

    def update(
        self,
        observed_xyz: np.ndarray,
        measurement_cov_diag: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Applies 3D measurement update.
        Returns: (updated_state, mahalanobis_distance)
        """
        z = np.asarray(observed_xyz, dtype=np.float64).reshape(3)
        R = self.get_measurement_covariance(measurement_cov_diag)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        S_inv = np.linalg.inv(S)

        d_sq = float(y.T @ S_inv @ y)
        d_m = float(np.sqrt(max(0.0, d_sq)))

        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y

        # Joseph-form covariance update
        I = np.eye(6, dtype=np.float64)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Cache diagnostics
        self.last_innovation = y.copy()
        self.last_innovation_cov = S.copy()
        self.last_kalman_gain = K.copy()
        self.last_mahalanobis = d_m

        return self.x.copy(), d_m

    @property
    def position_xyz(self) -> np.ndarray:
        """Estimated [X, Y, Z] in metres."""
        return self.x[:3].copy()

    @property
    def velocity_xyz(self) -> np.ndarray:
        """Estimated [VX, VY, VZ] in m/s."""
        return self.x[3:].copy()

    @property
    def speed_m_s(self) -> float:
        """Scalar magnitude of 3D velocity in m/s."""
        return float(np.linalg.norm(self.x[3:]))

    @property
    def speed_km_h(self) -> float:
        """Scalar speed in km/h."""
        return float(self.speed_m_s * 3.6)

    @property
    def covariance(self) -> np.ndarray:
        """Full 6x6 state covariance matrix P."""
        return self.P.copy()

    def get_position_uncertainty(self) -> Tuple[float, float, float]:
        """Standard deviations of position (sigma_X, sigma_Y, sigma_Z) in metres."""
        return (
            float(np.sqrt(max(0.0, self.P[0, 0]))),
            float(np.sqrt(max(0.0, self.P[1, 1]))),
            float(np.sqrt(max(0.0, self.P[2, 2])))
        )

    def get_velocity_uncertainty(self) -> Tuple[float, float, float]:
        """Standard deviations of velocity (sigma_VX, sigma_VY, sigma_VZ) in m/s."""
        return (
            float(np.sqrt(max(0.0, self.P[3, 3]))),
            float(np.sqrt(max(0.0, self.P[4, 4]))),
            float(np.sqrt(max(0.0, self.P[5, 5])))
        )

    def get_speed_uncertainty(self) -> float:
        """
        Computes 1-sigma speed uncertainty in km/h via linear error propagation:
            sigma_speed = sqrt( (grad_v)^T * P_vv * (grad_v) ) * 3.6
        """
        v = self.x[3:]
        speed = float(np.linalg.norm(v))
        if speed < 1e-4:
            return float(np.sqrt(np.mean([self.P[3, 3], self.P[4, 4], self.P[5, 5]])) * 3.6)

        grad = v / speed  # 3x1 unit direction vector
        P_vv = self.P[3:6, 3:6]
        var_speed_m_s = float(grad.T @ P_vv @ grad)
        sigma_speed_km_h = float(np.sqrt(max(0.0, var_speed_m_s)) * 3.6)
        return sigma_speed_km_h

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns structured diagnostic dictionary."""
        sig_x, sig_y, sig_z = self.get_position_uncertainty()
        sig_vx, sig_vy, sig_vz = self.get_velocity_uncertainty()
        return {
            "X_m": round(float(self.x[0]), 3),
            "Y_m": round(float(self.x[1]), 3),
            "Z_m": round(float(self.x[2]), 3),
            "VX_m_s": round(float(self.x[3]), 2),
            "VY_m_s": round(float(self.x[4]), 2),
            "VZ_m_s": round(float(self.x[5]), 2),
            "speed_km_h": round(self.speed_km_h, 2),
            "speed_uncertainty_km_h": round(self.get_speed_uncertainty(), 2),
            "sigma_X_m": round(sig_x, 4),
            "sigma_Y_m": round(sig_y, 4),
            "sigma_Z_m": round(sig_z, 4),
            "last_mahalanobis": round(self.last_mahalanobis, 3),
            "last_dt_sec": round(self.last_dt, 5)
        }
