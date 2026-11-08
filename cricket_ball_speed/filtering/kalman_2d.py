r"""
2D Image-Space Kalman Filter.
State vector: [u, v, Vu, Vv]^T (pixel position and pixel velocities in px/sec).

Mathematical Model:
-------------------
Continuous-time kinematic process:
    \dot{x}(t) = A x(t) + G w(t)
    where A = [[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]],
          G = [[0, 0],
               [0, 0],
               [1, 0],
               [0, 1]]
    and w(t) ~ N(0, q_std^2 * I_2) represents Continuous White Noise Acceleration (CWNA).

Discrete-time transition for nonuniform intervals Delta t = t_k - t_{k-1}:
    F(Delta t) = [[1, 0, Delta t, 0      ],
                  [0, 1, 0,       Delta t],
                  [0, 0, 1,       0      ],
                  [0, 0, 0,       1      ]]

Discrete process noise covariance Q(Delta t):
    Q(Delta t) = q_std^2 * [[Delta t^4 / 4, 0,             Delta t^3 / 2, 0            ],
                            [0,             Delta t^4 / 4, 0,             Delta t^3 / 2],
                            [Delta t^3 / 2, 0,             Delta t^2,     0            ],
                            [0,             Delta t^3 / 2, 0,             Delta t^2    ]]

Measurement model:
    z_k = H x_k + v_k,  v_k ~ N(0, R_k)
    H = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    R_k = (r_std / confidence)^2 * I_2

Adaptive measurement noise:
    Lower confidence detections (e.g. motion-blurred ball) inflate R_k, reducing filter pull.
    High confidence detections produce tight R_k.

Covariance update uses the numerically stable Joseph form:
    P = (I - K H) P (I - K H)^T + K R K^T
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


class Kalman2D:
    """
    2D discrete-time Kalman filter for tracking ball pixel position and velocity.
    Supports nonuniform frame intervals, adaptive gating, and full covariance propagation.
    """

    def __init__(
        self,
        initial_u: float,
        initial_v: float,
        process_noise_std: float = 80.0,
        measurement_noise_std: float = 2.0,
        initial_velocity_std: float = 1000.0,
        initial_pos_std: float = 5.0
    ):
        """
        Args:
            initial_u: Initial horizontal pixel coordinate.
            initial_v: Initial vertical pixel coordinate.
            process_noise_std: Acceleration spectral density in px/s^2.
            measurement_noise_std: Baseline measurement error in pixels.
            initial_velocity_std: Initial velocity uncertainty in px/s.
            initial_pos_std: Initial position uncertainty in pixels.
        """
        # State: [u, v, Vu, Vv]^T
        self.x = np.array([float(initial_u), float(initial_v), 0.0, 0.0], dtype=np.float64)

        # Initial covariance: high initial velocity variance to lock onto motion rapidly
        p_pos = float(initial_pos_std ** 2)
        p_vel = float(initial_velocity_std ** 2)
        self.P = np.diag([p_pos, p_pos, p_vel, p_vel]).astype(np.float64)

        self.q_std = float(process_noise_std)
        self.r_std = float(measurement_noise_std)

        # Measurement matrix H (observing u and v directly)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Diagnostics cache
        self.last_dt: float = 0.0
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None
        self.last_kalman_gain: Optional[np.ndarray] = None
        self.last_mahalanobis: float = 0.0

    def predict(self, dt: float) -> np.ndarray:
        """
        Propagates state and covariance across time interval dt.
        Handles nonuniform intervals dt > 0.
        """
        dt = max(1e-4, float(dt))
        self.last_dt = dt

        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Continuous White Noise Acceleration Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q_var = self.q_std ** 2

        Q = q_var * np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2]
        ], dtype=np.float64)

        # Predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def get_measurement_covariance(self, confidence: float = 1.0) -> np.ndarray:
        """Computes adaptive measurement noise covariance R scaled by confidence."""
        conf_clamped = max(0.10, min(1.0, float(confidence)))
        effective_std = self.r_std / conf_clamped
        r_var = effective_std ** 2
        return np.diag([r_var, r_var]).astype(np.float64)

    def compute_mahalanobis(self, u: float, v: float, confidence: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes the Mahalanobis distance without modifying the filter state.
        Returns: (mahalanobis_distance, innovation_vector, innovation_covariance_S)
        """
        z = np.array([float(u), float(v)], dtype=np.float64)
        R = self.get_measurement_covariance(confidence)

        # Innovation: y = z - H * x
        y = z - self.H @ self.x
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R

        try:
            S_inv = np.linalg.inv(S)
            d_sq = float(y.T @ S_inv @ y)
            d_m = float(np.sqrt(max(0.0, d_sq)))
        except np.linalg.LinAlgError:
            d_m = float(np.linalg.norm(y) / max(1.0, self.r_std))

        return d_m, y, S

    def is_within_gate(
        self,
        u: float,
        v: float,
        confidence: float = 1.0,
        gate_threshold: float = 3.5
    ) -> Tuple[bool, float, float]:
        """
        Tests whether a candidate detection falls within the adaptive kinematic gate.
        Returns: (is_accepted, mahalanobis_distance, gate_threshold)
        """
        d_m, _, _ = self.compute_mahalanobis(u, v, confidence)
        return (d_m <= gate_threshold), d_m, gate_threshold

    def update(self, u: float, v: float, confidence: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Applies measurement update using observation (u, v).
        Returns: (updated_state, mahalanobis_distance)
        """
        z = np.array([float(u), float(v)], dtype=np.float64)
        R = self.get_measurement_covariance(confidence)

        # Innovation and Innovation Covariance
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        S_inv = np.linalg.inv(S)

        # Mahalanobis distance
        d_sq = float(y.T @ S_inv @ y)
        d_m = float(np.sqrt(max(0.0, d_sq)))

        # Optimal Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Joseph-form covariance update: guarantees symmetry and positive semi-definiteness
        I = np.eye(4, dtype=np.float64)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Cache diagnostics
        self.last_innovation = y.copy()
        self.last_innovation_cov = S.copy()
        self.last_kalman_gain = K.copy()
        self.last_mahalanobis = d_m

        return self.x.copy(), d_m

    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated pixel position (u, v)."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated pixel velocity (Vu, Vv) in px/s."""
        return float(self.x[2]), float(self.x[3])

    @property
    def covariance(self) -> np.ndarray:
        """Full 4x4 state covariance matrix P."""
        return self.P.copy()

    def get_position_uncertainty(self) -> Tuple[float, float]:
        """Standard deviations of position (sigma_u, sigma_v) in pixels."""
        return (
            float(np.sqrt(max(0.0, self.P[0, 0]))),
            float(np.sqrt(max(0.0, self.P[1, 1])))
        )

    def get_velocity_uncertainty(self) -> Tuple[float, float]:
        """Standard deviations of velocity (sigma_Vu, sigma_Vv) in px/s."""
        return (
            float(np.sqrt(max(0.0, self.P[2, 2]))),
            float(np.sqrt(max(0.0, self.P[3, 3])))
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns structured diagnostic dictionary exposing filter uncertainties."""
        sigma_u, sigma_v = self.get_position_uncertainty()
        sigma_vu, sigma_vv = self.get_velocity_uncertainty()
        return {
            "u": round(float(self.x[0]), 2),
            "v": round(float(self.x[1]), 2),
            "Vu": round(float(self.x[2]), 2),
            "Vv": round(float(self.x[3]), 2),
            "sigma_u_px": round(sigma_u, 3),
            "sigma_v_px": round(sigma_v, 3),
            "sigma_Vu_px_s": round(sigma_vu, 2),
            "sigma_Vv_px_s": round(sigma_vv, 2),
            "last_mahalanobis": round(self.last_mahalanobis, 3),
            "last_dt_sec": round(self.last_dt, 5)
        }
