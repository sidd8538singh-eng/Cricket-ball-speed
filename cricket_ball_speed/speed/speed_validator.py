"""
Speed Validator: Evaluates calculated bowling speeds against aerodynamic and athletic physics.

CRITICAL SCIENTIFIC PRINCIPLES:
1. Plausibility ranges (30-175 km/h) are strictly diagnostic gates; NEVER clamp or fudge results into them.
2. Returns is_valid=False with an explicit rejection reason if the delivery fails physical validation.
3. Checks aerodynamic drag consistency: in real flight, balls decelerate; huge accelerations indicate trajectory distortion.
4. Checks statistical uncertainty: if 95% confidence interval is excessively wide, flags result.
"""

from typing import Tuple, Optional, Dict, Any, List
from .release_speed import ReleaseSpeedResult
from .c1_c2_speed import C1C2SpeedResult


class SpeedValidator:
    """
    Validates speed outputs against physical laws, athletic boundaries, and uncertainty limits.
    """

    def __init__(
        self,
        min_plausible_speed_kmh: float = 30.0,
        max_plausible_speed_kmh: float = 175.0,
        max_uncertainty_kmh: float = 8.5,
        max_c1_c2_overspeed_kmh: float = 10.0,
        max_c1_c2_deceleration_kmh: float = 40.0
    ):
        """
        Parameters:
            min_plausible_speed_kmh: Slowest cricket delivery (spin/slow ball, ~30 km/h).
            max_plausible_speed_kmh: Fastest plausible bowling delivery (world record 161.3 km/h, upper bound 175 km/h).
            max_uncertainty_kmh: Maximum allowable 1-sigma uncertainty (default 8.5 km/h).
            max_c1_c2_overspeed_kmh: Maximum amount C1-C2 average speed can exceed release speed (drag means it shouldn't).
            max_c1_c2_deceleration_kmh: Maximum allowable drop between release and C2 (~5-8m, normal is ~5-15 km/h).
        """
        self.min_speed = min_plausible_speed_kmh
        self.max_speed = max_plausible_speed_kmh
        self.max_unc = max_uncertainty_kmh
        self.max_overspeed = max_c1_c2_overspeed_kmh
        self.max_deceleration = max_c1_c2_deceleration_kmh

    def validate(
        self,
        release_result: Optional[ReleaseSpeedResult],
        c1_c2_result: Optional[C1C2SpeedResult] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates computed speeds against physical criteria.
        Returns:
            (is_valid, rejection_reason)
        """
        is_valid, reason, _ = self.validate_with_details(release_result, c1_c2_result)
        return is_valid, reason

    def validate_with_details(
        self,
        release_result: Optional[ReleaseSpeedResult],
        c1_c2_result: Optional[C1C2SpeedResult] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validates computed speeds and generates a structured audit dictionary.
        """
        audit: Dict[str, Any] = {
            "checks_passed": [],
            "checks_failed": [],
            "is_valid": False,
            "rejection_reason": None,
        }

        if release_result is None or not release_result.is_valid or release_result.status == "NOT_MEASURED":
            reason = (
                release_result.rejection_reason
                if release_result and release_result.rejection_reason
                else "SPEED_ESTIMATION_FAILED: No valid release speed computed."
            )
            audit["checks_failed"].append("VALID_RELEASE_RESULT")
            audit["rejection_reason"] = reason
            return False, reason, audit

        speed = release_result.speed_km_h
        unc = release_result.uncertainty_km_h

        # 1. Physical range check
        if speed < self.min_speed:
            reason = (
                f"SPEED_BELOW_MINIMUM: Estimated speed {speed:.1f} km/h is below cricket threshold "
                f"({self.min_speed:.1f} km/h)."
            )
            audit["checks_failed"].append("MINIMUM_SPEED_BOUND")
            audit["rejection_reason"] = reason
            return False, reason, audit
        audit["checks_passed"].append("MINIMUM_SPEED_BOUND")

        if speed > self.max_speed:
            reason = (
                f"SPEED_ABOVE_MAXIMUM: Estimated speed {speed:.1f} km/h exceeds physical bowling limit "
                f"({self.max_speed:.1f} km/h)."
            )
            audit["checks_failed"].append("MAXIMUM_SPEED_BOUND")
            audit["rejection_reason"] = reason
            return False, reason, audit
        audit["checks_passed"].append("MAXIMUM_SPEED_BOUND")

        # 2. Uncertainty threshold
        if unc > self.max_unc:
            reason = (
                f"SPEED_UNCERTAINTY_EXCESSIVE: Speed uncertainty +- {unc:.1f} km/h exceeds allowable limit "
                f"(+- {self.max_unc:.1f} km/h)."
            )
            audit["checks_failed"].append("UNCERTAINTY_BOUND")
            audit["rejection_reason"] = reason
            return False, reason, audit
        audit["checks_passed"].append("UNCERTAINTY_BOUND")

        # 3. Aerodynamic consistency between release and C1-C2 average
        if c1_c2_result is not None and c1_c2_result.is_valid and c1_c2_result.status == "MEASURED":
            c1_c2_spd = c1_c2_result.average_speed_km_h
            # In free flight, cricket balls decelerate due to air drag (~5-8% over 8 metres)
            # A massive acceleration (> 10 km/h) indicates trajectory distortion
            if c1_c2_spd > (speed + self.max_overspeed):
                reason = (
                    f"AERODYNAMIC_INCONSISTENCY_ACCELERATION: C1-C2 speed ({c1_c2_spd:.1f} km/h) "
                    f"unphysically exceeds release speed ({speed:.1f} km/h) by > {self.max_overspeed:.1f} km/h."
                )
                audit["checks_failed"].append("AERODYNAMIC_ACCELERATION_CHECK")
                audit["rejection_reason"] = reason
                return False, reason, audit
            audit["checks_passed"].append("AERODYNAMIC_ACCELERATION_CHECK")

            # Check for excessive unphysical deceleration (e.g. ball drops > 40 km/h in first 6m)
            if (speed - c1_c2_spd) > self.max_deceleration:
                reason = (
                    f"AERODYNAMIC_INCONSISTENCY_DECELERATION: Ball dropped {speed - c1_c2_spd:.1f} km/h "
                    f"between release ({speed:.1f} km/h) and C2 ({c1_c2_spd:.1f} km/h), exceeding max drag deceleration limit."
                )
                audit["checks_failed"].append("AERODYNAMIC_DECELERATION_CHECK")
                audit["rejection_reason"] = reason
                return False, reason, audit
            audit["checks_passed"].append("AERODYNAMIC_DECELERATION_CHECK")

        audit["is_valid"] = True
        return True, None, audit
