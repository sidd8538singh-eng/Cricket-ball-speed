"""
Ball Observation and Processing Subsystem.
Includes observation representations, sub-pixel centroid refinement, apparent diameter,
quality scoring, and kinematic outlier rejection.
"""

from .ball_observations import BallObservation
from .bbox_refinement import BoundingBoxRefiner
from .apparent_diameter import ApparentDiameterEstimator
from .observation_quality import ObservationQualityAssessor
from .outlier_rejection import OutlierRejector

__all__ = [
    "BallObservation",
    "BoundingBoxRefiner",
    "ApparentDiameterEstimator",
    "ObservationQualityAssessor",
    "OutlierRejector",
]
