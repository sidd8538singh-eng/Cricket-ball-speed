"""
Model Class Mapper & Resolver.
Dynamically inspects YOLO model metadata, extracts actual class mappings,
and maps target semantic entities (ball, bowler, stumps, pitch) to their respective class IDs.
Prevents hardcoded class ID assumptions (e.g., assuming class 0 is always the target).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger("CricketBallSpeed.ClassMapper")

# Standard keyword aliases for semantic target classes
DEFAULT_BALL_KEYWORDS = ["ball", "cricket_ball", "cricket ball", "sports ball", "sportsball", "baseball", "tennis ball"]
DEFAULT_BOWLER_KEYWORDS = ["bowler", "person", "player", "athlete", "human"]
DEFAULT_STUMPS_KEYWORDS = ["stumps", "stump", "wicket", "wickets", "cricket_stumps", "cricket_wicket"]
DEFAULT_PITCH_KEYWORDS = ["pitch", "cricket_pitch", "wicket_pitch", "pitch_surface"]


@dataclass
class ClassResolutionResult:
    """Encapsulates the resolved class IDs and names for a model."""
    target_class_ids: List[int]
    target_class_names: List[str]
    all_model_classes: Dict[int, str]
    is_fallback: bool = False
    resolution_source: str = "model_metadata"  # "model_metadata", "explicit_config", "fallback"


class ModelClassResolver:
    """
    Inspects model class definitions and resolves semantic labels to integer class IDs.
    """

    @staticmethod
    def extract_model_names(model: Any) -> Dict[int, str]:
        """
        Extracts class dictionary {class_id: class_name} from Ultralytics YOLO,
        PyTorch checkpoint, or custom model dictionary.
        """
        if model is None:
            return {}

        # 1. Ultralytics model object with .names attribute
        if hasattr(model, "names") and model.names is not None:
            names = model.names
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                return {int(i): str(name) for i, name in enumerate(names)}

        # 2. PyTorch checkpoint dictionary
        if isinstance(model, dict):
            if "names" in model:
                names = model["names"]
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, (list, tuple)):
                    return {int(i): str(name) for i, name in enumerate(names)}
            if "model" in model and hasattr(model["model"], "names"):
                names = model["model"].names
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, (list, tuple)):
                    return {int(i): str(name) for i, name in enumerate(names)}

        return {}

    @classmethod
    def resolve_classes(
        cls,
        model: Any,
        target_keywords: List[str],
        explicit_class_ids: Optional[List[int]] = None,
        explicit_class_names: Optional[List[str]] = None,
        fallback_class_ids: Optional[List[int]] = None,
        entity_name: str = "target"
    ) -> ClassResolutionResult:
        """
        Resolves class IDs for a given entity type using:
        1. Model metadata matched against target keywords or explicit names.
        2. Explicitly configured class IDs.
        3. Fallback defaults.
        """
        all_classes = cls.extract_model_names(model)

        # 1. If explicit class IDs were provided by config and validated against model
        if explicit_class_ids:
            target_ids = list(explicit_class_ids)
            target_names = [all_classes.get(cid, f"class_{cid}") for cid in target_ids]
            logger.info(
                f"Resolved {entity_name} using explicit config class IDs: {target_ids} ({target_names})"
            )
            return ClassResolutionResult(
                target_class_ids=target_ids,
                target_class_names=target_names,
                all_model_classes=all_classes,
                is_fallback=False,
                resolution_source="explicit_config"
            )

        # 2. Match keywords against model.names
        if all_classes:
            matched_ids: List[int] = []
            matched_names: List[str] = []

            # Normalize keywords
            search_keywords = [k.lower().strip().replace(" ", "_") for k in target_keywords]
            if explicit_class_names:
                search_keywords = [n.lower().strip().replace(" ", "_") for n in explicit_class_names] + search_keywords

            for cid, cname in all_classes.items():
                norm_name = cname.lower().strip().replace(" ", "_")
                # Direct match or substring match
                for kw in search_keywords:
                    if norm_name == kw or kw in norm_name or norm_name in kw:
                        if cid not in matched_ids:
                            matched_ids.append(cid)
                            matched_names.append(cname)
                        break

            if matched_ids:
                logger.info(
                    f"Resolved {entity_name} from model metadata: class IDs {matched_ids} ({matched_names})"
                )
                return ClassResolutionResult(
                    target_class_ids=matched_ids,
                    target_class_names=matched_names,
                    all_model_classes=all_classes,
                    is_fallback=False,
                    resolution_source="model_metadata"
                )

        # 3. Fallback
        default_ids = fallback_class_ids if fallback_class_ids is not None else [0]
        default_names = [all_classes.get(cid, f"class_{cid}") for cid in default_ids]
        logger.warning(
            f"Could not automatically match {entity_name} in model metadata {all_classes}. "
            f"Falling back to default class IDs: {default_ids}"
        )
        return ClassResolutionResult(
            target_class_ids=default_ids,
            target_class_names=default_names,
            all_model_classes=all_classes,
            is_fallback=True,
            resolution_source="fallback"
        )
