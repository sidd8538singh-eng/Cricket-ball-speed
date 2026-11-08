"""
Model Validator: Validates model paths, weights existence, device availability,
and input/output configurations.
Ensures the three mandatory models (ball, bowler, stumps) are present and verified before inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import os
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("CricketBallSpeed.ModelValidator")


@dataclass
class ModelInfo:
    name: str
    path: str
    mandatory: bool
    exists: bool
    file_size_bytes: int = 0
    device: str = "cpu"
    status: str = "OK"
    error: Optional[str] = None


@dataclass
class ValidationReport:
    is_valid: bool
    device_selected: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    missing_mandatory: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "device_selected": self.device_selected,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "missing_mandatory": self.missing_mandatory,
            "warnings": self.warnings,
            "models": {
                k: {
                    "path": v.path,
                    "mandatory": v.mandatory,
                    "exists": v.exists,
                    "file_size_bytes": v.file_size_bytes,
                    "status": v.status,
                    "error": v.error
                } for k, v in self.models.items()
            }
        }


class ModelValidator:
    """
    Validates model paths, mandatory requirements, and hardware acceleration devices.
    """

    MANDATORY_MODELS = ["ball", "bowler", "stumps"]
    OPTIONAL_MODELS = ["pitch"]

    @classmethod
    def resolve_device(cls, requested_device: str = "auto") -> Tuple[str, bool, Optional[str]]:
        """
        Determines the execution device:
        - "auto": chooses CUDA if available, else CPU.
        - "cuda": verifies CUDA is available.
        - "cpu": standard CPU.
        """
        if not TORCH_AVAILABLE:
            logger.info("PyTorch is not available; defaulting to CPU mode.")
            return "cpu", False, None

        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None

        req = requested_device.lower().strip()
        if req == "auto":
            if cuda_avail:
                selected = "cuda:0"
                logger.info(f"Hardware acceleration detected: using GPU '{gpu_name}' ({selected}).")
            else:
                selected = "cpu"
                logger.info("No CUDA GPU detected; defaulting to CPU.")
        elif req.startswith("cuda"):
            if not cuda_avail:
                logger.warning(f"CUDA was requested ('{requested_device}') but torch.cuda.is_available() is False. Falling back to CPU.")
                selected = "cpu"
            else:
                selected = requested_device
        else:
            selected = "cpu"

        return selected, cuda_avail, gpu_name

    @classmethod
    def validate_models(
        cls,
        model_paths: Dict[str, str],
        requested_device: str = "auto",
        raise_on_error: bool = False
    ) -> ValidationReport:
        """
        Validates all model paths and device configurations.
        """
        device, cuda_avail, gpu_name = cls.resolve_device(requested_device)
        models_info: Dict[str, ModelInfo] = {}
        missing_mandatory: List[str] = []
        warnings: List[str] = []

        # Validate mandatory models
        for name in cls.MANDATORY_MODELS:
            path = model_paths.get(name, "")
            exists = bool(path and os.path.isfile(path))
            size = os.path.getsize(path) if exists else 0
            err = None
            status = "OK"

            if not exists:
                status = "MISSING_MANDATORY"
                err = f"Mandatory model '{name}' not found at path: {path}"
                missing_mandatory.append(name)
                warnings.append(err)

            models_info[name] = ModelInfo(
                name=name,
                path=path,
                mandatory=True,
                exists=exists,
                file_size_bytes=size,
                device=device,
                status=status,
                error=err
            )

        # Validate optional models (pitch)
        for name in cls.OPTIONAL_MODELS:
            path = model_paths.get(name, "")
            exists = bool(path and os.path.isfile(path))
            size = os.path.getsize(path) if exists else 0
            status = "OK" if exists else "OPTIONAL_ABSENT"

            models_info[name] = ModelInfo(
                name=name,
                path=path,
                mandatory=False,
                exists=exists,
                file_size_bytes=size,
                device=device,
                status=status,
                error=None
            )

        is_valid = len(missing_mandatory) == 0

        report = ValidationReport(
            is_valid=is_valid,
            device_selected=device,
            cuda_available=cuda_avail,
            gpu_name=gpu_name,
            models=models_info,
            missing_mandatory=missing_mandatory,
            warnings=warnings
        )

        if not is_valid and raise_on_error:
            raise FileNotFoundError(
                f"Missing mandatory models: {', '.join(missing_mandatory)}. "
                f"Please provide paths to ball, bowler, and stumps weights."
            )

        return report
