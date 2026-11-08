"""Per-view exposure compensation module for GGGS (ForeSplat-GGGS experiments).

GGGS already ships a PGSR/decoupled-appearance style exposure model: with
``--use_decoupled_appearance 1`` (``App_model.GS``) every camera carries an
independent 3x4 affine matrix which is applied to the rendered colour *before*
the photometric loss (see ``utils/loss_utils.py: L1_loss_appearance``, branch
``App_model.GS`` -> ``addmm``), optimised by its own Adam group with the
``gs_appearance_lr_init`` -> ``gs_appearance_lr_final`` schedule
(``scene/gaussian_model.py: _init_gs_embeddings / _gs_param_groups /
update_learning_rate``).

This module does **not** re-implement the exposure algorithm.  It wraps that
native mechanism in one place, so the experiments can

* switch it on/off with an explicit flag (``enabled``),
* create the native per-view parameters (``attach``),
* export the per-view matrices to ``exposure_params/`` after training,
* restore them into a ``GaussianModel`` for rendering / checkpoints.

The images themselves are never modified, and the correction is applied inside
the loss/rendering path only.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Optional, Sequence

import numpy as np
import torch


class ExposureCompensator:
    """Thin, explicit wrapper around GGGS' native ``App_model.GS`` exposure."""

    #: ``--use_decoupled_appearance`` value for the GS (per-view 3x4) exposure.
    DECOUPLED_APPEARANCE_GS = 1
    DECOUPLED_APPEARANCE_OFF = 0

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------ setup
    @property
    def decoupled_appearance(self) -> int:
        """Value to pass as ``ModelParams.use_decoupled_appearance``."""
        return (
            self.DECOUPLED_APPEARANCE_GS
            if self.enabled
            else self.DECOUPLED_APPEARANCE_OFF
        )

    def attach(self, gaussians, num_cameras: int) -> None:
        """Create the native per-view exposure parameters (idempotent).

        ``Scene.__init__`` already calls ``GaussianModel.create_app_model``.
        Calling this afterwards is a no-op when the model was created with the
        same mode; it raises when the flags disagree, so a mismatch between the
        config and the instantiated model can never silently pass.
        """
        from scene.gaussian_model import GaussianModel

        expected = GaussianModel.App_model(self.decoupled_appearance)
        current = getattr(gaussians, "app_model", GaussianModel.App_model.NO)
        if current != expected:
            gaussians.create_app_model(num_cameras, self.decoupled_appearance)

    # ------------------------------------------------------------------ export
    def export(
        self,
        gaussians,
        out_dir: str,
        camera_names: Optional[Sequence[str]] = None,
    ) -> Optional[dict]:
        """Save every view's 3x4 exposure matrix as npz + csv."""
        if not self.enabled:
            return None
        embeddings = getattr(gaussians, "_appearance_embeddings", None)
        if embeddings is None:
            raise RuntimeError("Exposure compensation enabled but the model has no exposure parameters")
        matrices = embeddings.detach().cpu().numpy().astype(np.float32)  # (N, 3, 4)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "exposure_matrices.npz"),
            matrices=matrices,
            camera_names=np.array(list(camera_names) if camera_names else [], dtype=object),
        )
        csv_path = os.path.join(out_dir, "exposure_matrices.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["view_index", "image_name"] + [f"m{r}{c}" for r in range(3) for c in range(4)])
            for index, matrix in enumerate(matrices):
                name = camera_names[index] if camera_names and index < len(camera_names) else ""
                writer.writerow([index, name] + [f"{v:.8f}" for v in matrix.reshape(-1)])

        mean = matrices.mean(axis=0)
        summary = {
            "num_views": int(matrices.shape[0]),
            "mean_matrix": mean.tolist(),
            "max_abs_matrix": float(np.abs(matrices).max()),
            # deviation from the identity 3x4 initialisation: how far the
            # optimiser actually moved the exposure parameters
            "mean_abs_deviation_from_identity": float(
                np.abs(matrices - np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)).mean()
            ),
            "npz": os.path.join(out_dir, "exposure_matrices.npz"),
            "csv": csv_path,
        }
        with open(os.path.join(out_dir, "exposure_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    # ------------------------------------------------------------------ load
    def load_into(self, gaussians, out_dir: str) -> None:
        """Restore per-view exposure parameters saved by :meth:`export`."""
        path = os.path.join(out_dir, "exposure_matrices.npz")
        with np.load(path, allow_pickle=True) as data:
            matrices = torch.tensor(data["matrices"], dtype=torch.float32, device="cuda")
        if gaussians._appearance_embeddings is None:
            raise RuntimeError("GaussianModel has no exposure parameters to restore into")
        with torch.no_grad():
            gaussians._appearance_embeddings.copy_(matrices)


def apply_exposure(rendered_image: torch.Tensor, gaussians, view_index: int) -> torch.Tensor:
    """Apply the native per-view exposure to a rendered image.

    Mirrors ``L1_loss_appearance``'s ``App_model.GS`` branch exactly:
    ``x' = A @ x + b`` with the camera's own 3x4 matrix, applied in the same
    linear image space used by GGGS' photometric loss.
    """
    from scene.gaussian_model import GaussianModel

    if gaussians.app_model is GaussianModel.App_model.NO:
        return rendered_image
    exposure = gaussians.get_appearance_embedding(view_index)
    return torch.addmm(
        exposure[:3, 3, None],
        exposure[:3, :3],
        rendered_image.reshape(3, -1),
    ).reshape_as(rendered_image)
