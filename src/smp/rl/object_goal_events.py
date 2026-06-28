"""Event helpers for object-goal SMP reward integration."""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from smp.rl.object_goal_features import DEFAULT_WINDOW_SIZE, ObjectGoalFeatureBuilder
from smp.rl.object_goal_prior import DEFAULT_SDS_TIMESTEPS, ObjectGoalTwoStagePrior

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _parse_timesteps(value: tuple[int, ...] | list[int] | str) -> tuple[int, ...]:
  if isinstance(value, str):
    parts = [part.strip() for part in value.replace(" ", ",").split(",")]
    out = tuple(int(part) for part in parts if part)
  else:
    out = tuple(int(part) for part in value)
  if not out:
    raise ValueError("fixed_timesteps must be non-empty")
  return out


def _resolve(path: str | Path, base: Path | None = None) -> Path:
  out = Path(path).expanduser()
  if not out.is_absolute() and base is not None:
    out = base / out
  return out.resolve()


def _install_numpy_pickle_aliases() -> None:
  if "numpy._core" not in sys.modules:
    core_pkg = types.ModuleType("numpy._core")
    core_pkg.__path__ = []
    sys.modules["numpy._core"] = core_pkg
  if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
  if "numpy._core.numerictypes" not in sys.modules:
    sys.modules["numpy._core.numerictypes"] = np.core.numerictypes
  if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath


def _set_context_tensor(
  env: "ManagerBasedRlEnv",
  name: str,
  value: torch.Tensor,
) -> None:
  setattr(env, name, value.to(device=env.device, dtype=torch.float32))


def load_object_goal_hf_bps_context(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  g1_diffusion_root: str = "../g1-diffusion",
  input_pkl: str = "../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl",
  start: int = 0,
  window_size: int = DEFAULT_WINDOW_SIZE,
) -> None:
  """Load real HF-BPS context tensors onto the env for reward evaluation.

  This is a bridge helper, not a fake object task.  A full mjlab task still needs
  a real object asset whose body pose maps to the HF-BPS mesh-centroid pose.
  """
  del env_ids
  g1_root = _resolve(g1_diffusion_root)
  input_path = _resolve(input_pkl)
  if not input_path.exists():
    raise FileNotFoundError(f"HF-BPS context PKL not found: {input_path}")

  _install_numpy_pickle_aliases()
  with open(input_path, "rb") as f:
    data = pickle.load(f)
  builder = ObjectGoalFeatureBuilder(g1_root, device=env.device)
  sample = builder.hf_bps_window_to_tensors(data, start=start, window_size=window_size)

  _set_context_tensor(env, "_object_goal_bps_encoding", sample["bps"])
  _set_context_tensor(env, "_object_goal_static_bps_context", sample["static_bps_context"])
  _set_context_tensor(env, "_object_goal_final_object_pose_raw", sample["goal_raw"])
  if "object_verts" in sample:
    _set_context_tensor(env, "_object_goal_object_verts", sample["object_verts"])
  if "object_rotation" in sample:
    _set_context_tensor(env, "_object_goal_object_rotations", sample["object_rotation"])
  if "contact_labels" in sample:
    _set_context_tensor(env, "_object_goal_contact_labels", sample["contact_labels"])

  mesh_file = data.get("mesh_file")
  if mesh_file:
    env._object_goal_mesh_file = str(mesh_file)  # type: ignore[attr-defined]
  if data.get("object_mesh_scale") is not None:
    env._object_goal_object_mesh_scale = float(data["object_mesh_scale"])  # type: ignore[attr-defined]
  if data.get("object_name") is not None:
    env._object_goal_object_name = str(data["object_name"])  # type: ignore[attr-defined]

  # These are useful for diagnostic envs that do not yet instantiate a real
  # object entity. The reward still fails without explicit object pose or a real
  # object body, preserving the no-free-box constraint.
  env._object_goal_context_source = str(input_path)  # type: ignore[attr-defined]
  env._object_goal_context_window = (int(start), int(start) + int(window_size))  # type: ignore[attr-defined]
  env._object_goal_bps_basis_required = True  # type: ignore[attr-defined]


def init_object_goal_prior(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  g1_diffusion_root: str = "../g1-diffusion",
  stage1_ckpt_path: str = "",
  stage2_ckpt_path: str = "",
  fixed_timesteps: tuple[int, ...] | list[int] | str = DEFAULT_SDS_TIMESTEPS,
  ws: float = 6.0,
  contact_threshold: float = 0.03,
  stage1_contact_search_threshold: float | None = 0.08,
  stage1_max_contact_offset: float | None = 0.02,
  stage1_max_contact_correction: float | None = 0.06,
  stage1_fallback_contact_search_threshold: float | None = 0.25,
  stage1_fallback_max_contact_correction: float | None = 0.60,
  hf_bps_context_pkl: str = "",
  context_start: int = 0,
) -> None:
  """Startup event: load the frozen two-stage object-goal prior onto ``env``."""
  del env_ids
  if not stage1_ckpt_path or not stage2_ckpt_path:
    msg = (
      "init_object_goal_prior requires both stage1_ckpt_path and "
      "stage2_ckpt_path. They must point to corrected object_goal_two_stage "
      "checkpoints, not legacy Stage 1/Stage 2 checkpoints."
    )
    raise RuntimeError(msg)

  timesteps = _parse_timesteps(fixed_timesteps)
  prior = ObjectGoalTwoStagePrior(
    g1_diffusion_root=g1_diffusion_root,
    stage1_ckpt_path=stage1_ckpt_path,
    stage2_ckpt_path=stage2_ckpt_path,
    device=env.device,
    fixed_timesteps=timesteps,
    ws=ws,
    contact_threshold=contact_threshold,
    stage1_contact_search_threshold=stage1_contact_search_threshold,
    stage1_max_contact_offset=stage1_max_contact_offset,
    stage1_max_contact_correction=stage1_max_contact_correction,
    stage1_fallback_contact_search_threshold=stage1_fallback_contact_search_threshold,
    stage1_fallback_max_contact_correction=stage1_fallback_max_contact_correction,
  )
  env._object_goal_prior = prior  # type: ignore[attr-defined]
  env._object_goal_fixed_timesteps = timesteps  # type: ignore[attr-defined]
  env._object_goal_ws = float(ws)  # type: ignore[attr-defined]

  if hf_bps_context_pkl:
    load_object_goal_hf_bps_context(
      env,
      g1_diffusion_root=g1_diffusion_root,
      input_pkl=hf_bps_context_pkl,
      start=context_start,
      window_size=prior.window_size,
    )
