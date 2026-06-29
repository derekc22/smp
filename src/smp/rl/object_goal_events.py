"""Event helpers for object-goal SMP reward integration."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from smp.rl.object_goal_assets import (
  ObjectGoalRuntimeContextBuilder,
  hf_sample_window_tensors,
  load_hf_bps_pickle,
  load_hf_object_asset_metadata,
  mesh_centroid_to_body_pos,
)
from smp.rl.object_goal_features import (
  DEFAULT_WINDOW_SIZE,
  ObjectGoalFeatureBuilder,
  ObjectGoalMotionBuffer,
)
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

  data = load_hf_bps_pickle(input_path)
  builder = ObjectGoalFeatureBuilder(g1_root, device=env.device)
  sample = builder.hf_bps_window_to_tensors(data, start=start, window_size=window_size)
  metadata = load_hf_object_asset_metadata(
    input_pkl=input_path,
    g1_diffusion_root=g1_root,
  )
  sample_window = hf_sample_window_tensors(
    metadata,
    start=start,
    window_size=window_size,
    device=env.device,
  )

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

  env._object_goal_asset_metadata = metadata  # type: ignore[attr-defined]
  env._object_goal_runtime_context_builder = ObjectGoalRuntimeContextBuilder(  # type: ignore[attr-defined]
    metadata,
    device=env.device,
  )
  _set_context_tensor(
    env,
    "_object_goal_mesh_centroid_offset_local",
    torch.as_tensor(metadata.mesh_centroid_offset_local, dtype=torch.float32),
  )
  _set_context_tensor(
    env,
    "_object_goal_object_mesh_vertices_local",
    torch.as_tensor(metadata.mesh_vertices_local, dtype=torch.float32),
  )
  _set_context_tensor(
    env,
    "_object_goal_bps_basis",
    torch.as_tensor(metadata.bps_basis, dtype=torch.float32),
  )
  env._object_goal_bps_radius = float(metadata.bps_radius)  # type: ignore[attr-defined]
  env._object_goal_body_origin_semantics = "original_mesh_origin"  # type: ignore[attr-defined]
  env._object_goal_sample_window = sample_window  # type: ignore[attr-defined]
  env._object_goal_final_object_quat_wxyz = sample_window["object_quat_wxyz"][  # type: ignore[attr-defined]
    :,
    -1,
  ]

  env._object_goal_context_source = str(input_path)  # type: ignore[attr-defined]
  env._object_goal_context_window = (int(start), int(start) + int(window_size))  # type: ignore[attr-defined]
  env._object_goal_bps_basis_required = True  # type: ignore[attr-defined]


def ensure_object_goal_hf_bps_context(env: "ManagerBasedRlEnv") -> None:
  """Attach real HF-BPS object context before managers probe terms.

  mjlab builds observation/reward managers before startup events run, so terms
  that need the object-goal centroid offset must be able to initialize the
  context from the task config.  This helper is intentionally idempotent and
  loads only HF-BPS metadata/context tensors; it does not load diffusion
  checkpoints or instantiate any object entity.
  """
  if hasattr(env, "_object_goal_mesh_centroid_offset_local"):
    return
  if not hasattr(env, "cfg") or "init_object_goal_prior" not in env.cfg.events:
    msg = (
      "Object-goal context is missing and env.cfg.events['init_object_goal_prior'] "
      "is not available to initialize it."
    )
    raise RuntimeError(msg)

  params = env.cfg.events["init_object_goal_prior"].params
  g1_diffusion_root = params.get("g1_diffusion_root", "../g1-diffusion")
  input_pkl = params.get("hf_bps_context_pkl", "")
  if not input_pkl:
    msg = (
      "Object-goal context is missing and init_object_goal_prior.params does not "
      "provide 'hf_bps_context_pkl'."
    )
    raise RuntimeError(msg)
  context_start = int(params.get("context_start", 0))
  window_size = int(
    params.get(
      "context_window_size",
      params.get("window_size", DEFAULT_WINDOW_SIZE),
    )
  )
  load_object_goal_hf_bps_context(
    env,
    g1_diffusion_root=g1_diffusion_root,
    input_pkl=input_pkl,
    start=context_start,
    window_size=window_size,
  )


@torch.no_grad()
def object_goal_sample_reset(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
) -> None:
  """Reset robot/object state and prime the 47D object-goal reward buffer.

  This is intentionally deterministic for the first object-goal integration:
  it replays one real HF-BPS window, writes its tail frame to sim, and fills the
  online reward buffer with the full window.  The object body is authored at the
  original mesh origin, so the body state is offset from the HF-BPS centroid.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  ensure_object_goal_hf_bps_context(env)
  if not hasattr(env, "_object_goal_sample_window"):
    msg = "object_goal_sample_reset requires load_object_goal_hf_bps_context first"
    raise RuntimeError(msg)
  try:
    obj = env.scene["object"]
  except KeyError as exc:
    msg = "object_goal_sample_reset requires env.scene['object']"
    raise RuntimeError(msg) from exc

  sample: dict[str, torch.Tensor] = env._object_goal_sample_window  # type: ignore[attr-defined]
  prior: ObjectGoalTwoStagePrior | None = getattr(env, "_object_goal_prior", None)
  window_size = int(prior.window_size if prior is not None else sample["root_pos"].shape[1])
  n = int(env_ids.numel())

  def expand(name: str) -> torch.Tensor:
    value = sample[name].to(device=env.device, dtype=torch.float32)
    if value.shape[1] != window_size:
      value = value[:, :window_size]
    return value.expand(n, -1, -1).clone()

  root_pos = expand("root_pos")
  root_quat = expand("root_quat_wxyz")
  dof_pos = expand("dof_pos")
  object_centroid = expand("object_centroid")
  object_quat = expand("object_quat_wxyz")

  origins = env.scene.env_origins[env_ids]
  robot = env.scene["robot"]
  root_vel = torch.zeros(n, 6, device=env.device)
  robot_root_state = torch.cat(
    [root_pos[:, -1] + origins, root_quat[:, -1], root_vel],
    dim=-1,
  )
  robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    dof_pos[:, -1],
    torch.zeros_like(dof_pos[:, -1]),
    env_ids=env_ids,
  )

  offset = torch.as_tensor(
    env._object_goal_mesh_centroid_offset_local,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  object_body_pos = mesh_centroid_to_body_pos(
    object_centroid[:, -1],
    object_quat[:, -1],
    offset,
  )
  object_root_state = torch.cat(
    [object_body_pos + origins, object_quat[:, -1], root_vel],
    dim=-1,
  )
  obj.write_root_state_to_sim(object_root_state, env_ids=env_ids)

  if not hasattr(env, "_object_goal_buffer"):
    if prior is None:
      msg = "object_goal_sample_reset requires env._object_goal_prior before buffer init"
      raise RuntimeError(msg)
    env._object_goal_buffer = ObjectGoalMotionBuffer(  # type: ignore[attr-defined]
      num_envs=env.num_envs,
      window_size=window_size,
      g1_diffusion_root=prior.g1_diffusion_root,
      device=env.device,
    )
  buffer: ObjectGoalMotionBuffer = env._object_goal_buffer  # type: ignore[attr-defined]
  buffer.reset(
    env_ids,
    root_pos,
    root_quat,
    dof_pos,
    object_centroid,
    object_quat,
  )


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
  context_window_size: int | None = None,
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

  if hf_bps_context_pkl and not hasattr(env, "_object_goal_mesh_centroid_offset_local"):
    load_object_goal_hf_bps_context(
      env,
      g1_diffusion_root=g1_diffusion_root,
      input_pkl=hf_bps_context_pkl,
      start=context_start,
      window_size=prior.window_size if context_window_size is None else context_window_size,
    )
