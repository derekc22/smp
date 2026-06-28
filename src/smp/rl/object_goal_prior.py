"""Frozen two-stage object-goal g1-diffusion prior for SMP rewards."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smp.rl.object_goal_features import (
  DEFAULT_WINDOW_SIZE,
  HAND_DIM,
  OBJECT_POSE_DIM,
  ROBOT_OBJECT_STATE_DIM,
  ROBOT_STATE_DIM,
  STAGE2_COND_DIM,
  STATIC_BPS_DIM,
)

DEFAULT_SDS_TIMESTEPS = (160, 300, 440)


def _ensure_g1_root(g1_diffusion_root: str | Path) -> Path:
  root = Path(g1_diffusion_root).expanduser().resolve()
  if not root.is_dir():
    msg = f"g1_diffusion_root does not exist or is not a directory: {root}"
    raise FileNotFoundError(msg)
  root_str = str(root)
  if root_str not in sys.path:
    sys.path.insert(0, root_str)
  return root


def _resolve(path: str | Path, base: Path) -> Path:
  out = Path(path).expanduser()
  if not out.is_absolute():
    out = base / out
  return out.resolve()


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _as_stat(
  norm_stats: dict[str, Any],
  name: str,
  dim: int,
  device: torch.device,
) -> torch.Tensor:
  if name not in norm_stats or norm_stats[name] is None:
    msg = f"Checkpoint norm_stats missing required '{name}'"
    raise ValueError(msg)
  value = torch.as_tensor(norm_stats[name], device=device, dtype=torch.float32).reshape(-1)
  if value.numel() != dim:
    msg = f"norm_stats['{name}'] must have {dim} values, got {value.numel()}"
    raise ValueError(msg)
  return value


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
  if x.ndim == 3:
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1).clamp_min(1e-8)
  if x.ndim == 2:
    return (x - mean.view(1, -1)) / std.view(1, -1).clamp_min(1e-8)
  msg = f"normalize expects rank 2 or 3 tensor, got {tuple(x.shape)}"
  raise ValueError(msg)


def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
  if x.ndim == 3:
    return x * std.view(1, 1, -1) + mean.view(1, 1, -1)
  if x.ndim == 2:
    return x * std.view(1, -1) + mean.view(1, -1)
  msg = f"denormalize expects rank 2 or 3 tensor, got {tuple(x.shape)}"
  raise ValueError(msg)


def _tensor_norm_per_env(x: torch.Tensor) -> torch.Tensor:
  return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)


class ObjectGoalTwoStagePrior:
  """Load and score frozen object-goal Stage 1/Stage 2 checkpoints."""

  def __init__(
    self,
    g1_diffusion_root: str | Path,
    stage1_ckpt_path: str | Path,
    stage2_ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    fixed_timesteps: tuple[int, ...] = DEFAULT_SDS_TIMESTEPS,
    ws: float = 6.0,
    contact_threshold: float = 0.03,
    stage1_contact_search_threshold: float | None = 0.08,
    stage1_max_contact_offset: float | None = 0.02,
    stage1_max_contact_correction: float | None = 0.06,
    stage1_fallback_contact_search_threshold: float | None = 0.25,
    stage1_fallback_max_contact_correction: float | None = 0.60,
  ) -> None:
    self.g1_diffusion_root = _ensure_g1_root(g1_diffusion_root)
    self.device = torch.device(device)
    self.fixed_timesteps = tuple(int(t) for t in fixed_timesteps)
    self.ws = float(ws)

    self._import_g1_symbols()
    self.stage1_ckpt_path = _resolve(stage1_ckpt_path, self.g1_diffusion_root)
    self.stage2_ckpt_path = _resolve(stage2_ckpt_path, self.g1_diffusion_root)
    self.stage1_ckpt = torch.load(
      self.stage1_ckpt_path,
      map_location=self.device,
      weights_only=False,
    )
    self.stage2_ckpt = torch.load(
      self.stage2_ckpt_path,
      map_location=self.device,
      weights_only=False,
    )

    self._validate_stage1_checkpoint(self.stage1_ckpt)
    self._validate_stage2_checkpoint(self.stage2_ckpt)

    self.stage1_model = self._stage1_model_from_ckpt(self.stage1_ckpt)
    self.stage2_model, self.stage2_state_dim, self.stage2_cond_dim = (
      self._stage2_model_from_ckpt(self.stage2_ckpt)
    )
    self.stage1_schedule = self._schedule_from_ckpt(self.stage1_ckpt)
    self.stage2_schedule = self._schedule_from_ckpt(self.stage2_ckpt)
    self.window_size = int(
      self.stage2_ckpt["config"].get("dataset", {}).get(
        "window_size",
        DEFAULT_WINDOW_SIZE,
      )
    )

    s1_norm = self.stage1_ckpt["norm_stats"]
    s2_norm = self.stage2_ckpt["norm_stats"]
    self.stage1_hand_mean = _as_stat(s1_norm, "hand_mean", HAND_DIM, self.device)
    self.stage1_hand_std = _as_stat(s1_norm, "hand_std", HAND_DIM, self.device)
    self.stage1_goal_mean = _as_stat(s1_norm, "goal_mean", OBJECT_POSE_DIM, self.device)
    self.stage1_goal_std = _as_stat(s1_norm, "goal_std", OBJECT_POSE_DIM, self.device)
    self.stage2_state_mean = _as_stat(
      s2_norm,
      "state_mean",
      ROBOT_OBJECT_STATE_DIM,
      self.device,
    )
    self.stage2_state_std = _as_stat(
      s2_norm,
      "state_std",
      ROBOT_OBJECT_STATE_DIM,
      self.device,
    )
    self.stage2_hand_mean = _as_stat(s2_norm, "hand_mean", HAND_DIM, self.device)
    self.stage2_hand_std = _as_stat(s2_norm, "hand_std", HAND_DIM, self.device)
    self.stage2_goal_mean = _as_stat(s2_norm, "goal_mean", OBJECT_POSE_DIM, self.device)
    self.stage2_goal_std = _as_stat(s2_norm, "goal_std", OBJECT_POSE_DIM, self.device)
    self.stage2_normalize_hands = bool(
      self.stage2_ckpt["config"].get("dataset", {}).get("normalize_hands", False)
    )
    self.stage2_prediction_type = str(self.stage2_ckpt["prediction_type"])

    self.contact_processor = self.ContactConstraintProcessor(
      contact_threshold=float(contact_threshold),
      contact_search_threshold=stage1_contact_search_threshold,
      max_contact_offset=stage1_max_contact_offset,
      max_contact_correction=stage1_max_contact_correction,
      fallback_contact_search_threshold=stage1_fallback_contact_search_threshold,
      fallback_max_contact_correction=stage1_fallback_max_contact_correction,
    )

  @property
  def stage2_num_timesteps(self) -> int:
    return int(self.stage2_schedule.timesteps)

  def _import_g1_symbols(self) -> None:
    from models.stage1_diffusion import (  # type: ignore[import-not-found]
      Stage1HandDiffusion,
      Stage1HandDiffusionMLP,
    )
    from models.stage2_diffusion import (  # type: ignore[import-not-found]
      Stage2MLPModel,
      Stage2TransformerModel,
    )
    from utils.contact_constraints import (  # type: ignore[import-not-found]
      ContactConstraintProcessor,
    )
    from utils.diffusion import (  # type: ignore[import-not-found]
      DiffusionConfig,
      DiffusionSchedule,
    )

    self.Stage1HandDiffusion = Stage1HandDiffusion
    self.Stage1HandDiffusionMLP = Stage1HandDiffusionMLP
    self.Stage2MLPModel = Stage2MLPModel
    self.Stage2TransformerModel = Stage2TransformerModel
    self.ContactConstraintProcessor = ContactConstraintProcessor
    self.DiffusionConfig = DiffusionConfig
    self.DiffusionSchedule = DiffusionSchedule

  def _validate_stage1_checkpoint(self, ckpt: dict[str, Any]) -> None:
    _require(
      ckpt.get("pipeline_type") == "object_goal_two_stage",
      "Stage 1 checkpoint must have pipeline_type='object_goal_two_stage'",
    )
    _require(ckpt.get("stage") == 1, "Stage 1 checkpoint must have stage=1")
    _require(
      ckpt.get("prediction_type") == "x0",
      "Stage 1 online sampling currently requires prediction_type='x0'",
    )
    condition = ckpt.get("condition") or {}
    _require(
      int(condition.get("object_pose_trajectory_dim", -1)) == OBJECT_POSE_DIM,
      "Stage 1 condition.object_pose_trajectory_dim must be 9",
    )
    _require(
      int(condition.get("goal_dim", -1)) == OBJECT_POSE_DIM,
      "Stage 1 condition.goal_dim must be 9",
    )
    norm = ckpt.get("norm_stats") or {}
    for name in ("hand_mean", "hand_std", "goal_mean", "goal_std"):
      _require(name in norm and norm[name] is not None, f"Stage 1 missing norm_stats.{name}")
    self._validate_schedule_metadata(ckpt, "Stage 1")

  def _validate_stage2_checkpoint(self, ckpt: dict[str, Any]) -> None:
    _require(
      ckpt.get("pipeline_type") == "object_goal_two_stage",
      "Stage 2 checkpoint must have pipeline_type='object_goal_two_stage'",
    )
    _require(ckpt.get("stage") == 2, "Stage 2 checkpoint must have stage=2")
    _require(
      str(ckpt.get("prediction_type")) in {"x0", "epsilon"},
      "Stage 2 prediction_type must be 'x0' or 'epsilon'",
    )
    _require(
      int(ckpt.get("state_dim", -1)) == ROBOT_OBJECT_STATE_DIM,
      "Stage 2 state_dim must be 47",
    )
    _require(
      int(ckpt.get("cond_dim", -1)) == STAGE2_COND_DIM,
      "Stage 2 cond_dim must be 3078",
    )
    _require(
      bool(ckpt.get("hand_contact_rectification_required", False)),
      "Stage 2 must declare hand_contact_rectification_required=True",
    )
    layout = ckpt.get("layout") or {}
    _require(layout.get("robot_state") == [0, ROBOT_STATE_DIM], "Bad Stage 2 robot slice")
    _require(
      layout.get("object_pose") == [ROBOT_STATE_DIM, ROBOT_OBJECT_STATE_DIM],
      "Bad Stage 2 object slice",
    )
    condition = ckpt.get("condition") or {}
    _require(
      condition.get("per_frame") == "hands + static_bps_context",
      "Stage 2 condition.per_frame must be 'hands + static_bps_context'",
    )
    _require(
      int(condition.get("global_goal_dim", -1)) == OBJECT_POSE_DIM,
      "Stage 2 condition.global_goal_dim must be 9",
    )
    _require(
      condition.get("object_context_mode", "static_bps") == "static_bps",
      "Stage 2 must use object_context_mode='static_bps'",
    )
    norm = ckpt.get("norm_stats") or {}
    for name in (
      "state_mean",
      "state_std",
      "hand_mean",
      "hand_std",
      "goal_mean",
      "goal_std",
    ):
      _require(name in norm and norm[name] is not None, f"Stage 2 missing norm_stats.{name}")
    self._validate_schedule_metadata(ckpt, "Stage 2")

  def _validate_schedule_metadata(self, ckpt: dict[str, Any], label: str) -> None:
    schedule = ckpt.get("schedule") or {}
    for name in ("timesteps", "beta_start", "beta_end"):
      _require(name in schedule, f"{label} checkpoint missing schedule.{name}")

  def _stage1_model_from_ckpt(self, ckpt: dict[str, Any]) -> torch.nn.Module:
    config = ckpt["config"]
    arch = config.get("train", {}).get("architecture", "transformer")
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = int(dataset_cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    common = {
      "bps_dim": int(model_cfg.get("bps_dim", STATIC_BPS_DIM)),
      "centroid_dim": int(model_cfg.get("centroid_dim", 3)),
      "encoder_hidden": int(model_cfg.get("encoder_hidden", 512)),
      "object_feature_dim": int(model_cfg.get("object_feature_dim", 256)),
      "encoder_layers": int(model_cfg.get("encoder_layers", 3)),
      "hand_dim": int(model_cfg.get("hand_dim", HAND_DIM)),
      "object_pose_dim": OBJECT_POSE_DIM,
      "global_cond_dim": OBJECT_POSE_DIM,
      "global_cond_hidden": model_cfg.get("global_cond_hidden"),
    }
    if arch == "transformer":
      model = self.Stage1HandDiffusion(
        **common,
        d_model=int(model_cfg.get("d_model", 256)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_transformer_layers=int(model_cfg.get("num_layers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", window_size)),
      )
    elif arch == "mlp":
      model = self.Stage1HandDiffusionMLP(
        **common,
        denoiser_hidden=int(model_cfg.get("denoiser_hidden", 512)),
        denoiser_layers=int(model_cfg.get("denoiser_layers", 4)),
      )
    else:
      msg = f"Unsupported Stage 1 architecture {arch!r}"
      raise ValueError(msg)
    model.load_state_dict(ckpt["model"])
    model.to(self.device)
    model.eval()
    model.requires_grad_(False)
    return model

  def _stage2_model_from_ckpt(
    self,
    ckpt: dict[str, Any],
  ) -> tuple[torch.nn.Module, int, int]:
    config = ckpt["config"]
    arch = config.get("train", {}).get("architecture", "transformer")
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    window_size = int(dataset_cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    state = ckpt["model"]
    if "out_proj.weight" in state:
      state_dim = int(state["out_proj.weight"].shape[0])
      input_dim = int(state["state_proj.weight"].shape[1])
      cond_dim = input_dim - state_dim
    else:
      state_dim = int(ckpt["state_dim"])
      cond_dim = int(ckpt["cond_dim"])
    _require(state_dim == ROBOT_OBJECT_STATE_DIM, "Stage 2 model output must be 47D")
    _require(cond_dim == STAGE2_COND_DIM, "Stage 2 model condition must be 3078D")
    common = {
      "state_dim": state_dim,
      "cond_dim": cond_dim,
      "global_cond_dim": OBJECT_POSE_DIM,
      "global_cond_hidden": model_cfg.get("global_cond_hidden"),
      "contact_dim": int(model_cfg.get("contact_dim", 0)),
    }
    if arch == "transformer":
      model = self.Stage2TransformerModel(
        **common,
        d_model=int(model_cfg.get("d_model", 512)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", window_size)),
      )
    elif arch == "mlp":
      model = self.Stage2MLPModel(
        **common,
        hidden_dim=int(model_cfg.get("mlp_hidden", 512)),
        num_layers=int(model_cfg.get("mlp_layers", 4)),
      )
    else:
      msg = f"Unsupported Stage 2 architecture {arch!r}"
      raise ValueError(msg)
    model.load_state_dict(ckpt["model"])
    model.to(self.device)
    model.eval()
    model.requires_grad_(False)
    return model, state_dim, cond_dim

  def _schedule_from_ckpt(self, ckpt: dict[str, Any]):
    schedule = ckpt["schedule"]
    cfg = self.DiffusionConfig(
      timesteps=int(schedule["timesteps"]),
      beta_start=float(schedule["beta_start"]),
      beta_end=float(schedule["beta_end"]),
    )
    return self.DiffusionSchedule(cfg).to(self.device)

  def normalize_stage2_state(self, x0_raw: torch.Tensor) -> torch.Tensor:
    x0_raw = x0_raw.to(device=self.device, dtype=torch.float32)
    if x0_raw.shape[-1] != ROBOT_OBJECT_STATE_DIM:
      msg = f"Stage 2 x0 must be 47D, got {tuple(x0_raw.shape)}"
      raise ValueError(msg)
    return _normalize(x0_raw, self.stage2_state_mean, self.stage2_state_std)

  @torch.inference_mode()
  def sample_stage1_hands(
    self,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    object_pose: torch.Tensor,
    goal_raw: torch.Tensor,
  ) -> torch.Tensor:
    bps = bps_encoding.to(device=self.device, dtype=torch.float32)
    centroid = object_centroid.to(device=self.device, dtype=torch.float32)
    object_pose = object_pose.to(device=self.device, dtype=torch.float32)
    goal_raw = goal_raw.to(device=self.device, dtype=torch.float32)
    if bps.shape[-1] != STATIC_BPS_DIM:
      msg = f"Stage 1 BPS must be 3072D, got {tuple(bps.shape)}"
      raise ValueError(msg)
    if centroid.shape[-1] != 3:
      msg = f"Stage 1 object centroid must be 3D, got {tuple(centroid.shape)}"
      raise ValueError(msg)
    if object_pose.shape[-1] != OBJECT_POSE_DIM:
      msg = f"Stage 1 object_pose must be 9D, got {tuple(object_pose.shape)}"
      raise ValueError(msg)
    if goal_raw.shape[-1] != OBJECT_POSE_DIM:
      msg = f"Stage 1 goal must be 9D, got {tuple(goal_raw.shape)}"
      raise ValueError(msg)
    bsz, horizon, _ = centroid.shape
    x = torch.randn(bsz, horizon, HAND_DIM, device=self.device, dtype=torch.float32)
    goal = _normalize(goal_raw, self.stage1_goal_mean, self.stage1_goal_std)
    for step in reversed(range(int(self.stage1_schedule.timesteps))):
      t = torch.full((bsz,), step, device=self.device, dtype=torch.long)
      x0_pred = self.stage1_model(
        x,
        t,
        bps,
        centroid,
        object_pose=object_pose,
        global_cond=goal,
      )
      if step > 0:
        alpha_bar_t = self.stage1_schedule.alpha_bar[step]
        alpha_bar_prev = self.stage1_schedule.alpha_bar[step - 1]
        alpha_t = self.stage1_schedule.alpha[step]
        mean = (
          torch.sqrt(alpha_bar_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred
          + torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x
        )
        x = mean + torch.sqrt(self.stage1_schedule.beta[step]) * torch.randn_like(x)
      else:
        x = x0_pred
    return _denormalize(x, self.stage1_hand_mean, self.stage1_hand_std)

  def rectify_hands(
    self,
    hands_raw: torch.Tensor,
    object_verts: torch.Tensor | None,
    object_rotations: torch.Tensor | None,
    contact_labels: torch.Tensor | None = None,
    diagnostic_skip_rectification: bool = False,
  ) -> tuple[torch.Tensor, dict[str, Any]]:
    hands = hands_raw.to(device=self.device, dtype=torch.float32)
    if diagnostic_skip_rectification:
      return hands, {"rectification": "diagnostic_skip_requested"}
    if object_verts is None or object_rotations is None:
      msg = (
        "Missing object_verts/object_rotations; hand contact rectification is "
        "required for the main object-goal reward path."
      )
      raise RuntimeError(msg)
    if hands.ndim != 3 or hands.shape[-1] != HAND_DIM:
      msg = f"hands_raw must have shape (B, T, 6), got {tuple(hands.shape)}"
      raise ValueError(msg)

    verts = object_verts.to(device=self.device, dtype=torch.float32)
    rots = object_rotations.to(device=self.device, dtype=torch.float32)
    if verts.ndim == 3:
      verts = verts.unsqueeze(0)
    if rots.ndim == 3:
      rots = rots.unsqueeze(0)
    if verts.shape[0] != hands.shape[0] or verts.shape[1] != hands.shape[1]:
      msg = f"object_verts shape {tuple(verts.shape)} incompatible with hands {tuple(hands.shape)}"
      raise ValueError(msg)
    if rots.shape[0] != hands.shape[0] or rots.shape[1] != hands.shape[1]:
      msg = f"object_rotations shape {tuple(rots.shape)} incompatible with hands {tuple(hands.shape)}"
      raise ValueError(msg)

    labels_np = None
    if contact_labels is not None:
      labels = contact_labels.detach().cpu().numpy()
      labels_np = labels[None] if labels.ndim == 1 else labels

    hands_np = hands.detach().cpu().numpy()
    verts_np = verts.detach().cpu().numpy()
    rots_np = rots.detach().cpu().numpy()
    rectified = []
    per_batch = []
    for bidx in range(hands.shape[0]):
      batch_labels = None if labels_np is None else labels_np[bidx]
      hands_rect, metadata = self.contact_processor.process(
        hands_np[bidx],
        verts_np[bidx],
        rots_np[bidx],
        contact_labels=batch_labels,
      )
      rectified.append(hands_rect)
      per_batch.append(metadata)
    metadata_out: dict[str, Any] = {"per_batch": per_batch, "batch_size": hands.shape[0]}
    if len(per_batch) == 1:
      metadata_out.update(per_batch[0])
    rectified_np = np.stack(rectified, axis=0).astype(np.float32)
    return torch.from_numpy(rectified_np).to(device=self.device), metadata_out

  def build_stage2_condition(
    self,
    hands_rectified: torch.Tensor,
    static_bps_context: torch.Tensor,
    goal_raw: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    hands = hands_rectified.to(device=self.device, dtype=torch.float32)
    goal_raw = goal_raw.to(device=self.device, dtype=torch.float32)
    if hands.ndim != 3 or hands.shape[-1] != HAND_DIM:
      msg = f"hands_rectified must have shape (B, T, 6), got {tuple(hands.shape)}"
      raise ValueError(msg)
    if self.stage2_normalize_hands:
      hands = _normalize(hands, self.stage2_hand_mean, self.stage2_hand_std)
    bsz, horizon, _ = hands.shape

    bps = static_bps_context.to(device=self.device, dtype=torch.float32)
    if bps.ndim == 2 and bps.shape == (bsz, STATIC_BPS_DIM):
      bps = bps[:, None, :].expand(-1, horizon, -1)
    elif bps.ndim == 2 and bsz == 1 and bps.shape == (horizon, STATIC_BPS_DIM):
      bps = bps.unsqueeze(0)
    elif bps.ndim == 3 and bps.shape[:2] == (bsz, horizon):
      pass
    else:
      msg = (
        "static_bps_context must be (B, 3072), (T, 3072) for B=1, or "
        f"(B, T, 3072); got {tuple(bps.shape)}"
      )
      raise ValueError(msg)
    if bps.shape[-1] != STATIC_BPS_DIM:
      msg = f"static BPS context must be 3072D, got {tuple(bps.shape)}"
      raise ValueError(msg)

    cond = torch.cat([hands, bps], dim=-1)
    if cond.shape[-1] != self.stage2_cond_dim:
      msg = f"Stage 2 condition dim mismatch: built {cond.shape[-1]}, expected {self.stage2_cond_dim}"
      raise ValueError(msg)
    goal = _normalize(goal_raw, self.stage2_goal_mean, self.stage2_goal_std)
    return cond, goal

  @torch.inference_mode()
  def compute_sds_reward(
    self,
    x0_raw: torch.Tensor,
    cond: torch.Tensor,
    goal: torch.Tensor,
    fixed_timesteps: tuple[int, ...] | None = None,
    ws: float | None = None,
    normalizer: Any | None = None,
    normalize: bool = True,
  ) -> tuple[torch.Tensor, dict[str, Any]]:
    x0 = self.normalize_stage2_state(x0_raw)
    cond = cond.to(device=self.device, dtype=torch.float32)
    goal = goal.to(device=self.device, dtype=torch.float32)
    if cond.shape[:2] != x0.shape[:2] or cond.shape[-1] != STAGE2_COND_DIM:
      msg = f"Stage 2 cond shape {tuple(cond.shape)} incompatible with x0 {tuple(x0.shape)}"
      raise ValueError(msg)
    if goal.shape != (x0.shape[0], OBJECT_POSE_DIM):
      msg = f"Stage 2 goal must have shape {(x0.shape[0], OBJECT_POSE_DIM)}, got {tuple(goal.shape)}"
      raise ValueError(msg)

    timesteps = tuple(self.fixed_timesteps if fixed_timesteps is None else fixed_timesteps)
    if not timesteps:
      raise ValueError("fixed_timesteps must be non-empty")
    weight = self.ws if ws is None else float(ws)
    bsz = x0.shape[0]
    total_err = torch.zeros(bsz, device=self.device)
    total_raw = torch.zeros(bsz, device=self.device)
    eps_norm = torch.zeros(bsz, device=self.device)
    eps_hat_norm = torch.zeros(bsz, device=self.device)

    for t_scalar in timesteps:
      if not 0 <= int(t_scalar) < int(self.stage2_schedule.timesteps):
        msg = (
          f"fixed_timestep {t_scalar} out of range "
          f"[0, {int(self.stage2_schedule.timesteps)})"
        )
        raise ValueError(msg)
      t = torch.full((bsz,), int(t_scalar), dtype=torch.long, device=self.device)
      noise = torch.randn_like(x0)
      x_t = self.stage2_schedule.q_sample(x0, t, noise)
      pred = self.stage2_model(x_t, t, cond, global_cond=goal)
      if self.stage2_prediction_type == "epsilon":
        eps_hat = pred
      elif self.stage2_prediction_type == "x0":
        alpha_bar = self.stage2_schedule.alpha_bar[t].view(-1, 1, 1)
        eps_hat = (x_t - torch.sqrt(alpha_bar) * pred) / torch.sqrt(
          (1.0 - alpha_bar).clamp_min(1e-8)
        )
      else:
        msg = f"Unsupported Stage 2 prediction_type {self.stage2_prediction_type!r}"
        raise ValueError(msg)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize and normalizer is not None:
        total_err += normalizer.update_and_normalize(int(t_scalar), mse_per_env)
      else:
        total_err += mse_per_env
      eps_norm += _tensor_norm_per_env(noise)
      eps_hat_norm += _tensor_norm_per_env(eps_hat)

    raw_err = total_raw / len(timesteps)
    err = total_err / len(timesteps)
    reward = torch.exp(-err * weight)
    diagnostics: dict[str, Any] = {
      "sds_error": raw_err,
      "sds_error_normalized": err,
      "r_smp": reward,
      "timesteps": tuple(int(t) for t in timesteps),
      "epsilon_norm": eps_norm / len(timesteps),
      "epsilon_hat_norm": eps_hat_norm / len(timesteps),
      "x0_norm": _tensor_norm_per_env(x0),
      "condition_norm": _tensor_norm_per_env(cond),
      "goal_norm": torch.linalg.vector_norm(goal, dim=1),
      "prediction_type": self.stage2_prediction_type,
    }
    return reward, diagnostics

  def compute_reward_from_raw_inputs(
    self,
    x0_raw: torch.Tensor,
    bps_encoding: torch.Tensor,
    object_centroid: torch.Tensor,
    object_pose: torch.Tensor,
    static_bps_context: torch.Tensor,
    goal_raw: torch.Tensor,
    object_verts: torch.Tensor | None,
    object_rotations: torch.Tensor | None,
    contact_labels: torch.Tensor | None = None,
    fixed_timesteps: tuple[int, ...] | None = None,
    ws: float | None = None,
    normalizer: Any | None = None,
    normalize: bool = True,
    diagnostic_skip_rectification: bool = False,
  ) -> tuple[torch.Tensor, dict[str, Any]]:
    hands_raw = self.sample_stage1_hands(
      bps_encoding=bps_encoding,
      object_centroid=object_centroid,
      object_pose=object_pose,
      goal_raw=goal_raw,
    )
    hands_rect, contact_metadata = self.rectify_hands(
      hands_raw,
      object_verts=object_verts,
      object_rotations=object_rotations,
      contact_labels=contact_labels,
      diagnostic_skip_rectification=diagnostic_skip_rectification,
    )
    cond, goal = self.build_stage2_condition(hands_rect, static_bps_context, goal_raw)
    reward, diagnostics = self.compute_sds_reward(
      x0_raw=x0_raw,
      cond=cond,
      goal=goal,
      fixed_timesteps=fixed_timesteps,
      ws=ws,
      normalizer=normalizer,
      normalize=normalize,
    )
    diagnostics["contact_metadata"] = contact_metadata
    diagnostics["hands_raw_norm"] = _tensor_norm_per_env(hands_raw)
    diagnostics["hands_rectified_norm"] = _tensor_norm_per_env(hands_rect)
    return reward, diagnostics
