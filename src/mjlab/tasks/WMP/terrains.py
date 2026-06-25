from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomGridTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

_WMP_EDGE_MASK_KEY = "wmp_x_edge_mask"
_WMP_EDGE_RESOLUTION_KEY = "wmp_edge_resolution"
_WMP_EDGE_RESOLUTION = 0.1


def _empty_edge_mask(size: tuple[float, float]) -> np.ndarray:
  shape = (
    max(1, int(round(size[0] / _WMP_EDGE_RESOLUTION))),
    max(1, int(round(size[1] / _WMP_EDGE_RESOLUTION))),
  )
  return np.zeros(shape, dtype=np.bool_)


def _mark_x_edge(
  mask: np.ndarray,
  x: float,
  y_range: tuple[float, float],
  *,
  size: tuple[float, float],
  dilation_cells: int = 1,
) -> None:
  nx, ny = mask.shape
  x_id = int(np.clip(np.floor(x / size[0] * nx), 0, nx - 1))
  y0 = int(np.clip(np.floor(y_range[0] / size[1] * ny), 0, ny - 1))
  y1 = int(np.clip(np.ceil(y_range[1] / size[1] * ny), y0 + 1, ny))
  x0 = max(0, x_id - dilation_cells)
  x1 = min(nx, x_id + dilation_cells + 1)
  mask[x0:x1, y0:y1] = True


def _edge_metadata(mask: np.ndarray) -> dict[str, np.ndarray]:
  return {
    _WMP_EDGE_MASK_KEY: mask,
    _WMP_EDGE_RESOLUTION_KEY: np.asarray(_WMP_EDGE_RESOLUTION, dtype=np.float32),
  }


@dataclass(kw_only=True)
class BoxGapTerrainCfg(SubTerrainCfg):
  gap_size_range: tuple[float, float] = (0.0, 1.0)
  runway_width_range: tuple[float, float] = (1.0, 2.0)
  floor_depth: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    gap_size = self.gap_size_range[0] + difficulty * (
      self.gap_size_range[1] - self.gap_size_range[0]
    )
    runway_width = rng.uniform(*self.runway_width_range)
    center_x = self.size[0] / 2
    center_y = self.size[1] / 2
    half_width = runway_width / 2
    geoms = []
    colors = []

    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth - 0.05),
    )
    geoms.append(floor)
    colors.append((0.08, 0.08, 0.08, 1.0))

    x1 = center_x - 1.0
    x2 = center_x + 2.0
    x3 = x1 - gap_size
    x4 = x2 + gap_size
    x5 = gap_size
    runway_segments = ((x5, x3), (x1, x2), (x4, self.size[0]))
    edge_mask = _empty_edge_mask(self.size)
    runway_y_range = (center_y - half_width, center_y + half_width)
    for x in (x3, x1, x2, x4):
      _mark_x_edge(edge_mask, x, runway_y_range, size=self.size)
    for start, end in runway_segments:
      length = max(0.0, end - start)
      if length <= 0.05:
        continue
      segment = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, half_width, self.floor_depth / 2),
        pos=(start + length / 2, center_y, -self.floor_depth / 2),
      )
      geoms.append(segment)
      colors.append((0.25, 0.45, 0.75, 1.0))

    origin = np.array([center_x, center_y, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=geom, color=color)
        for geom, color in zip(geoms, colors, strict=True)
      ],
      metadata=_edge_metadata(edge_mask),
    )


@dataclass(kw_only=True)
class BoxPitClimbTerrainCfg(SubTerrainCfg):
  block_height_range: tuple[float, float] = (0.0, 0.6)
  block_length_range: tuple[float, float] = (1.0, 1.2)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    block_height = self.block_height_range[0] + difficulty * (
      self.block_height_range[1] - self.block_height_range[0]
    )
    length_a = rng.uniform(*self.block_length_range)
    length_b = rng.uniform(*self.block_length_range)
    geoms = []
    colors = []

    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(self.size[0] / 2, self.size[1] / 2, -0.05),
    )
    geoms.append(floor)
    colors.append((0.35, 0.35, 0.35, 1.0))

    edge_mask = _empty_edge_mask(self.size)
    for start, length in ((1.0, length_a), (6.0, length_b)):
      _mark_x_edge(edge_mask, start, (0.0, self.size[1]), size=self.size)
      _mark_x_edge(edge_mask, start + length, (0.0, self.size[1]), size=self.size)
      if block_height <= 0.0:
        continue
      block = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, self.size[1] / 2, block_height / 2),
        pos=(start + length / 2, self.size[1] / 2, block_height / 2),
      )
      geoms.append(block)
      colors.append((0.7, 0.45, 0.25, 1.0))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=geom, color=color)
        for geom, color in zip(geoms, colors, strict=True)
      ],
      metadata=_edge_metadata(edge_mask),
    )


@dataclass(kw_only=True)
class BoxWmpTiltPassageTerrainCfg(SubTerrainCfg):
  corridor_width_range: tuple[float, float] = (0.32, 0.28)
  block_length_range: tuple[float, float] = (0.4, 0.8)
  block_height: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    corridor_width = self.corridor_width_range[0] + difficulty * (
      self.corridor_width_range[1] - self.corridor_width_range[0]
    )
    block_length = rng.uniform(*self.block_length_range)
    center_x = self.size[0] / 2
    center_y = self.size[1] / 2
    side_width = (self.size[1] - corridor_width) / 2
    geoms = []
    colors = []

    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(center_x, center_y, -0.05),
    )
    geoms.append(floor)
    colors.append((0.35, 0.35, 0.35, 1.0))

    y_centers = (
      center_y - corridor_width / 2 - side_width / 2,
      center_y + corridor_width / 2 + side_width / 2,
    )
    x_centers = (
      center_x + 2.0 + block_length / 2,
      center_x - 2.0 - block_length / 2,
    )
    for x_center in x_centers:
      for y_center in y_centers:
        block = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(block_length / 2, side_width / 2, self.block_height / 2),
          pos=(x_center, y_center, self.block_height / 2),
        )
        geoms.append(block)
        colors.append((0.55, 0.35, 0.85, 1.0))

    origin = np.array([center_x, center_y, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=geom, color=color)
        for geom, color in zip(geoms, colors, strict=True)
      ],
    )


@dataclass(kw_only=True)
class BoxCrawlTerrainCfg(SubTerrainCfg):
  bar_height_range: tuple[float, float] = (0.35, 0.2)
  bar_width_range: tuple[float, float] = (0.2, 0.4)
  floor_depth: float = 0.2

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    bar_height = self.bar_height_range[0] + difficulty * (
      self.bar_height_range[1] - self.bar_height_range[0]
    )
    bar_width = rng.uniform(*self.bar_width_range)
    geoms = []
    colors = []
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.floor_depth / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.floor_depth / 2),
    )
    geoms.append(floor)
    colors.append((0.35, 0.35, 0.35, 1.0))
    center_x = self.size[0] / 2
    for x in (center_x - 2.0 - bar_width / 2, center_x + 2.0 + bar_width / 2):
      bar = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(bar_width / 2, self.size[1] / 2, 0.5),
        pos=(x, self.size[1] / 2, bar_height + 0.5),
      )
      geoms.append(bar)
      colors.append((0.65, 0.25, 0.25, 1.0))
    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.05])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=geom, color=color)
        for geom, color in zip(geoms, colors, strict=True)
      ],
    )


def _terrain_columns(
  prefix: str,
  cfg: SubTerrainCfg,
  count: int,
) -> dict[str, SubTerrainCfg]:
  return {f"{prefix}_{idx}": replace(cfg, proportion=1.0) for idx in range(count)}


WMP_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=25.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  sub_terrains={
    "rough_slope_0": HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.4),
      platform_width=3.0,
      border_width=0.25,
    ),
    **_terrain_columns(
      "stairs_up",
      BoxPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.23),
        step_width=0.32,
        platform_width=3.0,
        border_width=0.25,
      ),
      3,
    ),
    **_terrain_columns(
      "stairs_down",
      BoxInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.23),
        step_width=0.32,
        platform_width=3.0,
        border_width=0.25,
      ),
      3,
    ),
    **_terrain_columns("gap", BoxGapTerrainCfg(proportion=1.0), 5),
    **_terrain_columns("pit_climb", BoxPitClimbTerrainCfg(proportion=1.0), 5),
    "tilt_0": BoxWmpTiltPassageTerrainCfg(proportion=1.0),
    "crawl_0": BoxCrawlTerrainCfg(proportion=1.0),
    "rough_flat_0": HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


WMP_PLAY_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=5,
  num_cols=8,
  curriculum=True,
  sub_terrains={
    "flat": BoxFlatTerrainCfg(proportion=0.1),
    "random_grid": BoxRandomGridTerrainCfg(
      proportion=0.2,
      grid_width=0.4,
      grid_height_range=(0.0, 0.25),
      platform_width=1.0,
    ),
    **WMP_ROUGH_TERRAINS_CFG.sub_terrains,
  },
  add_lights=True,
)
