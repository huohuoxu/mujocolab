from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxRandomGridTerrainCfg,
  BoxTiltedGridTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  SubTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BoxGapTerrainCfg(SubTerrainCfg):
  gap_size_range: tuple[float, float] = (0.2, 1.0)
  runway_width_range: tuple[float, float] = (1.0, 2.0)
  floor_depth: float = 2.0
  platform_width: float = 1.0

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
    before_len = max(0.05, center_x - gap_size / 2)
    after_len = max(0.05, self.size[0] - center_x - gap_size / 2)
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

    left = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(before_len / 2, half_width, self.floor_depth / 2),
      pos=(before_len / 2, center_y, -self.floor_depth / 2),
    )
    geoms.append(left)
    colors.append((0.25, 0.45, 0.75, 1.0))

    right = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(after_len / 2, half_width, self.floor_depth / 2),
      pos=(self.size[0] - after_len / 2, center_y, -self.floor_depth / 2),
    )
    geoms.append(right)
    colors.append((0.25, 0.45, 0.75, 1.0))

    side_width = max(0.05, (self.size[1] - runway_width) / 2)
    for y in (side_width / 2, self.size[1] - side_width / 2):
      side = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.size[0] / 2, side_width / 2, self.floor_depth / 2),
        pos=(self.size[0] / 2, y, -self.floor_depth / 2),
      )
      geoms.append(side)
      colors.append((0.18, 0.18, 0.18, 1.0))

    origin = np.array([self.platform_width, center_y, 0.05])
    return TerrainOutput(
      origin=origin,
      geometries=[
        TerrainGeometry(geom=geom, color=color)
        for geom, color in zip(geoms, colors, strict=True)
      ],
    )


@dataclass(kw_only=True)
class BoxPitClimbTerrainCfg(SubTerrainCfg):
  block_height_range: tuple[float, float] = (0.05, 0.6)
  block_length_range: tuple[float, float] = (1.0, 1.2)
  pit_depth_range: tuple[float, float] = (0.0, 0.6)
  pit_length_range: tuple[float, float] = (0.9, 1.4)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    block_height = self.block_height_range[0] + difficulty * (
      self.block_height_range[1] - self.block_height_range[0]
    )
    pit_depth = self.pit_depth_range[0] + difficulty * (
      self.pit_depth_range[1] - self.pit_depth_range[0]
    )
    length_a = rng.uniform(*self.block_length_range)
    length_b = rng.uniform(*self.block_length_range)
    pit_length = rng.uniform(*self.pit_length_range)
    geoms = []
    colors = []

    pit_floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(self.size[0] / 2, self.size[1] / 2, -pit_depth - 0.05),
    )
    geoms.append(pit_floor)
    colors.append((0.12, 0.12, 0.12, 1.0))

    pit_center = self.size[0] / 2
    pit_start = pit_center - pit_length / 2
    pit_end = pit_center + pit_length / 2
    flat_segments = (
      (0.0, max(0.05, pit_start)),
      (min(self.size[0] - 0.05, pit_end), self.size[0]),
    )
    for start, end in flat_segments:
      length = max(0.05, end - start)
      plate = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, self.size[1] / 2, 0.05),
        pos=(start + length / 2, self.size[1] / 2, -0.05),
      )
      geoms.append(plate)
      colors.append((0.35, 0.35, 0.35, 1.0))

    for start, length in ((1.0, length_a), (6.0, length_b)):
      block = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(length / 2, self.size[1] / 2, block_height / 2),
        pos=(start + length / 2, self.size[1] / 2, block_height / 2),
      )
      geoms.append(block)
      colors.append((0.7, 0.45, 0.25, 1.0))

    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.05])
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
    for x in (2.0, 5.5):
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


WMP_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=25.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  sub_terrains={
    "rough_slope": HfPyramidSlopedTerrainCfg(
      proportion=0.05,
      slope_range=(0.0, 0.4),
      platform_width=3.0,
      border_width=0.25,
    ),
    "stairs_up": BoxPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.05, 0.23),
      step_width=0.32,
      platform_width=3.0,
      border_width=0.25,
    ),
    "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.05, 0.23),
      step_width=0.32,
      platform_width=3.0,
      border_width=0.25,
    ),
    "gap": BoxGapTerrainCfg(proportion=0.25),
    "pit_climb": BoxPitClimbTerrainCfg(proportion=0.25),
    "tilt": BoxTiltedGridTerrainCfg(
      proportion=0.05,
      grid_width=1.0,
      tilt_range_deg=20.0,
      height_range=0.25,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=0.5,
    ),
    "crawl": BoxCrawlTerrainCfg(proportion=0.05),
    "rough_flat": HfRandomUniformTerrainCfg(
      proportion=0.05,
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
