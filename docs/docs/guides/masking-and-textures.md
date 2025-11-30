---
sidebar_position: 4
---

# Masking & Textures

Heightcraft provides powerful tools for generating auxiliary maps that are essential for terrain texturing and game engine integration. These features allow you to mask water areas and generate slope and curvature maps for advanced material blending.

## Sea Level Masking

The `--sea_level` option allows you to "flood" the terrain up to a specific height. Any terrain below this Z-value will be flattened to that level, simulating a water surface.

### Usage

```bash
heightcraft terrain.obj --sea_level 10.5
```

### Outputs

- **Modified Height Map**: The main height map will have all values below `10.5` flattened to `10.5`.
- **Water Mask**: A separate file `height_map_water_mask.png` will be generated.
    - **White (1.0)**: Represents water (areas that were below the sea level).
    - **Black (0.0)**: Represents land.

This mask is perfect for defining where to render water shaders or for physics layers in your game engine.

## Texture Maps

Heightcraft can generate additional maps derived from the terrain geometry to help with texturing.

### Slope Map

The slope map represents the steepness of the terrain at each point.

```bash
heightcraft terrain.obj --slope_map
```

- **Output**: `height_map_slope_map.png`
- **Values**: Normalized from 0.0 (flat) to 1.0 (steepest).
- **Use Case**: Use this to blend between grass (flat) and rock/cliff (steep) textures.

### Curvature Map

The curvature map represents the convexity or concavity of the terrain.

```bash
heightcraft terrain.obj --curvature_map
```

- **Output**: `height_map_curvature_map.png`
- **Values**: Normalized around 0.5 (flat).
    - **> 0.5**: Convex (peaks, ridges).
    - **< 0.5**: Concave (valleys, crevices).
- **Use Case**: Use this to add wear/erosion to peaks or accumulate dirt/snow in crevices.

## Combined Workflow

You can combine all these options for a complete terrain generation pipeline:

```bash
heightcraft terrain.obj \
  --sea_level 5.0 \
  --slope_map \
  --curvature_map \
  --output_path my_terrain.png
```

This command will produce:
1. `my_terrain.png`: The height map with a flat sea floor.
2. `my_terrain_water_mask.png`: Mask for the water.
3. `my_terrain_slope_map.png`: Slope data for texturing.
4. `my_terrain_curvature_map.png`: Curvature data for texturing.
