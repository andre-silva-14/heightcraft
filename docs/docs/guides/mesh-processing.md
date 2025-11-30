---
sidebar_position: 3
---

# Mesh Processing

Convert 3D meshes (`.obj`, `.stl`, `.ply`, `.glb`) into heightmaps. This is useful for baking terrain geometry from modeling software like Blender or ZBrush.

## Usage

```bash
heightcraft terrain.obj --output_path heightmap.png
```

## Large Models

For very large meshes that don't fit in memory, Heightcraft offers a `--large_model` mode (though standard mode is sufficient for most use cases).

```bash
heightcraft huge_terrain.obj --large_model
```

## Sampling

You can control the sampling density to trade off between quality and processing time.

```bash
# Increase samples for higher quality
heightcraft terrain.obj --num_samples 5000000
```
