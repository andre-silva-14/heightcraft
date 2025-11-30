---
slug: training-and-upscaling
title: Training Custom AI Models for Heightmap Upscaling
authors: [heightcraft-team]
tags: [ai, upscaling, tutorial, heightmap]
---

In this post, we'll dive into one of Heightcraft's most powerful features: **Training custom AI upscaling models**.

While Heightcraft comes with a general-purpose model, training on your specific type of terrain (e.g., rocky mountains, sand dunes, or urban landscapes) can yield significantly better results.

<!--truncate-->

## Why Train a Custom Model?

The default model is trained on a diverse dataset of DEMs (Digital Elevation Models). However, terrain features vary wildly. A model trained on Swiss Alps data might not perform well on Martian crater data.

By training a custom model, you teach the AI to hallucinate details specific to your target biome.

## Step 1: Prepare Your Dataset

You need a set of high-resolution heightmaps. These will be the "ground truth".
- **Format**: `.tiff` or `.png` (16-bit or 32-bit preferred).
- **Structure**: Place them all in a single directory.

```bash
mkdir my_dataset
cp /path/to/high_res_tiffs/*.tiff my_dataset/
```

## Step 2: Run Training

Use the `--train` flag to start the training process. Heightcraft handles data augmentation and splitting automatically.

```bash
heightcraft --train \
  --dataset_path ./my_dataset \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained_model my_custom_model.h5
```

- `--epochs`: How many times to iterate over the dataset. 50-100 is usually good.
- `--pretrained_model`: The output path for your new model.

## Step 3: Upscale with Your Model

Once training is complete, you can use your new model to upscale low-resolution inputs.

```bash
heightcraft low_res_input.png \
  --upscale \
  --pretrained_model my_custom_model.h5 \
  --upscale_factor 4 \
  --output_path high_res_output.png
```

## Results

You should see sharper edges and more plausible details compared to bicubic interpolation or the generic model.

Happy upscaling! 🚀
