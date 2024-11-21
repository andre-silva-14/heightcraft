# File: height_map_generator_dynamic_resolution.py
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from time import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def load_3d_model(file_path):
    """Loads and preprocesses the 3D model from the given file path."""
    try:
        start_time = time()
        logging.info(f"Loading 3D model from {file_path}...")
        mesh = trimesh.load_mesh(file_path)
        logging.info(f"Model loaded successfully in {time() - start_time:.2f} seconds.")
    except Exception as e:
        raise ValueError(f"Failed to load the 3D model: {e}")
    
    # Ensure the model is centered at the origin
    logging.info("Centering the model at the origin...")
    mesh.apply_translation(-mesh.centroid)
    
    # Align the model so that its largest inertia axis is aligned with Z-axis
    logging.info("Aligning the model using principal inertia axes...")
    try:
        _, rotation = mesh.principal_inertia_axes()
        mesh.apply_transform(rotation)
        logging.info("Model aligned successfully.")
    except Exception:
        raise RuntimeError("Failed to calculate principal inertia axes.")
    
    return mesh


def calculate_dynamic_resolution(mesh, max_resolution):
    """Calculates the dynamic resolution based on the model's aspect ratio."""
    logging.info("Calculating dynamic resolution...")
    
    # Get the model's bounding box in the X-Y plane
    bounds = mesh.bounds
    model_width = bounds[1][0] - bounds[0][0]  # X range
    model_height = bounds[1][1] - bounds[0][1]  # Y range
    aspect_ratio = model_width / model_height if model_height != 0 else 1.0

    if aspect_ratio >= 1.0:
        # Wider than tall
        width = max_resolution
        height = int(width / aspect_ratio)
    else:
        # Taller than wide
        height = max_resolution
        width = int(height * aspect_ratio)

    logging.info(f"Dynamic resolution determined: {width}x{height}")
    return width, height


def sample_points_on_gpu(mesh, num_samples, device="cuda"):
    """Samples points uniformly from the surface of a 3D model on the GPU."""
    logging.info(f"Sampling {num_samples} points from the 3D model using GPU...")

    # Convert mesh data to tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    # Get vertex positions for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute face areas using the cross product
    face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)

    # Normalize face areas to form a probability distribution
    face_probs = face_areas / torch.sum(face_areas)

    # Sample faces based on their areas
    face_indices = torch.multinomial(face_probs, num_samples, replacement=True)

    # Random barycentric coordinates
    r1 = torch.sqrt(torch.rand(num_samples, device=device))
    r2 = torch.rand(num_samples, device=device)
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2

    # Interpolate sampled points
    sampled_faces = faces[face_indices]
    sampled_points = (
        u.unsqueeze(1) * vertices[sampled_faces[:, 0]] +
        v.unsqueeze(1) * vertices[sampled_faces[:, 1]] +
        w.unsqueeze(1) * vertices[sampled_faces[:, 2]]
    )

    logging.info(f"Sampling complete: {num_samples} points generated.")
    return sampled_points


def generate_height_map(mesh, target_resolution, use_gpu=True, num_samples=10000, num_threads=4):
    """Generates a height map (grayscale image) from a 3D model."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Sampling points
    if device.type == "cuda":
        sampled_points = sample_points_on_gpu(mesh, num_samples, device)
        points_2d = sampled_points[:, :2].cpu().numpy()  # Project to X-Y plane
        z_values = sampled_points[:, 2].cpu().numpy()   # Extract Z-coordinates
    else:
        logging.info("Sampling points from the 3D model using trimesh on CPU...")
        points = mesh.sample(num_samples)
        points_2d = points[:, :2]  # Project to the X-Y plane
        z_values = points[:, 2]    # Extract Z-coordinates

    # Calculate bounds for normalization
    min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
    min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
    min_z, max_z = z_values.min(), z_values.max()
    width, height = target_resolution
    bounds = (min_x, max_x, min_y, max_y, min_z, max_z)

    if device.type == "cuda":
        # GPU-based height map generation
        height_map = torch.zeros((height, width), dtype=torch.float32, device=device)
        x_coords = ((points_2d[:, 0] - min_x) / (max_x - min_x) * (width - 1)).astype(int)
        y_coords = ((points_2d[:, 1] - min_y) / (max_y - min_y) * (height - 1)).astype(int)
        z_normalized = ((z_values - min_z) / (max_z - min_z) * 255).astype(int)
        for x, y, z in zip(x_coords, y_coords, z_normalized):
            height_map[y, x] = max(height_map[y, x], z)
        return height_map.cpu().numpy()
    else:
        # CPU-based height map generation with parallel processing
        logging.info(f"Processing height map in parallel using {num_threads} threads...")
        points_batches = np.array_split(points_2d, num_threads)
        z_batches = np.array_split(z_values, num_threads)
        height_map = np.zeros((height, width))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    process_points_cpu, points_batches[i], z_batches[i], bounds, target_resolution
                )
                for i in range(num_threads)
            ]
            for future in as_completed(futures):
                height_map = np.maximum(height_map, future.result())
        return height_map


def display_image(image):
    """Displays the height map using matplotlib."""
    logging.info("Displaying the height map...")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Hide axes
    plt.show()


def save_height_map(image, output_path="height_map.png"):
    """Saves the generated height map to a file."""
    logging.info(f"Saving the height map to {output_path}...")
    plt.imsave(output_path, image, cmap='gray', vmin=0, vmax=255)
    logging.info(f"Height map saved successfully to {output_path}.")


def main():
    """Main function to parse arguments and run the height map generation."""
    parser = argparse.ArgumentParser(description="Generate a height map from a 3D model.")
    parser.add_argument("file_path", type=str, help="Path to the input 3D model file.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="height_map.png",
        help="Path to save the generated height map. Default: height_map.png",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=256,
        help="Maximum resolution for the longest dimension. Default: 256.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Enable GPU acceleration. Default: CPU only.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of points to sample from the 3D model surface. Default: 10,000.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing on CPU. Default: 4.",
    )

    args = parser.parse_args()

    try:
        logging.info("Starting the height map generation process...")
        mesh = load_3d_model(args.file_path)
        target_resolution = calculate_dynamic_resolution(mesh, args.max_resolution)
        height_map = generate_height_map(
            mesh,
            target_resolution=target_resolution,
            use_gpu=args.use_gpu,
            num_samples=args.num_samples,
            num_threads=args.num_threads,
        )
        display_image(height_map)  # Show the height map
        save_height_map(height_map, args.output_path)  # Save the height map
        logging.info("Height map generation completed successfully.")
    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
