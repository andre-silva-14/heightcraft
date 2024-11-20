# File: height_map_generator_gpu_sampling.py
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from time import time

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


def generate_height_map(mesh, target_resolution=(256, 256), use_gpu=True, num_samples=10000):
    """Generates a height map (grayscale image) from a 3D model using GPU acceleration."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Sample points on the GPU
    sampled_points = sample_points_on_gpu(mesh, num_samples, device)
    points_2d = sampled_points[:, :2]  # Project to the X-Y plane
    z_values = sampled_points[:, 2]    # Extract Z-coordinates for height mapping

    # Calculate bounds for normalization
    logging.info("Calculating bounds for normalization...")
    min_x, max_x = points_2d[:, 0].min().item(), points_2d[:, 0].max().item()
    min_y, max_y = points_2d[:, 1].min().item(), points_2d[:, 1].max().item()
    min_z, max_z = z_values.min().item(), z_values.max().item()
    width, height = target_resolution

    # Normalize coordinates to the image resolution
    points_2d[:, 0] = (points_2d[:, 0] - min_x) / (max_x - min_x) * (width - 1)
    points_2d[:, 1] = (points_2d[:, 1] - min_y) / (max_y - min_y) * (height - 1)
    z_normalized = (z_values - min_z) / (max_z - min_z) * 255

    # Initialize an empty height map on the GPU
    height_map = torch.zeros((height, width), dtype=torch.float32, device=device)

    # Update height map using tensor operations
    logging.info("Updating height map...")
    x_coords = points_2d[:, 0].long().clamp(0, width - 1)
    y_coords = points_2d[:, 1].long().clamp(0, height - 1)
    height_map.index_put_((y_coords, x_coords), z_normalized, accumulate=True)

    # Transfer the height map back to the CPU for visualization and saving
    height_map_cpu = height_map.cpu().numpy()
    logging.info("Height map generation complete.")
    return height_map_cpu


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


def main(file_path, output_path="height_map.png", target_resolution=(256, 256), use_gpu=True):
    """Main function to load a 3D model, generate a height map, and save it."""
    try:
        start_time = time()
        logging.info("Starting the height map generation process...")
        mesh = load_3d_model(file_path)
        height_map = generate_height_map(mesh, target_resolution, use_gpu=use_gpu)
        display_image(height_map)  # Show the height map
        save_height_map(height_map, output_path)  # Save the height map
        logging.info(f"Process completed in {time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    file_path = "path/to/your/model.obj"  # Adjust the path to your 3D model
    output_path = "height_map.png"        # Output file for the height map
    use_gpu = True                        # Enable or disable GPU acceleration
    main(file_path, output_path, target_resolution=(256, 256), use_gpu=use_gpu)
