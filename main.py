import trimesh
import numpy as np
import matplotlib.pyplot as plt

def load_3d_model(file_path):
    """Loads the 3D model from the given file path."""
    mesh = trimesh.load_mesh(file_path)
    
    # Ensure the model is centered at the origin
    mesh.apply_translation(-mesh.centroid)  # Translate to center the model
    
    # Align the model so that it's upright (along Z axis)
    # Find the principal axes of the model (using PCA)
    _, rotation = mesh.principal_inertia_axes()
    mesh.apply_transform(rotation)  # Apply the rotation to align the model
    
    return mesh

def generate_topology_image(mesh, target_resolution=(256, 256)):
    """Generates a topology image from a 3D model, preserving aspect ratio."""
    # Sample points on the surface of the 3D model
    points = mesh.sample(10000)  # Sample 10,000 points on the surface

    # Project 3D points onto a 2D plane (for simplicity, we use the X-Y plane)
    points_2d = points[:, :2]  # Only take x and y coordinates

    # Find the extents of the 3D model in the X and Y directions
    min_x, max_x = points_2d[:, 0].min(), points_2d[:, 0].max()
    min_y, max_y = points_2d[:, 1].min(), points_2d[:, 1].max()
    
    # Calculate the aspect ratio of the original model in the X-Y plane
    model_width = max_x - min_x
    model_height = max_y - min_y
    model_aspect_ratio = model_width / model_height
    
    # Calculate the new image dimensions based on the target resolution and model aspect ratio
    target_width, target_height = target_resolution
    target_image_aspect_ratio = target_width / target_height

    if model_aspect_ratio > target_image_aspect_ratio:
        # The model is wider than the target, so adjust width
        new_width = target_width
        new_height = int(target_width / model_aspect_ratio)
    else:
        # The model is taller than the target, so adjust height
        new_height = target_height
        new_width = int(target_height * model_aspect_ratio)

    # Make sure the resolution does not exceed the target resolution
    new_width = min(new_width, target_width)
    new_height = min(new_height, target_height)

    # Normalize the points to fit within the new resolution
    points_2d[:, 0] = (points_2d[:, 0] - min_x) / (max_x - min_x) * (new_width - 1)
    points_2d[:, 1] = (points_2d[:, 1] - min_y) / (max_y - min_y) * (new_height - 1)
    
    # Create an empty grayscale image
    topology_image = np.zeros((new_height, new_width))
    
    # Calculate grayscale values based on the distance from the center of the model
    center = np.mean(points, axis=0)  # Use the centroid as the center
    for point in points:
        # Calculate Euclidean distance from the point to the center
        distance = np.linalg.norm(point - center)
        
        # Map the distance to a grayscale value (0 to 100)
        grayscale_value = min(100, distance / np.max(distance) * 100)
        
        # Map the 2D point coordinates to pixel indices
        x, y = int(points_2d[points.tolist().index(point), 0]), int(points_2d[points.tolist().index(point), 1])
        
        # Set the pixel value in the topology image
        topology_image[y, x] = grayscale_value
    
    return topology_image

def display_image(image):
    """Displays the topology image using matplotlib."""
    plt.imshow(image, cmap='gray', vmin=0, vmax=100)
    plt.axis('off')  # Hide axes
    plt.show()

def main(file_path):
    """Main function to run the program."""
    mesh = load_3d_model(file_path)
    topology_image = generate_topology_image(mesh)
    display_image(topology_image)

if __name__ == "__main__":
    # Example: Replace with the path to your 3D model file
    file_path = "path/to/your/model.obj"  # Adjust the path to your model
    main(file_path)
