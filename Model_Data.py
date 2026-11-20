import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_data(file_path):
    """
    Load data from in .tab format and return it as a NumPy array.

    Parameters:
    file_path (str): The path to the .tab file.

    Returns:
    vert (np.ndarray): The loaded vertex data as a NumPy array.
    face (np.ndarray): The loaded face data as a NumPy array.
    """
    try:
        with open(file_path, 'r') as file:
            nv, nf = map(int, file.readline().split()) # Finds number of faces and vertcies in km
            vert = np.loadtxt(file_path, skiprows=1, max_rows=nv)[:,1:4] # Loads vertex coordinates
            face = np.loadtxt(file_path, skiprows=nv+1, max_rows=nf, dtype=int)[:,1:4] - 1 # Loads face information (0-Based)
        assert vert.shape == (nv, 3) and face.shape == (nf, 3), "Incomplete data"
        return vert, face
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def get_model(vert, fac, step=1, shade=True, light=[0.5, -0.5, 0.7]):
    """
    Creates 3D model from vertex and face data.
    
    Parameters:
    vert (np.ndarray): The vertex coordinates.
    face (np.ndarray): The face indices.
    step (int): Step size for plotting vertices.
    shade (bool): Whether to apply shading based on light direction.
    light (list): The light direction vector.
    
    Returns:
    collection (Poly3DCollection): A Poly3DCollection object representing the 3D model.
    """
    tris = vert[face[::step]] # Triangles

    # Colors
    if shade:
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) # Normals
        n= n/(np.linalg.norm(n, axis=1, keepdims=True) + 1e-12) # Normalizing Normals
        light = np.array(light) # Light direction
        light = light / np.linalg.norm(light) # Normalizing Light
        shade = 0.15 + 0.7 * np.clip(n @ light, 0, 1) # Shading Factor
        colours = plt.cm.bone(shade) # Colour Mapping

        collection = Poly3DCollection(tris, facecolors=colours, edgecolors=colours, linewidths=0.1)
        return collection
    else:
        collection = Poly3DCollection(tris, facecolors='lightgrey', edgecolors='k', linewidths=0.1)
        return collection


def plot_model(vert, face, step=1, elev=0, axim=0, shade=True, light=[0.5, -0.5, 0.7], save=None):
    """
    Plot a 3D model using vertex and face data.

    Parameters:
    vert (np.ndarray): The vertex coordinates.
    face (np.ndarray): The face indices.
    step (int): Step size for plotting vertices.
    elev (float): Elevation angle in the z plane.
    axim (float): Azimuth angle in the x,y plane.
    shade (bool): Whether to apply shading based on light direction.
    light (list): The light direction vector.
    save (str): Path to save the plot. If None, the plot is shown instead.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection for the faces
    ax.add_collection3d(get_model(vert, face, step=step, shade=shade, light=light))
    
    # Set limits and labels
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    
    # Set view angle
    ax.view_init(elev=elev, azim=axim)
    
    # Set limits based on vertex data
    max_range = np.array([vert[:, 0].max()-vert[:, 0].min(), 
                          vert[:, 1].max()-vert[:, 1].min(), 
                          vert[:, 2].max()-vert[:, 2].min()]).max() / 2.0
    mid_x = (vert[:, 0].max()+vert[:, 0].min())/2
    mid_y = (vert[:, 1].max()+vert[:, 1].min())/2
    mid_z = (vert[:, 2].max()+vert[:, 2].min())/2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save:
        plt.savefig(save)
        print(f"Plot saved to {save}")
    plt.show()

if __name__ == "__main__":
    Path = "Asteroid_Model/data/vertex/ver64q.tab"
    vert, face = load_data(Path)
    plot_model(vert, face, step=1, elev=20, axim=45, shade=False)
