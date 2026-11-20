import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_data(file_path):
    """
    Load data from in .tab format and return it as a NumPy array.

    Parameters:
    file_path (str): The path to the .tab file.

    Returns:
    np.ndarray: The loaded data as a NumPy array.
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

def plot_model(vert, face, step=1, elev=0, axim=0, save=None):
    """
    Plot a 3D model using vertex and face data.

    Parameters:
    vert (np.ndarray): The vertex coordinates.
    face (np.ndarray): The face indices.
    step (int): Step size for plotting vertices.
    elev (float): Elevation angle in the z plane.
    axim (float): Azimuth angle in the x,y plane.
    save (str): Path to save the plot. If None, the plot is shown instead.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a Poly3DCollection for the faces
    poly3d = [[vert[vert_idx] for vert_idx in face[i]] for i in range(face.shape[0])]
    collection = Poly3DCollection(poly3d, alpha=0.5, edgecolor='k')
    
    ax.add_collection3d(collection)
    
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

    mid_x = (vert[:, 0].max()+vert[:, 0].min()) * 0.5
    mid_y = (vert[:, 1].max()+vert[:, 1].min()) * 0.5
    mid_z = (vert[:, 2].max()+vert[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save:
        plt.savefig(save)
        print(f"Plot saved to {save}")
    plt.show()

if __name__ == "__main__":
    Path = "Asteroid_Model/data/vertex/ver128q.tab"
    vert, face = load_data(Path)
    plot_model(vert, face, step=1, elev=20, axim=45)
