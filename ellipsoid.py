from Model_Data import plot_model, load_data
import matplotlib.pyplot as plt
import numpy as np

def euler_matrix(roll, pitch, yaw):
    """ Outputs Euler rotation matrix from given angle in radians.

    Parameters:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        np.ndarray: The Euler rotation matrix.
    """
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz =  np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def plot_ellipsoid(rx, ry, rz, cx, cy, cz, roll, pitch, yaw, npoints = 50, ax=None, show=True, save=None, **kwargs):
    """ Plots ellipsoid given location values

    Parameters:
        rx (float): Radius in x direction before rotation.
        ry (float): Radius in y direction before rotation.
        rz (float): Radius in z direction before rotation.
        cx (float): Center position in x direction.
        cy (float): Center position in y direction.
        cz (float): Center position in z direction.
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
        npoints (int): Number of points in each direction for meshgrid.
        ax (plt.axes, optional): Matplotlib axes object to plot on. If None, a new figure and axes are created.
        show (bool, optional): Whether to display the plot. If False, the plot is not shown.
        save (str, optional): Path to save the plot. If None, the plot is shown instead.
        **kwargs: Additional keyword arguments for the plot_surface function.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    defaults = {"alpha": 1.0, "edgecolor": "none", "color": "lightgrey"}
    defaults.update(kwargs)
    
    # Creating unit sphere
    u = np.linspace(0, 2 * np.pi, npoints)
    v = np.linspace(0, np.pi, npoints)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scaling to ellipsoid
    pts = np.stack([rx * x, ry * y, rz * z], axis=-1)

    # Rotation and translation
    rot = euler_matrix(roll,pitch,yaw)
    pts = pts @ rot.T + np.array([cx, cy, cz])
    ax.plot_surface(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], **defaults)
    if save:
        plt.savefig(save)
        print(f"Plot saved to {save}")
    if show:
        plt.show()

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Path = "Asteroid_Model/data/vertex/ver64q.tab"
    vert, face = load_data(Path)

    # First ellipsoid
    r1= [0.22,0.15,0.1]
    c1= [-0.06,0,0]
    a1 = np.deg2rad([0,-7,-13])
    plot_ellipsoid(r1[0],r1[1],r1[2],c1[0],c1[1],c1[2],a1[0],a1[1],a1[2],ax=ax, show=False, color='b', alpha=0.8)

    # Second ellipsoid
    r2= [0.105,0.09,0.085]
    c2= [0.21,0,-0.027]
    a2 = np.deg2rad([0,55,0])
    plot_ellipsoid(r2[0],r2[1],r2[2],c2[0],c2[1],c2[2],a2[0],a2[1],a2[2],ax=ax, show=False, color='r', alpha=0.8)

    plot_model(vert, face, step=1, elev=20, axim=45, shade=False,show=False, ax=ax, alpha=0.2)
    plt.show()
