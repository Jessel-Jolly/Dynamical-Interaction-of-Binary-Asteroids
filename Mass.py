import numpy as np

def ellipsoid_volume(rx, ry, rz):
    """ Calculate the volume of an ellipsoid given its radii.

    Parameters:
        rx (float): Radius in x direction (km).
        ry (float): Radius in y direction (km).
        rz (float): Radius in z direction (km).

    Returns:
        float: The volume of the ellipsoid (m^3).
    """
    return (4/3) * np.pi * rx * ry * rz * 1e9

def mass_ellipsoid(rx, ry, rz, rho):
    """ Calculate the mass of an ellipsoid given its radii and density.

    Parameters:
        rx (float): Radius in x direction (km).
        ry (float): Radius in y direction (km).
        rz (float): Radius in z direction (km).
        rho (float): Density of the ellipsoid (kg/m^3).

    Returns:
        float: The mass of the ellipsoid (kg).
    """
    return rho * ellipsoid_volume(rx, ry, rz)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    from ellipsoid import plot_ellipsoid
    from Model_Data import load_data, plot_model
    e1, e2= np.loadtxt("ellipsoids.csv", delimiter=",", skiprows=1)
    rho1 = 1930
    rho2 = 2450
    m1 = mass_ellipsoid(*e1[:3], rho1)
    m2 = mass_ellipsoid(*e2[:3], rho2)
    print(f"Mass of Body: {m1:.2e} kg")
    print(f"Mass of Head: {m2:.2e} kg")
    print(f"Total Mass: {m1+m2:.2e} kg")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vert, face = load_data("Asteroid_Model/data/vertex/ver64q.tab")
    plot_ellipsoid(*e1, ax=ax, show=False, color="b", alpha=0.6)
    plot_ellipsoid(*e2, ax=ax, show=False, color="r", alpha=0.6)
    plot_model(vert, face, step=1, elev=30, axim=-70, shade=False, ax=ax, show=False, alpha=0.3)
    plt.savefig("ellipsoids.png")
    plt.show()
