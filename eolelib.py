import numpy as np
from numpy.fft import fftshift, ifftn
import unittest
import pandas as pd
from netCDF4 import Dataset


def UShear(z, z_ref=100, u_inf=10, model="powerlaw", z0=0.03):
    """
    Compute the mean velocity profile as a function of height z using either the power law or log law.

    Parameters:
    - z: float
        Height above the ground (m).
    - z_ref: float, optional (default=100)
        Reference height (m) where the reference wind speed is known.
    - u_inf: float, optional (default=10)
        Reference wind speed at height z_ref (m/s).
    - model: str, optional (default="powerlaw")
        The model to use for the velocity profile, either "powerlaw" or "loglaw".
    - z0: float, optional (default=0.03)
        Surface roughness length (m), used only for the log law model.

    Returns:
    - float: The wind velocity at height z.
    """
    if model == "powerlaw":
        # Power-law velocity profile: U(z) = (z / z_ref) ^ alpha * u_inf
        # where alpha = 1/7 is a common exponent for neutral conditions.
        return (z / z_ref) ** (0.14285714285714285) * u_inf
    elif model == "loglaw":
        # Logarithmic velocity profile: U(z) = (U_star / kappa) * log(z / z0)
        # where kappa is the von Karman constant and U_star is the friction velocity.
        kappa = 0.4  # von Karman constant
        U_star = u_inf / (np.log(z_ref / z0) / kappa)  # Friction velocity
        return (U_star / kappa) * np.log(z / z0)

def VonKarmanTurbulenceField(Lx, Ly, Lz, Nx, Ny, Nz, L=35.4):
    """
    Generates a synthetic 3D turbulent velocity field using the Von Karman power spectrum.

    Parameters:
    - Lx, Ly, Lz: float
        Dimensions of the domain in the x, y, and z directions (m).
    - Nx, Ny, Nz: int
        Number of grid points in the x, y, and z directions, respectively.
    - L: float
        The characteristic length scale of turbulence (m), Default L=35.4.

    Returns:
    - u, v, w: ndarray
        Arrays representing the velocity components in the x, y, and z directions, respectively.
    """

    # Step 1: Mesh for the wave number vector k (kx, ky, kz)
    # The wave numbers are computed using the Nyquist frequency for each direction.
    ky, kz, kx = np.meshgrid((np.pi / Ly) * np.arange(-Ny, Ny, 2), 
                             (np.pi / Lz) * np.arange(-Nz, Nz, 2), 
                             (np.pi / Lx) * np.arange(-Nx, Nx, 2), 
                             indexing='ij')

    # Step 2: Power spectrum S(k), based on the Von Karman model
    # S = (L^(-2) + |k|^2)^(-17/12), where |k|^2 = kx^2 + ky^2 + kz^2
    # The exponent -17/12 comes from the Von Karman turbulence theory.
    S = (L**(-2) + kx**2 + ky**2 + kz**2)**(-17/12)

    # Step 3: Generate random phase spectra for each velocity component.
    # The phase is uniformly distributed in [0, 2*pi].
    PhiX = np.exp(2j * np.pi * np.random.rand(Ny, Nz, Nx))
    PhiY = np.exp(2j * np.pi * np.random.rand(Ny, Nz, Nx))
    PhiZ = np.exp(2j * np.pi * np.random.rand(Ny, Nz, Nx))

    # Step 4: Inverse Fourier transform to obtain the velocity components.
    # The cross product of the wave vector k with the random phases gives the velocity components.
    # This ensures incompressibility of the flow (i.e., div(velocity) = 0).
    u = ifftn(fftshift(S * (ky * PhiZ - kz * PhiY)), s=(Ny, Nz, Nx)).real
    v = ifftn(fftshift(S * (kz * PhiX - kx * PhiZ)), s=(Ny, Nz, Nx)).real
    w = ifftn(fftshift(S * (kx * PhiY - ky * PhiX)), s=(Ny, Nz, Nx)).real

    # Step 5: Return the velocity components (u, v, w)
    return u, v, w

def get_velocity(x, y, z, velocity_table, Lx, Ly, Lz):

    """
    Returns the interpolated velocity at coordinates (x, y, z) using trilinear interpolation.
    
    Parameters:
    - x, y, z: Coordinates where the velocity is needed.
    - velocity_table: 3D numpy array of shape (Ny, Nz, Nx) representing the velocity table.
    - Lx, Ly, Lz: Corresponding dimensions of the grid.

    Returns:
    - interpolated velocity at (x, y, z).
    """
    # Assert that x, y, z are within the grid's bounds
    assert 0 <= x <= Lx, "x-coordinate is out of bounds."
    assert 0 <= y <= Ly, "y-coordinate is out of bounds."
    assert 0 <= z <= Lz, "z-coordinate is out of bounds."
    
    # Get the grid shape from the velocity table
    Ny, Nz, Nx = velocity_table.shape

    # Compute the relative coordinates inside the grid
    x_index = x / Lx * (Nx - 1)
    y_index = y / Ly * (Ny - 1)
    z_index = z / Lz * (Nz - 1)
    
    # Get the integer indices for interpolation
    x0 = int(np.floor(x_index))
    x1 = min(x0 + 1, Nx - 1)
    y0 = int(np.floor(y_index))
    y1 = min(y0 + 1, Ny - 1)
    z0 = int(np.floor(z_index))
    z1 = min(z0 + 1, Nz - 1)
    
    # Get the fractional part for interpolation
    xd = x_index - x0
    yd = y_index - y0
    zd = z_index - z0
    
    # Fetch the values from the velocity table at the eight corners of the cube
    v000 = velocity_table[y0, z0, x0]
    v001 = velocity_table[y0, z0, x1]
    v010 = velocity_table[y0, z1, x0]
    v011 = velocity_table[y0, z1, x1]
    v100 = velocity_table[y1, z0, x0]
    v101 = velocity_table[y1, z0, x1]
    v110 = velocity_table[y1, z1, x0]
    v111 = velocity_table[y1, z1, x1]
    
    # Trilinear interpolation
    v00 = v000 * (1 - xd) + v001 * xd
    v01 = v010 * (1 - xd) + v011 * xd
    v10 = v100 * (1 - xd) + v101 * xd
    v11 = v110 * (1 - xd) + v111 * xd
    
    v0 = v00 * (1 - zd) + v01 * zd
    v1 = v10 * (1 - zd) + v11 * zd
    
    interpolated_velocity = v0 * (1 - yd) + v1 * yd
    
    return interpolated_velocity

def integration_cubature57(f, R, weights_file, x_c=0, y_c=0):
    """
    Numerically approximates the integral of a function over a disk of radius R using cubature,
    with an optional offset for the center of the disk.

    This function integrates the given function `f(x, y)` in the Cartesian plane, 
    using polar quadrature nodes and weights provided in the `weights_file` file.
    The quadrature nodes and weights are transformed from polar coordinates 
    (r, phi) to Cartesian coordinates (x, y) for numerical approximation.
    Optionally, the center of the disk can be shifted by providing coordinates (x_c, y_c).

    Parameters:
    -----------
    f : function
        A function of two variables (x, y) that represents the integrand.
        
    R : float
        The radius of the disk over which the integral is computed.
        
    weights_file : str
        The file path to the .dat file containing quadrature nodes and weights.
        The .dat file should contain five columns:
        - 'w' for the weight,
        - 'y' for y-coordinate (not used),
        - 'z' for z-coordinate (not used),
        - 'r' for the radial coordinate,
        - 'phi' for the angular coordinate.
    
    x_c : float, optional
        The x-coordinate of the center of the circle (default is 0).
    
    y_c : float, optional
        The y-coordinate of the center of the circle (default is 0).

    Returns:
    --------
    float
        The approximate value of the integral over the disk.
    """
    
    # Read the table with quadrature nodes (w, y, z, r, phi)
    nodes = pd.read_csv(weights_file, delimiter=" ", dtype=float)
    
    result = 0
    
    # Loop through all quadrature nodes
    for i in range(len(nodes)):
        # Polar coordinates from the table
        r = float(R * nodes["r"][i])  # scale the radius by R
        phi = float(nodes["phi"][i])  # angle in radians
        
        # Corresponding quadrature weight (scaled by R^2)
        w = R**2 * float(nodes["w"][i])
        
        # Transform to Cartesian coordinates, including the shift for the center (x_c, y_c)
        x = x_c + r * np.cos(phi)
        y = y_c + r * np.sin(phi)
        
        # Compute the weighted contribution to the integral
        result += w * f(x, y)
    
    return result

    from netCDF4 import Dataset


def export_to_netcdf(turbine_parameters, filename="turbine_parameters.nc"):
    """
    Export a list of turbine parameters to a NetCDF file, handling complex numbers by
    saving real and imaginary parts separately.

    Parameters:
    - turbine_parameters (list of dict): List where each dictionary contains parameters for a turbine.
    - filename (str): Name of the NetCDF file to save.

    Complex values are stored as separate real and imaginary components.
    """
    with Dataset(filename, "w", format="NETCDF4") as nc_file:
        num_turbines = len(turbine_parameters)
        nc_file.createDimension("turbine", num_turbines)

        # Determine scalar, array, and complex keys
        scalar_keys = []
        array_keys = []
        complex_keys = []

        for key, value in turbine_parameters[0].items():
            if isinstance(value, (list, np.ndarray)):
                array_keys.append(key)
                if np.iscomplexobj(value):
                    complex_keys.append(key)
                    print(f"Detected complex array key for export: {key}")
            elif np.iscomplex(value):
                complex_keys.append(key)
                print(f"Detected complex scalar key for export: {key}")
            else:
                scalar_keys.append(key)

        # Define dimensions for array parameters
        for key in array_keys:
            length = len(turbine_parameters[0][key])
            nc_file.createDimension(f"{key}_dim", length)

        # Save scalar variables
        for key in scalar_keys:
            var = nc_file.createVariable(key, "f4", ("turbine",))
            var[:] = [record[key] for record in turbine_parameters]

        # Save array variables, splitting complex numbers into real and imaginary parts
        for key in array_keys:
            if key in complex_keys:
                # Separate into real and imaginary components
                var_real = nc_file.createVariable(f"{key}_real", "f4", ("turbine", f"{key}_dim"))
                var_imag = nc_file.createVariable(f"{key}_imag", "f4", ("turbine", f"{key}_dim"))
                var_real[:, :] = np.array([np.real(record[key]) for record in turbine_parameters])
                var_imag[:, :] = np.array([np.imag(record[key]) for record in turbine_parameters])
                print(f"Exported complex array key: {key}_real and {key}_imag")
            else:
                var = nc_file.createVariable(key, "f4", ("turbine", f"{key}_dim"))
                var[:, :] = np.array([record[key] for record in turbine_parameters])

        # Save scalar complex values
        for key in complex_keys:
            if key not in array_keys:
                var_real = nc_file.createVariable(f"{key}_real", "f4", ("turbine",))
                var_imag = nc_file.createVariable(f"{key}_imag", "f4", ("turbine",))
                var_real[:] = [np.real(record[key]) for record in turbine_parameters]
                var_imag[:] = [np.imag(record[key]) for record in turbine_parameters]
                print(f"Exported complex scalar key: {key}_real and {key}_imag")

        nc_file.description = "Turbine parameters dataset with complex values split into real and imaginary parts."


def load_from_netcdf(filename="turbine_parameters.nc"):
    """
    Load turbine parameters from a NetCDF file into a list of dictionaries, reconstructing complex numbers.

    Parameters:
    - filename (str): Name of the NetCDF file to load.

    Returns:
    - list of dict: List where each dictionary contains parameters for a turbine.

    Complex values are reconstructed from separate real and imaginary components.
    """
    turbine_parameters = []

    with Dataset(filename, "r", format="NETCDF4") as nc_file:
        scalar_keys = set()
        array_keys = []
        complex_keys = set()  # Using set to avoid duplicate entries
        expected_keys = set()  # Track all expected keys for verification

        # Identify scalar, array, and complex variables
        for key in nc_file.variables:
            if "_real" in key or "_imag" in key:
                base_key = key.rsplit("_", 1)[0]
                complex_keys.add(base_key)  # Avoid duplicates with set
                scalar_keys.add(base_key)
                expected_keys.add(f"{base_key}_real")
                expected_keys.add(f"{base_key}_imag")
                print(f"Detected complex key during load: {base_key}")
            elif len(nc_file.variables[key].dimensions) == 1:
                scalar_keys.add(key)
                expected_keys.add(key)
            else:
                array_keys.append(key)
                expected_keys.add(key)

        # Check if all expected keys are in the file
        missing_keys = expected_keys - set(nc_file.variables.keys())
        if missing_keys:
            print(f"Warning: Missing keys in NetCDF file: {missing_keys}")
            # Raise an exception or handle missing keys as needed
            raise KeyError(f"NetCDF file is missing expected keys: {missing_keys}")

        # Construct each turbine's dictionary
        for i in range(len(nc_file.dimensions["turbine"])):
            turbine_data = {}
            print(scalar_keys)
            print(complex_keys)
            print(array_keys)
            for key in scalar_keys:
                if key in complex_keys:
                    # Reconstruct scalar complex number
                    turbine_data[key] = complex(
                        nc_file.variables[f"{key}_real"][i],
                        nc_file.variables[f"{key}_imag"][i]
                    )
                    print(f"Reconstructed complex scalar key: {key}")
                else:
                    turbine_data[key] = nc_file.variables[key][i].item()

            for key in array_keys:
                if key in complex_keys:
                    # Reconstruct array complex number
                    real_part = np.array(nc_file.variables[f"{key}_real"][i, :])
                    imag_part = np.array(nc_file.variables[f"{key}_imag"][i, :])
                    turbine_data[key] = (real_part + 1j * imag_part).tolist()
                    print(f"Reconstructed complex array key: {key}")
                else:
                    turbine_data[key] = nc_file.variables[key][i, :].tolist()
            
            # Debugging: Confirm all loaded keys for each turbine data
            print(f"Loaded turbine data keys: {turbine_data.keys()}")
                    
            turbine_parameters.append(turbine_data)

    return turbine_parameters