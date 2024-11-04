import numpy as np
from numpy.fft import fftshift, ifftn
import unittest
import pandas as pd
import xarray as xr


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

def export_to_netcdf(data_list, filename):
    """
    Export a list of dictionaries to a NetCDF file, handling complex numbers by splitting into real and imaginary parts.

    This function takes a list of dictionaries, where each dictionary represents a single data entry with various data types.
    If complex numbers are present, they are split into separate real and imaginary components before saving to NetCDF.
    This allows for safe storage of complex data in a format that does not directly support complex numbers.

    Parameters:
        data_list (list): A list of dictionaries, each containing data of various types (including complex numbers).
        filename (str): The path where the NetCDF file will be saved.

    Example:
        data_list = [{'temp': 23, 'location': 2 + 3j}, {'temp': 19, 'location': 5 + 6j}]
        export_to_netcdf(data_list, 'data.nc')
    """
    # Create a dictionary to store the data for xarray
    data_vars = {}

    # Iterate through each dictionary in the list
    for i, data in enumerate(data_list):
        for key, value in data.items():
            if key in data_vars:
                data_vars[key].append(value)
            else:
                data_vars[key] = [value]

    # Convert each list of values to a DataArray, handling complex values by splitting into real and imaginary parts
    xr_data_vars = {}
    for key, value in data_vars.items():
        value = np.array(value)
        
        if np.iscomplexobj(value):
            xr_data_vars[f"{key}_real"] = xr.DataArray(value.real, dims=['index'] if value.ndim == 1 else ['index', 'complex_dim'])
            xr_data_vars[f"{key}_imag"] = xr.DataArray(value.imag, dims=['index'] if value.ndim == 1 else ['index', 'complex_dim'])
        else:
            dims = ['index'] if value.ndim == 1 else ['index', 'complex_dim']
            xr_data_vars[key] = xr.DataArray(value, dims=dims)
    
    # Create the dataset and save to NetCDF
    ds = xr.Dataset(xr_data_vars)
    ds.to_netcdf(filename, engine='netcdf4')
    #print(f"Data exported successfully to {filename}")

def load_from_netcdf(filename):
    """
    Load data from a NetCDF file into a list of dictionaries, reconstructing complex numbers if they were split.

    This function reads a NetCDF file and reconstructs complex numbers by combining real and imaginary parts where applicable.
    It iterates over each index in the file, collecting data variables into dictionaries and storing them in a list.

    Parameters:
        filename (str): The path to the NetCDF file to load.

    Returns:
        list: A list of dictionaries, each representing a single data entry. Complex numbers are reconstructed where applicable.

    Example:
        data_list = load_from_netcdf('data.nc')
    """
    ds = xr.open_dataset(filename, engine='netcdf4')
    data_list = []
    
    # Iterate over each index (dictionary entry)
    for i in range(len(ds.index)):
        data_dict = {}
        for key in ds.data_vars:
            if key.endswith("_real") or key.endswith("_imag"):
                # Reconstruct complex data
                base_key = key.rsplit("_", 1)[0]
                if f"{base_key}_real" in ds.data_vars and f"{base_key}_imag" in ds.data_vars:
                    real_part = ds[f"{base_key}_real"].sel(index=i).values
                    imag_part = ds[f"{base_key}_imag"].sel(index=i).values
                    data_dict[base_key] = real_part + 1j * imag_part
            else:
                # Normal data
                data_dict[key] = ds[key].sel(index=i).values
        data_list.append(data_dict)
    
    #print("Data loaded successfully")
    return data_list
