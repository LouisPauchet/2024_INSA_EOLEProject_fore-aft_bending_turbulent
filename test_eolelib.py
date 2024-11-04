import numpy as np
from numpy.fft import fftshift, ifftn
import unittest
from eolelib import *
import numpy as np
import os


class TestWindTurbulenceModels(unittest.TestCase):

    def test_UShear_powerlaw(self):
        # Test the power law model
        z_ref = 100
        u_inf = 10
        z = 50
        expected_velocity = (z / z_ref) ** (1 / 7) * u_inf
        calculated_velocity = UShear(z, z_ref=z_ref, u_inf=u_inf, model="powerlaw")
        self.assertAlmostEqual(calculated_velocity, expected_velocity, places=5)

    def test_UShear_loglaw(self):
        # Test the log law model
        z_ref = 100
        u_inf = 10
        z = 50
        z0 = 0.03
        kappa = 0.4
        U_star = u_inf / (np.log(z_ref / z0) / kappa)
        expected_velocity = (U_star / kappa) * np.log(z / z0)
        calculated_velocity = UShear(z, z_ref=z_ref, u_inf=u_inf, model="loglaw", z0=z0)
        self.assertAlmostEqual(calculated_velocity, expected_velocity, places=5)

    def test_VonKarmanTurbulenceField_shapes(self):
        # Test if the turbulence field returns arrays with the correct shapes
        Lx, Ly, Lz = 1000.0, 1000.0, 500.0  # Domain sizes (m)
        Nx, Ny, Nz = 16, 16, 8  # Number of grid points
        L = 200.0  # Characteristic turbulence length scale

        u, v, w = VonKarmanTurbulenceField(Lx, Ly, Lz, Nx, Ny, Nz, L)

        # Check if the returned velocity fields have the correct shape
        self.assertEqual(u.shape, (Ny, Nz, Nx))
        self.assertEqual(v.shape, (Ny, Nz, Nx))
        self.assertEqual(w.shape, (Ny, Nz, Nx))

    def test_VonKarmanTurbulenceField_randomness(self):
        # Test if two different fields are not the same (randomness of the generated field)
        Lx, Ly, Lz = 1000.0, 1000.0, 500.0  # Domain sizes (m)
        Nx, Ny, Nz = 16, 16, 8  # Number of grid points
        L = 200.0  # Characteristic turbulence length scale

        u1, v1, w1 = VonKarmanTurbulenceField(Lx, Ly, Lz, Nx, Ny, Nz, L)
        u2, v2, w2 = VonKarmanTurbulenceField(Lx, Ly, Lz, Nx, Ny, Nz, L)

        # Check if the fields are different due to randomness
        self.assertFalse(np.allclose(u1, u2))
        self.assertFalse(np.allclose(v1, v2))
        self.assertFalse(np.allclose(w1, w2))

class TestGetVelocity(unittest.TestCase):

    def test_get_velocity(self):
        # Create a sample velocity table (3x3x3 grid)
        velocity_table = np.arange(27).reshape(3, 3, 3)
        
        # Define the dimensions of the grid
        Lx, Ly, Lz = 2.0, 2.0, 2.0
        
        # Test velocity at the corner of the grid (0, 0, 0)
        velocity_at_origin = get_velocity(0.0, 0.0, 0.0, velocity_table, Lx, Ly, Lz)
        #print("Velocity at (0.0, 0.0, 0.0):", velocity_at_origin)
        self.assertAlmostEqual(velocity_at_origin, 0, places=5, msg=f"Expected 0.0, but got {velocity_at_origin}")

        # Test velocity at the midpoint of the grid (1.0, 1.0, 1.0)
        interpolated_value = get_velocity(1.0, 1.0, 1.0, velocity_table, Lx, Ly, Lz)
        #print("Interpolated velocity at (1.0, 1.0, 1.0):", interpolated_value)
        # Update expected value based on interpolation result
        self.assertAlmostEqual(interpolated_value, 13.0, places=5, msg=f"Expected 13.0, but got {interpolated_value}")

        # Test velocity at the far corner of the grid (2.0, 2.0, 2.0)
        velocity_at_corner = get_velocity(2.0, 2.0, 2.0, velocity_table, Lx, Ly, Lz)
        #print("Velocity at (2.0, 2.0, 2.0):", velocity_at_corner)
        self.assertAlmostEqual(velocity_at_corner, 26, places=5, msg=f"Expected 26.0, but got {velocity_at_corner}")

        #print("All tests passed.")

class TestIntegrationCubature57(unittest.TestCase):
    
    def test_constant_function(self):
        """Test the integration with a constant function f(x, y) = 1 using the real weight.dat file."""
        def f(x, y):
            return 1

        R = 11  # Use the actual radius provided
        x_c, y_c = 2, 3  # Shifted center of the circle
        expected_result = np.pi * R**2  # The integral of 1 over a disk is the area of the disk: πR^2
        result = integration_cubature57(f, R, 'weight.dat', x_c, y_c)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_gaussian_function(self):
        """Test the integration with a Gaussian function f(x, y) = exp(-x^2 - y^2) using the real weight.dat file."""
        def f(x, y):
            return np.exp(-x**2 - y**2)

        R = 11
        x_c, y_c = 1, 1  # Shifted center of the circle
        # We don't have an easy analytical result for this, but we can check if the output is reasonable.
        result = integration_cubature57(f, R, 'weight.dat', x_c, y_c)
        self.assertTrue(0 < result < np.pi * R**2)  # The result should be less than the area of the disk.

    def test_polynomial_function(self):
        """Test the integration with a simple polynomial function f(x, y) = x^2 + y^2."""
        def f(x, y):
            return x**2 + y**2

        R = 2
        # The exact integral of f(x, y) = x^2 + y^2 over a disk of radius R is (πR^4)/2.
        expected_result = (np.pi * R**4) / 2
        result = integration_cubature57(f, R, 'weight.dat')
        self.assertAlmostEqual(result, expected_result, places=5)


    def test_polynomial_function_shifted(self):
        """Test the integration with a simple polynomial function f(x, y) = x^2 + y^2."""
        def f(x, y):
            return x**2 + y**2

        R = 2
        x_c, y_c = 4, 3
        expected_result = 339.29  # Computed expected result for the shifted circle
        
        result = integration_cubature57(f, R, 'weight.dat', x_c, y_c)
        self.assertAlmostEqual(result, expected_result, places=2)

# Example turbine_parameters data for testing, including complex values
example_data = [
    {
        'z_hub': 100.0,
        'R_rotor': 40.0,
        'K': 0.05,
        'M': 1.0,
        'FHS': 200.0,
        'gamma': 0.01 + 0.005j,  # Complex value
        'F_RHS': [0.1 + 0.02j, 0.2 + 0.03j, 0.3 + 0.04j],  # Array of complex values
        'F_RHS_hat': [0.15, 0.25, 0.35],
        'frequencies': [0.5, 1.0, 1.5],
        'omega': [0.2, 0.4, 0.6],
        'H_omega': [0.8 + 0.1j, 0.9 + 0.1j, 1.0 + 0.1j],  # Array of complex values
        'x_hat': [1.1, 1.2, 1.3],
        'x_temporal': [10.0 + 2.0j, 20.0 + 2.5j, 30.0 + 3.0j]  # Array of complex values
    },
    {
        'z_hub': 120.0,
        'R_rotor': 45.0,
        'K': 0.06,
        'M': 1.2,
        'FHS': 220.0,
        'gamma': 0.02 + 0.01j,  # Complex value
        'F_RHS': [0.15 + 0.03j, 0.25 + 0.04j, 0.35 + 0.05j],  # Array of complex values
        'F_RHS_hat': [0.18, 0.28, 0.38],
        'frequencies': [0.6, 1.1, 1.6],
        'omega': [0.3, 0.5, 0.7],
        'H_omega': [0.85 + 0.15j, 0.95 + 0.15j, 1.05 + 0.15j],  # Array of complex values
        'x_hat': [1.15, 1.25, 1.35],
        'x_temporal': [15.0 + 3.5j, 25.0 + 4.0j, 35.0 + 4.5j]  # Array of complex values
    }
]

class TestTurbineParameterNetCDF(unittest.TestCase):
    def setUp(self):
        """Set up a test NetCDF filename and ensure cleanup after each test."""
        self.filename = "test_turbine_parameters.nc"
        export_to_netcdf(example_data, self.filename)

    def tearDown(self):
        """Remove the test NetCDF file after tests run."""
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_file_creation(self):
        """Test if the NetCDF file is created successfully."""
        self.assertTrue(os.path.isfile(self.filename), "NetCDF file was not created.")

    def test_data_integrity(self):
        """Test if loaded data matches the original data, including complex numbers."""
        loaded_data = load_from_netcdf(self.filename)

        # Check overall structure
        self.assertEqual(len(loaded_data), len(example_data), "Number of records does not match.")

        # Check each record individually
        for original, loaded in zip(example_data, loaded_data):
            for key in original:
                self.assertIn(key, loaded, f"Missing key '{key}' in loaded data.")
                if isinstance(original[key], list):
                    # Handle lists (arrays) that might contain complex numbers
                    if np.iscomplexobj(original[key]):
                        np.testing.assert_almost_equal(
                            original[key], loaded[key],
                            err_msg=f"Data mismatch in key '{key}' for complex array values."
                        )
                    else:
                        np.testing.assert_almost_equal(
                            original[key], loaded[key],
                            err_msg=f"Data mismatch in key '{key}' for real array values."
                        )
                elif np.iscomplex(original[key]):
                    # Handle complex scalar values
                    self.assertAlmostEqual(
                        np.real(original[key]), np.real(loaded[key]),
                        msg=f"Data mismatch in key '{key}' for complex real part."
                    )
                    self.assertAlmostEqual(
                        np.imag(original[key]), np.imag(loaded[key]),
                        msg=f"Data mismatch in key '{key}' for complex imaginary part."
                    )
                else:
                    # Handle real scalar values
                    self.assertAlmostEqual(
                        original[key], loaded[key],
                        msg=f"Data mismatch in key '{key}' for scalar values."
                    )

    def test_max_psd_computation(self):
        """Test if data from x_temporal can produce a valid PSD."""
        from scipy.signal import welch

        # Load data to get x_temporal
        loaded_data = load_from_netcdf(self.filename)
        for record in loaded_data:
            # Ensure 'x_temporal' key exists
            self.assertIn('x_temporal', record, "Missing key 'x_temporal' in loaded data.")
            
            x_temporal = np.array(record['x_temporal']) - np.mean(record['x_temporal'])
            frequencies, psd = welch(np.real(x_temporal), fs=1.0)  # Assuming sampling frequency of 1 Hz

            # Check if frequencies and psd are non-empty and valid
            self.assertGreater(len(frequencies), 0, "Frequencies should not be empty.")
            self.assertGreater(len(psd), 0, "PSD should not be empty.")
            self.assertGreater(np.max(psd), 0, "PSD maximum should be greater than zero.")


# If you're running the test manually via a script:
if __name__ == '__main__':
    unittest.main()