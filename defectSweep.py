#!/usr/bin/env python3
import numpy as np
from scipy.special import erf
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from tqdm import tqdm

# ============================
# Configuration and Parameters
# ============================

# Input Parameters (Human-Readable Units)
CANVAS_SIZE_MM = 10.0                     # Simulation field size: 10 mm x 10 mm
PIXELS_PER_MM = 100                       # Resolution: 100 pixels per millimeter
CANVAS_SIZE_PIXELS = int(CANVAS_SIZE_MM * PIXELS_PER_MM)  # Canvas resolution: 1000 x 1000 pixels

RADIUS_RANGE_MM = np.linspace(0.05, 1.0, 10)  # Defect radius: 0.05 mm to 1.0 mm in 10 steps
OPACITY_RANGE = np.linspace(0, 1, 10)         # Defect opacity: 0% to 100% in 10 steps

WAVELENGTH_NM = 1000.0                      # Wavelength: 1000 nm
PROPAGATION_DISTANCE_MM = 0.0             # Propagation distance: 100 mm

SPATIAL_FILTER_CUTOFF_FREQ_MM = 1.0          # Spatial filter cutoff frequency: mm⁻¹

# Derived Parameters
PIXEL_SIZE_MM = 1 / PIXELS_PER_MM           # Pixel size in mm
PIXEL_SIZE_M = PIXEL_SIZE_MM * 1e-3         # Pixel size in meters
CANVAS_SIZE_M = CANVAS_SIZE_MM * 1e-3       # Canvas size in meters
WAVELENGTH_M = WAVELENGTH_NM * 1e-9         # Wavelength in meters
PROPAGATION_DISTANCE_M = PROPAGATION_DISTANCE_MM * 1e-3  # Propagation distance in meters
SPATIAL_FILTER_CUTOFF_FREQ_M = SPATIAL_FILTER_CUTOFF_FREQ_MM * 1e3  # Convert mm⁻¹ to m⁻¹

RADIUS_RANGE_PIXELS = RADIUS_RANGE_MM / PIXEL_SIZE_MM    # Convert radii from mm to pixels

RESULTS_DIR = "results_simple"

# ============================
# Function Definitions
# ============================

def create_initial_field(canvas_size_pixels, canvas_size_m):
    """
    Creates the initial planar wavefront with an erf roll-off at the edges.

    Parameters:
    - canvas_size_pixels (int): Size of the canvas in pixels (assumed square).
    - canvas_size_m (float): Physical size of the canvas in meters.

    Returns:
    - initial_field (ndarray): 2D array representing the initial field with roll-off.
    """
    # Spatial coordinates in meters
    x = np.linspace(-canvas_size_m / 2, canvas_size_m / 2, canvas_size_pixels)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)

    # Apply erf roll-off at the edges
    roll_off_width_m = 0.5e-3  # Roll-off width of 0.5 mm converted to meters
    roll_off = 0.5 * (1 + erf((canvas_size_m / 2 - r) / roll_off_width_m))

    # Initial field with amplitude 1 and roll-off
    initial_field = roll_off.astype(np.complex64)

    return initial_field

def create_mask_with_defect(canvas_size_pixels, radius_m, opacity, pixel_size_m):
    """
    Creates a circular mask with a transmissive region outside a circular defect.

    Parameters:
    - canvas_size_pixels (int): Size of the canvas in pixels (assumed square).
    - radius_m (float): Radius of the circular defect in meters.
    - opacity (float): Opacity of the defect, scaling between 0 (transparent) and 1 (opaque).
    - pixel_size_m (float): Physical size of a pixel in meters.

    Returns:
    - mask (ndarray): 2D array representing the mask with a circular defect.
    """
    # Spatial coordinates in meters
    x = np.linspace(-canvas_size_pixels / 2, canvas_size_pixels / 2, canvas_size_pixels) * pixel_size_m
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)

    # Create the mask
    mask = np.ones((canvas_size_pixels, canvas_size_pixels), dtype=np.float32)
    mask[r <= radius_m] = 1 - opacity

    return mask

def fresnel_propagate(U_in, wavelength_m, z_m, dx_m):
    """
    Performs Fresnel diffraction propagation on the input field using the angular spectrum method.

    Parameters:
    - U_in (ndarray): Input complex field.
    - wavelength_m (float): Wavelength of light in meters.
    - z_m (float): Propagation distance in meters.
    - dx_m (float): Grid spacing in meters.

    Returns:
    - U_out (ndarray): Complex field after propagation.
    """
    N = U_in.shape[0]
    k = 2 * np.pi / wavelength_m  # Wave number

    # Frequency coordinates
    fx = fftfreq(N, d=dx_m)
    fx = fftshift(fx)
    FX, FY = np.meshgrid(fx, fx)

    # Angular spectrum
    H = np.exp(-1j * np.pi * wavelength_m * z_m * (FX**2 + FY**2))
    U_fft = fftshift(fft2(U_in))
    U_fft_prop = U_fft * H
    U_out = ifft2(ifftshift(U_fft_prop))

    return U_out

def apply_spatial_filter(U_in, cutoff_freq_m):
    """
    Applies a low-pass spatial filter to the complex field.

    Parameters:
    - U_in (ndarray): Complex field to be filtered.
    - cutoff_freq_m (float): Cutoff frequency in cycles per meter.

    Returns:
    - U_filtered (ndarray): Filtered complex field.
    """
    N = U_in.shape[0]
    dx_m = PIXEL_SIZE_M
    fx = fftfreq(N, d=dx_m)
    fx = fftshift(fx)
    FX, FY = np.meshgrid(fx, fx)
    FR = np.sqrt(FX**2 + FY**2)

    # Create low-pass filter mask
    H = np.zeros_like(FR)
    H[FR <= cutoff_freq_m] = 1.0

    # Apply filter in frequency domain
    U_fft = fftshift(fft2(U_in))
    U_fft_filtered = U_fft * H
    U_filtered = ifft2(ifftshift(U_fft_filtered))

    return U_filtered

def save_tiff(image_array, filename):
    """
    Saves a NumPy array as a 32-bit float TIFF image.

    Parameters:
    - image_array (ndarray): 2D array representing the image.
    - filename (str): Path to save the TIFF image.
    """
    # Ensure non-negative values
    image_array = np.maximum(image_array, 0)

    # Normalize to maximum intensity of 1 for better visualization
    image_array /= np.max(image_array)

    # Convert to 32-bit float and save
    image_32bit = image_array.astype(np.float32)
    img = Image.fromarray(image_32bit)
    img.save(filename, format='TIFF')

def run_simulation(radius_mm, opacity, output_folder):
    """
    Runs a single simulation: creates initial field, applies mask, propagates, filters, and saves the result.

    Parameters:
    - radius_mm (float): Radius of the defect in millimeters.
    - opacity (float): Opacity of the defect (0 to 1).
    - output_folder (str): Path to the folder where the image will be saved.
    """
    # Convert radius from mm to meters
    radius_m = radius_mm * 1e-3

    # Step 1: Create initial field with erf roll-off
    initial_field = create_initial_field(CANVAS_SIZE_PIXELS, CANVAS_SIZE_M)

    # Step 2: Apply defect mask
    mask = create_mask_with_defect(CANVAS_SIZE_PIXELS, radius_m, opacity, PIXEL_SIZE_M)
    field_with_defect = initial_field * mask

    # Step 3: Fresnel propagation
    propagated_field = fresnel_propagate(
        U_in=field_with_defect,
        wavelength_m=WAVELENGTH_M,
        z_m=PROPAGATION_DISTANCE_M,
        dx_m=PIXEL_SIZE_M
    )

    # Step 4: Apply spatial filter
    filtered_field = apply_spatial_filter(
        U_in=propagated_field,
        cutoff_freq_m=SPATIAL_FILTER_CUTOFF_FREQ_M
    )

    # Step 5: Compute intensity
    intensity = np.abs(filtered_field)**2

    # Step 6: Save the final intensity pattern as a TIFF image
    # Define filename with radius in mm and opacity
    filename = f"simulation_radius_{radius_mm:.2f}mm_opacity_{int(opacity*100)}%.tiff"
    filepath = os.path.join(output_folder, filename)
    save_tiff(intensity, filepath)

def main():
    """
    Main function to set up the simulation parameters, create output directory, and run simulations in parallel.
    """
    # Create a single date-stamped subfolder for the simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder_name = (
        f"{timestamp}_sweep_radius_{RADIUS_RANGE_MM[0]:.2f}-{RADIUS_RANGE_MM[-1]:.2f}mm_"
        f"opacity_{int(OPACITY_RANGE[0]*100)}-{int(OPACITY_RANGE[-1]*100)}%_"
        f"filter_{SPATIAL_FILTER_CUTOFF_FREQ_MM}mm^-1"
    )
    output_folder = os.path.join(RESULTS_DIR, subfolder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Generate all combinations of radius and opacity
    simulation_combinations = list(product(RADIUS_RANGE_MM, OPACITY_RANGE))
    total_simulations = len(simulation_combinations)

    # Run simulations in parallel with a progress bar
    with ProcessPoolExecutor() as executor:
        with tqdm(total=total_simulations, desc="Simulations", unit="sim") as pbar:
            futures = [
                executor.submit(run_simulation, radius_mm, opacity, output_folder)
                for radius_mm, opacity in simulation_combinations
            ]

            # Update the progress bar as simulations complete
            for future in futures:
                future.result()  # Wait for simulation to complete
                pbar.update(1)

if __name__ == "__main__":
    main()
