#!/usr/bin/env python3
import numpy as np
from scipy.special import erf
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm
import csv
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import itertools  # Add this import at the top with other imports
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from scipy.interpolate import griddata  # Add this import at the top with other imports
import streamlit as st

# ============================
# Configuration and Parameters
# ============================

# Input Parameters (Human-Readable Units)
CANVAS_SIZE_MM = 10.0                     # Simulation field size: 10 mm x 10 mm
PIXELS_PER_MM = 18.5                     # Resolution: 100 pixels per millimeter
CANVAS_SIZE_PIXELS = int(CANVAS_SIZE_MM * PIXELS_PER_MM)  # Canvas resolution: 1000 x 1000 pixels

DIAMETER_RANGE_MM = np.linspace(0.0, 2, 40)  # Defect diameter: 0.0 mm to 2.0 mm in 20 steps
OPACITY_RANGE = np.linspace(0.0, 1, 100)      # Defect opacity: 0% to 100% in 20 steps

# Golden Reference Parameters
GOLDEN_DIAMETER_MM = .88                    # Example diameter for golden reference
GOLDEN_OPACITY = 0.68                         # Example opacity for golden reference

WAVELENGTH_NM = 1000.0                        # Wavelength: 1000 nm
PROPAGATION_DISTANCE_MM = 0.0                 # Propagation distance: 0 mm

SPATIAL_FILTER_CUTOFF_FREQ_MM = 2           # Spatial filter cutoff frequency: mm⁻¹

# Similarity Threshold
SIMILARITY_THRESHOLD = 0.975  # Updated default similarity threshold

# Compare to Golden Image Flag
COMPARE_TO_GOLDEN = True                      # Flag to compare to golden image or just output all images

# Derived Parameters
PIXEL_SIZE_MM = 1 / PIXELS_PER_MM             # Pixel size in mm
PIXEL_SIZE_M = PIXEL_SIZE_MM * 1e-3           # Pixel size in meters
CANVAS_SIZE_M = CANVAS_SIZE_MM * 1e-3         # Canvas size in meters
WAVELENGTH_M = WAVELENGTH_NM * 1e-9           # Wavelength in meters
PROPAGATION_DISTANCE_M = PROPAGATION_DISTANCE_MM * 1e-3  # Propagation distance in meters
SPATIAL_FILTER_CUTOFF_FREQ_M = SPATIAL_FILTER_CUTOFF_FREQ_MM * 1e3  # Convert mm⁻¹ to m⁻¹

DIAMETER_RANGE_PIXELS = DIAMETER_RANGE_MM / PIXEL_SIZE_MM  # Convert diameters from mm to pixels

RESULTS_DIR = "results_simple"
GOLDEN_FILENAME = "golden.tiff"
MATCHING_CSV = "matching_parameters.csv"
PLOT_FILENAME = "opacity_vs_diameter.jpg"

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

def create_mask_with_defect(canvas_size_pixels, diameter_m, opacity, pixel_size_m):
    """
    Creates a circular mask with a transmissive region outside a circular defect.

    Parameters:
    - canvas_size_pixels (int): Size of the canvas in pixels (assumed square).
    - diameter_m (float): Diameter of the circular defect in meters.
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
    radius_m = diameter_m / 2
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

def compute_cross_correlation(image1, image2):
    """
    Computes the normalized cross-correlation between the central lineouts of two images.
    
    Parameters:
    - image1 (ndarray): First image array.
    - image2 (ndarray): Second image array.
    
    Returns:
    - max_correlation (float): Maximum normalized cross-correlation value.
    """
    # Extract central lineouts
    center_index = image1.shape[0] // 2
    lineout1 = image1[center_index, :]
    lineout2 = image2[center_index, :]

    # Perform cross-correlation
    correlation = np.correlate(lineout1 - np.mean(lineout1), lineout2 - np.mean(lineout2), mode='full')
    correlation /= (np.std(lineout1) * np.std(lineout2) * len(lineout1))
    
    max_correlation = np.max(correlation)
    return max_correlation

def generate_golden_reference():
    """
    Generates the golden reference image and returns it.

    Returns:
    - intensity (ndarray): 2D array of the golden intensity pattern.
    """
    # Convert diameter from mm to meters
    diameter_m = GOLDEN_DIAMETER_MM * 1e-3

    # Step 1: Create initial field with erf roll-off
    initial_field = create_initial_field(CANVAS_SIZE_PIXELS, CANVAS_SIZE_M)

    # Step 2: Apply defect mask
    mask = create_mask_with_defect(CANVAS_SIZE_PIXELS, diameter_m, GOLDEN_OPACITY, PIXEL_SIZE_M)
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

    return intensity  # Removed saving to file

def load_golden_reference(filepath):
    """
    Loads the golden reference image.

    Parameters:
    - filepath (str): Path to the golden TIFF image.

    Returns:
    - golden_image (ndarray): 2D array of the golden image.
    """
    with Image.open(filepath) as img:
        golden_image = np.array(img).astype(np.float32)
    return golden_image

def run_simulation(diameter_mm, opacity):
    """
    Runs a single simulation and returns the intensity image.

    Parameters:
    - diameter_mm (float): Diameter of the defect in millimeters.
    - opacity (float): Opacity of the defect (0 to 1).

    Returns:
    - intensity (ndarray): 2D array of the intensity pattern.
    """
    # Convert diameter from mm to meters
    diameter_m = diameter_mm * 1e-3

    # Step 1: Create initial field with erf roll-off
    initial_field = create_initial_field(CANVAS_SIZE_PIXELS, CANVAS_SIZE_M)

    # Step 2: Apply defect mask
    mask = create_mask_with_defect(CANVAS_SIZE_PIXELS, diameter_m, opacity, PIXEL_SIZE_M)
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

    return intensity

def run_simulation_and_compare(params, golden_image, threshold, output_folder):
    """
    Runs a simulation, compares the generated image with the golden reference,
    and returns matching parameters if similarity meets the threshold.

    Parameters:
    - params (tuple): (diameter_mm, opacity)
    - golden_image (ndarray): 2D array of the golden image.
    - threshold (float): Similarity threshold (e.g., 0.9 for 90%).
    - output_folder (str): Directory to save matching images.

    Returns:
    - result (dict): Dictionary with 'diameter_mm', 'opacity', and 'similarity'.
    """
    diameter_mm, opacity = params
    generated_intensity = run_simulation(diameter_mm, opacity)

    if COMPARE_TO_GOLDEN:
        # Compute cross-correlation similarity
        similarity = compute_cross_correlation(generated_intensity, golden_image)

        # Save the matching image if similarity meets the threshold
        # if similarity >= threshold:
        #     filename = f"matched_diameter_{diameter_mm:.2f}mm_opacity_{int(opacity*100)}%.tiff"
        #     filepath = os.path.join(output_folder, filename)
        #     save_tiff(generated_intensity, filepath)
    else:
        similarity = None

    return {'diameter_mm': diameter_mm, 'opacity': opacity, 'similarity': similarity}

def plot_matching_parameters(df, output_folder):
    """
    Plots a 2D contour plot of diameter vs. opacity with similarity as the z-axis and saves the plot as a JPEG.
    Overlays a red outline of the contour above the similarity threshold and adds a legend.
    Adds a red 'x' at the location of highest similarity.

    Parameters:
    - df (DataFrame): DataFrame with matching parameters.
    - output_folder (str): Directory to save the plot.
    """
    import pandas as pd

    # Create a grid for contour plot
    diameter_mm = df['diameter_mm'].values
    opacity = df['opacity'].values
    similarity = df['similarity'].values

    # Create grid data for contour plot
    diameter_grid, opacity_grid = np.meshgrid(np.unique(diameter_mm), np.unique(opacity))
    similarity_grid = np.zeros_like(diameter_grid)

    for i, (d, o, s) in enumerate(zip(diameter_mm, opacity, similarity)):
        x_idx = np.where(np.unique(diameter_mm) == d)[0][0]
        y_idx = np.where(np.unique(opacity) == o)[0][0]
        similarity_grid[y_idx, x_idx] = s

    # Clip similarity values to be within [0, 1]
    similarity_grid = np.clip(similarity_grid, 0, 1)

    plt.figure(figsize=(8, 6))
    contour_levels = np.arange(0, 1.01, 0.01)  # Contour levels at 1% increments
    contour = plt.contourf(diameter_grid, opacity_grid, similarity_grid, levels=contour_levels, cmap='viridis')
    plt.colorbar(contour, label='Similarity (Cross-Correlation)')
    
    # Overlay red outline for contours above the similarity threshold
    red_contour = plt.contour(diameter_grid, opacity_grid, similarity_grid, levels=[SIMILARITY_THRESHOLD], colors='red', linewidths=2)

    # Find the location of the highest similarity
    max_similarity_idx = np.unravel_index(np.argmax(similarity_grid, axis=None), similarity_grid.shape)
    max_diameter = diameter_grid[max_similarity_idx]
    max_opacity = opacity_grid[max_similarity_idx]
    max_similarity = similarity_grid[max_similarity_idx]

    # Add a red 'x' at the location of highest similarity
    plt.plot(max_diameter, max_opacity, 'rx', markersize=10, label=f'Highest Similarity: {max_similarity:.2f}\n(Diameter: {max_diameter:.2f} mm, Opacity: {max_opacity:.2f})')

    # Add legend for the red contour and red 'x'
    red_patch = plt.Line2D([0], [0], color='red', lw=2, label=f'Similarity Threshold: {SIMILARITY_THRESHOLD}')
    plt.legend(handles=[red_patch, plt.Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=10)], loc='lower right')

    plt.xlabel('Diameter (mm)')
    plt.ylabel('Opacity')
    plt.title('Opacity vs. Diameter for Matching Parameters')
    plt.grid(True)
    
    # Set plot limits to match sweep parameters
    plt.xlim(DIAMETER_RANGE_MM[0], DIAMETER_RANGE_MM[-1])
    plt.ylim(OPACITY_RANGE[0], OPACITY_RANGE[-1])
    
    plot_path = os.path.join(output_folder, PLOT_FILENAME)
    plt.savefig(plot_path, format='jpeg')
    plt.close()

def process_simulation(params, golden_image, threshold, output_folder):
    """
    Runs a simulation, compares the generated image with the golden reference,
    and returns matching parameters if similarity meets the threshold.

    Parameters:
    - params (tuple): (diameter_mm, opacity)
    - golden_image (ndarray): 2D array of the golden image.
    - threshold (float): Similarity threshold (e.g., 0.9 for 90%).
    - output_folder (str): Directory to save matching images.

    Returns:
    - match (dict or None): Dictionary with 'diameter_mm' and 'opacity' if matched, else None.
    """
    return run_simulation_and_compare(params, golden_image, threshold, output_folder)

def main():
    """
    Main function to set up the Streamlit app, handle user inputs,
    generate simulation images, and display results.
    """
    global GOLDEN_DIAMETER_MM, GOLDEN_OPACITY, SIMILARITY_THRESHOLD, DIAMETER_RANGE_MM, OPACITY_RANGE

    st.title("Defect Simulation App")
    st.write("Adjust the parameters to simulate defects and view the results.")

    # Sidebar for controllers
    st.sidebar.header("Simulation Parameters")

    # Sliders for GOLDEN_DIAMETER_MM and GOLDEN_OPACITY
    golden_diameter_mm = st.sidebar.slider(
        'Defect Diameter (mm)',
        min_value=0.0,
        max_value=5.0,  # Increased max from 2.0 to 5.0 mm
        value=GOLDEN_DIAMETER_MM,
        step=0.001,  # Finer step size
        format="%.3f"  # Display three decimals
    )
    golden_opacity = st.sidebar.slider(
        'Defect Opacity',
        min_value=0.0,
        max_value=1.0,
        value=GOLDEN_OPACITY,
        step=0.001,  # Finer step size
        format="%.3f"  # Display three decimals
    )

    # Numeric input for SIMILARITY_THRESHOLD
    similarity_threshold = st.sidebar.number_input(
        'Similarity Threshold',
        min_value=0.0,
        max_value=1.0,
        value=SIMILARITY_THRESHOLD,
        step=0.001,
        format="%.3f"  # Display three decimals
    )

    # Collapsible section for Contour Parameters
    with st.sidebar.expander("Contour Parameters"):
        # Prefixed labels with "Contour"
        diameter_range_min = st.number_input(
            'Contour Diameter Range Min (mm)', 
            value=0.0, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )
        diameter_range_max = st.number_input(
            'Contour Diameter Range Max (mm)', 
            value=5.0, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )
        diameter_step_size = st.number_input(
            'Contour Diameter Step Size (mm)', 
            value=0.1, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )
        opacity_range_min = st.number_input(
            'Contour Opacity Range Min', 
            value=0.0, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )
        opacity_range_max = st.number_input(
            'Contour Opacity Range Max', 
            value=1.0, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )
        opacity_step_size = st.number_input(
            'Contour Opacity Step Size', 
            value=0.1, 
            step=0.001,  # Finer step size
            format="%.3f"  # Display three decimals
        )

    # Update ranges based on user input
    DIAMETER_RANGE_MM = np.arange(diameter_range_min, diameter_range_max + diameter_step_size, diameter_step_size)
    OPACITY_RANGE = np.arange(opacity_range_min, opacity_range_max + opacity_step_size, opacity_step_size)

    # "Set Golden Defect" button
    if st.sidebar.button('Set Golden Defect'):
        # Update parameters
        GOLDEN_DIAMETER_MM = golden_diameter_mm
        GOLDEN_OPACITY = golden_opacity
        SIMILARITY_THRESHOLD = similarity_threshold

        # Generate golden reference without saving to file
        golden_image = generate_golden_reference()

        # Prepare simulation combinations
        simulation_combinations = list(product(DIAMETER_RANGE_MM, OPACITY_RANGE))
        results = []
        for params in simulation_combinations:
            result = run_simulation_and_compare(params, golden_image, SIMILARITY_THRESHOLD, ".")
            results.append(result)

        # Convert results to DataFrame without saving to CSV
        import pandas as pd
        df = pd.DataFrame(results)

        # Generate contour plot using the DataFrame directly
        plot_matching_parameters(df, ".")

        # Display contour plot
        plot_path = os.path.join(".", PLOT_FILENAME)
        contour_image = Image.open(plot_path)
        st.image(contour_image, caption="Opacity vs. Diameter Contour Plot", use_container_width=True)

    else:
        st.write("Press 'Set Golden Defect' to update the simulation.")

    # Generate simulation image
    intensity = run_simulation(golden_diameter_mm, golden_opacity)
    fig_simulation, ax = plt.subplots(figsize=(2.45, 2.45))  # Further reduce size by ~30%
    im = ax.imshow(intensity, cmap='gray', extent=[-CANVAS_SIZE_MM/2, CANVAS_SIZE_MM/2, -CANVAS_SIZE_MM/2, CANVAS_SIZE_MM/2])
    plt.colorbar(im, ax=ax)
    ax.set_title('Golden Defect', fontsize=10)  # Smaller title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_xlim(-CANVAS_SIZE_MM/2, CANVAS_SIZE_MM/2)
    ax.set_ylim(-CANVAS_SIZE_MM/2, CANVAS_SIZE_MM/2)
    st.write("Golden Defect:")
    st.pyplot(fig_simulation, use_container_width=True)

    # Add min and max position entry boxes before the cross-sectional plot
    min_position_mm = st.number_input(
        'Min Position (mm)', 
        value=-1.0, 
        step=0.001,  # Finer step size
        format="%.3f"  # Display three decimals
    )
    max_position_mm = st.number_input(
        'Max Position (mm)', 
        value=1.0, 
        step=0.001,  # Finer step size
        format="%.3f"  # Display three decimals
    )

    # Cross-sectional lineout
    center_line = intensity[intensity.shape[0] // 2, :]
    fig_lineout, ax_line = plt.subplots(figsize=(2.45, 1.0))  # Adjust aspect ratio
    
    # Calculate start and end indices based on min and max positions
    start_idx = max(int((min_position_mm + CANVAS_SIZE_MM / 2) * PIXELS_PER_MM), 0)
    end_idx = min(int((max_position_mm + CANVAS_SIZE_MM / 2) * PIXELS_PER_MM), CANVAS_SIZE_PIXELS)
    
    # Slice the data for zooming
    lineout_slice = center_line[start_idx:end_idx]
    x_positions_mm = np.linspace(
        min_position_mm, 
        max_position_mm, 
        end_idx - start_idx
    )
    
    # Plot the sliced data
    ax_line.plot(x_positions_mm, lineout_slice)
    ax_line.set_ylim(0, 1.2)  # Fix y-axis between 0 and 1.2
    ax_line.set_title('Cross-sectional Lineout', fontsize=10)  # Smaller title
    ax_line.set_xlabel('Position (mm)')
    ax_line.set_ylabel('Intensity')  # Corrected label
    ax_line.set_xticks(np.arange(min_position_mm, max_position_mm + 0.5, 0.5))  # Tick every 0.5 mm
    ax_line.set_yticks(np.arange(0, 1.2 + 0.2, 0.2))  # Tick every 0.2 units
    ax_line.set_xlim(min_position_mm, max_position_mm)  # Set based on user input
    st.write("Cross-sectional Lineout:")
    st.pyplot(fig_lineout, use_container_width=True)

if __name__ == "__main__":
    main()
