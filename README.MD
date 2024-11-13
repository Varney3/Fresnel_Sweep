## Purpose
The DefectSweep app is designed to compare simulation data with physical near-field images to analyze and identify defects sizes and opacities. It allows users to adjust simulation parameters to achieve a close match with observed physical data and to determine what size defects could generate observed candidates.

## Basic Workflow
1. **Prepare the Physical Near-Field Image:**
   - Use separate software such as ImageJ or MATLAB to open the physical near-field image you wish to analyze.
   - Extract a cross-sectional lineout from the image for comparison purposes.

2. **Launch the DefectSweep App:**
   - Access the app using the provided link.
   - The app interface includes sliders and input fields for adjusting defect parameters.

3. **Adjust Simulation Parameters:**
   - **Opacity and Diameter:**
     - Use the sliders to modify the defect's opacity and diameter in the simulation.
     - Adjust these parameters until the simulation's lineout closely resembles the physical lineout extracted earlier.
   - **Similarity Threshold:**
     - Enter a similarity threshold value to refine the comparison, narrowing down the range of acceptable parameter matches.

4. **Set Golden Defect:**
   - Once the simulated lineout aligns satisfactorily with the physical lineout, click the "Set Golden Defect" button.
   - This action generates a contour map displaying potential defect sizes and opacities that could produce the observed near-field defect image.

## Additional Settings
- **Simulation Parameters:**
  - Access additional settings through the sidebar to adjust parameters such as canvas size, resolution, propagation distance, and spatial filter cutoff frequency to accurately reflect the physical system being modeled.

- **Contour Plot Controls:**
  - Modify the contour plotting range and resolution to achieve the desired level of detail in the visualization.

## Key Features
- **Interactive Visualization:**
  - View the simulation image alongside the cross-sectional lineout and scale bar.
  - Generate contour maps based on the adjusted defect parameters and similarity threshold.

- **Parameter Refinement:**
  - Fine-tune defect characteristics to closely match the physical observations.

- **Comprehensive Analysis:**
  - Identify potential defect configurations through contour maps, facilitating a better understanding of the defects present in the physical system.

This app streamlines the defect analysis process by providing intuitive controls and clear visual feedback, enabling more accurate identification and characterization of defects.
