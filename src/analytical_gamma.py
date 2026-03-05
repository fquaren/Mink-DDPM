import numpy as np
from skimage import measure, morphology
from scipy.ndimage import label


# --- Analytical Gamma Functions ---
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km):
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix
