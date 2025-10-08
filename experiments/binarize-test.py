import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def process_historical_document(image_np):
    """
    Process a historical document using Otsu binarization and classify
    whether it is likely a text page or a figure page.
    Returns the binarized image and classification string.
    """

    # Convert to grayscale if RGB
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Apply Gaussian blur to reduce noise before Otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu’s thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so text is white-on-black for analysis
    binary_inv = 255 - binary

    # Foreground (ink) ratio
    foreground_ratio = np.sum(binary_inv > 0) / binary_inv.size

    # Connected components to estimate object structure
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    large_components = sizes[sizes > 50]  # remove noise

    # Heuristic classification
    if foreground_ratio < 0.02:
        classification = "Mostly blank"
    elif foreground_ratio < 0.2 and len(large_components) > 30:
        classification = "Text page"
    else:
        classification = "Figure or illustration page"

    print(f"→ Foreground ratio: {foreground_ratio:.3f}")
    print(f"→ Large components: {len(large_components)}")
    print(f"→ Classification: {classification}")

    return binary, classification


# --- Path to the scanned image ---
file_path = (
    "/Users/ysc4337/warlock/Impulse/pilot_testing/p1074_35556030758452/JPG_OG/015.jpg"
)

# Load image
loaded_pil = Image.open(file_path)
loaded_np = np.array(loaded_pil)

# Process image
processed_image, classification = process_historical_document(loaded_np)

# Display result
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(loaded_np, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(processed_image, cmap="gray")
axes[1].set_title(f"Processed (Otsu Binarized)\nDetected: {classification}")
axes[1].axis("off")

plt.show()
