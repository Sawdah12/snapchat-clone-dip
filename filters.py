import cv2
import numpy as np

# --- 1. NOISE REDUCTION & SMOOTHING ---
def apply_blur_bg(img): # Average/Mean Filter
    return cv2.blur(img, (10, 10))

def apply_gaussian(img): # Gaussian Filter
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_median(img): # Median Filter (Best for Salt & Pepper Noise)
    return cv2.medianBlur(img, 5)

# --- 2. EDGE DETECTION & SHARPENING ---
def apply_sobel(img): # Sobel Filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

def apply_canny(img): # Canny Edge Detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def apply_laplacian(img): # Laplacian (2nd Order Derivative)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)

def enhance_pixels(img): # Manual Sharpening Mask
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

# --- 3. COLOR MANAGEMENT (FROM YOUR IMAGES) ---
def filter_bright_low_con(img): # Linear Transformation
    return cv2.convertScaleAbs(img, alpha=0.5, beta=100)

def filter_dark_high_con(img): # Contrast Stretching
    return cv2.convertScaleAbs(img, alpha=2.0, beta=-50)

def boost_channel(img, channel_idx): # Channel Management
    img_copy = img.copy()
    # channel_idx: 0=Blue, 1=Green, 2=Red
    img_copy[:, :, channel_idx] = np.clip(img_copy[:, :, channel_idx] + 100, 0, 255)
    return img_copy

def filter_grayscale(img): # Grayscale Filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def filter_sepia(img): # Sepia Transformation Matrix
    kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    return cv2.transform(img, kernel)

def filter_invert(img): # Negative/X-Ray Transformation
    return cv2.bitwise_not(img)

# --- OVERLAY LOGIC ---
def overlay_transparent(background, overlay_path, x, y, w, h):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None: return background
    overlay = cv2.resize(overlay, (w, h))
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    overlay_crop = overlay[0:y2-y1, 0:x2-x1]
    alpha_mask = overlay_crop[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (background[y1:y2, x1:x2, c] * (1 - alpha_mask) + 
                                      overlay_crop[:, :, c] * alpha_mask)
    return background