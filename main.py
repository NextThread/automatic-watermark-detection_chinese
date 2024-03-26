import cv2
from src.estimate_watermark import *
from src.preprocess import *
from src.image_crawler import *
from src.watermark_reconstruct import *
import matplotlib.pyplot as plt

# Estimate watermark
gx, gy, gxlist, gylist = estimate_watermark('./images/fotolia_processed')

# Crop watermark
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# Random photo
img = cv2.imread('images/fotolia_processed/fotolia_137840668.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# Load cropped images
num_images = len(gxlist)
J, img_paths = get_cropped_images(
    'images/fotolia_processed', num_images, start, end, cropped_gx.shape)

# Subset of images
num_images = min(len(J), 25)
Jt = J[:num_images]

# Estimate alpha and blend factor
Wm = W_m - W_m.min()
alph_est = estimate_normalized_alpha(Jt, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(Jt, Wm, alph)

# Apply alpha correction
alpha = alph.copy()
for i in range(3):
    alpha[:, :, i] = C[i] * alpha[:, :, i]

Wm = Wm + alpha * est_Ik

# Normalize Wm
W = Wm.copy()
for i in range(3):
    W[:, :, i] /= C[i]

# Solve for all images
Wk, Ik, W, alpha1 = solve_images(Jt, W_m, alpha, W)
