import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.special import gamma, factorial
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def load_image(image_path):
    img = Image.open(image_path)
    img = np.asarray(img) / 255.0
    return img

def apply_gaussian_filter(img, sigma=2):
    if sigma > 0:
        img_filtered = gaussian_filter(img, sigma=sigma)
    else:
        img_filtered = img
    return img_filtered

def fractional_derivative(img, alpha, axis, kernel_size):
    half_size = kernel_size // 2
    kernel = np.zeros(kernel_size)
    for k in range(-half_size, half_size + 1):
        binomial_coeff = gamma(alpha + 1) / (factorial(abs(k)) * gamma(alpha - abs(k) + 1))
        sign = (-1)**k
        kernel[k + half_size] = sign * binomial_coeff

    if axis == 0: 
        kernel = kernel.reshape(1, -1)
    else:  
        kernel = kernel.reshape(-1, 1)
    
    return convolve(img, kernel, mode='nearest')

def compute_gradients(img, alpha, kernel_size):
    fd_x_r, fd_x_g, fd_x_b = [fractional_derivative(img[:, :, i], alpha, axis=0, kernel_size=kernel_size) for i in range(3)]
    fd_y_r, fd_y_g, fd_y_b = [fractional_derivative(img[:, :, i], alpha, axis=1, kernel_size=kernel_size) for i in range(3)]
    
    G_x = np.sqrt(fd_x_r**2 + fd_x_g**2 + fd_x_b**2)
    G_y = np.sqrt(fd_y_r**2 + fd_y_g**2 + fd_y_b**2)
    
    G = np.sqrt(G_x**2 + G_y**2)
    theta = np.arctan2(G_y, G_x)
    
    return G, theta

def non_maximal_suppression(G, theta):
    M, N = G.shape
    suppressed = np.zeros((M, N))
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = G[i + 1, j]
                r = G[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                suppressed[i, j] = G[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

def threshold_and_link_edges(G, threshold):
    return G > threshold

def edge_detection(image_path, alpha, sigma, threshold, kernel_size):
    img = load_image(image_path)
    img_filtered = apply_gaussian_filter(img, sigma)
    G, theta = compute_gradients(img_filtered, alpha, kernel_size)
    G_suppressed = non_maximal_suppression(G, theta)
    edges = threshold_and_link_edges(G_suppressed, threshold)

    edges_opencv = cv2.Canny((img_filtered * 255).astype(np.uint8), 50, 150) 
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Edges Detected')
    ax[2].imshow(edges_opencv, cmap='gray')  
    ax[2].set_title('Зображення з контурами (OpenCV)')  
    plt.show()
