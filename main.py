import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.special import gamma
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def load_image(image_path):
    img = Image.open(image_path)
    img = np.asarray(img) / 255.0
    return img

def apply_gaussian_filter(img, sigma=1):
    if sigma > 0:
        img_filtered = gaussian_filter(img, sigma=sigma)
    else:
        img_filtered = img
    return img_filtered

def fractional_derivative(img, alpha, axis):
    if axis not in [0, 1]:
        raise ValueError("Вісь повинна бути 0 (в напрямку x) або 1 (в напрямку y)")
    
    kernel_size = 21
    half_size = kernel_size // 2
    
    n = np.arange(-half_size, half_size + 1)
    kernel = np.where(n % 2 == 0, 1, -1) * gamma(alpha + 1) / (gamma(n + half_size + 1) * gamma(alpha - n - half_size + 1))
    kernel *= n**alpha
    
    if axis == 0:
        kernel = kernel.reshape(1, -1)
    else:
        kernel = kernel.reshape(-1, 1)
    
    return convolve(img, kernel, mode='nearest')

def compute_gradients(img, alpha):
    fd_x_r, fd_x_g, fd_x_b = [fractional_derivative(img[:, :, i], alpha, axis=0) for i in range(3)]
    fd_y_r, fd_y_g, fd_y_b = [fractional_derivative(img[:, :, i], alpha, axis=1) for i in range(3)]
    
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
            try:
                q = 255
                r = 255
                
                # Горизонтальна орієнтація (0 градусів)
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j + 1]
                    r = G[i, j - 1]
                # Діагональна орієнтація 135 градусів
                elif (22.5 <= angle[i, j] < 67.5):
                    q = G[i + 1, j - 1]
                    r = G[i - 1, j + 1]
                # Вертикальна орієнтація 90 градусів
                elif (67.5 <= angle[i, j] < 112.5):
                    q = G[i + 1, j]
                    r = G[i - 1, j]
                # Діагональна орієнтація 45 градусів
                elif (112.5 <= angle[i, j] < 157.5):
                    q = G[i - 1, j - 1]
                    r = G[i + 1, j + 1]

                if (G[i, j] >= q) and (G[i, j] >= r):
                    suppressed[i, j] = G[i, j]
                else:
                    suppressed[i, j] = 0

            except IndexError as e:
                pass

    return suppressed

def threshold_and_link_edges(G, threshold):
    return G > threshold

def edge_detection(image_path, alpha, sigma=1, threshold=0.5):
    img = load_image(image_path)
    img_filtered = apply_gaussian_filter(img, sigma)
    G, theta = compute_gradients(img_filtered, alpha)
    G_suppressed = non_maximal_suppression(G, theta)
    edges = threshold_and_link_edges(G_suppressed, threshold)
    
    # Відображення початкового зображення і зображення з визначеними контурами
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Початкове зображення')
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Зображення з контурами')
    plt.show()

def open_image(root, img_label):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = load_image(file_path)
        img_label.config(image=ImageTk.PhotoImage(Image.fromarray((img * 255).astype(np.uint8))))
        img_label.image = ImageTk.PhotoImage(Image.fromarray((img * 255).astype(np.uint8)))
        
        # Додамо виклик функції edge_detection з потрібними параметрами
        edge_detection(file_path, alpha=float(alpha_var.get()), sigma=1, threshold=float(threshold_var.get()))

root = tk.Tk()
root.title("Edge Detection")
root.geometry("800x600")
img_label = tk.Label(root)
img_label.pack()

open_button = tk.Button(root, text="Open Image", command=lambda: open_image(root, img_label))
open_button.pack()

alpha_label = tk.Label(root, text="Alpha:")
alpha_label.pack()
alpha_var = tk.StringVar(root, value="0.4")
alpha_entry = tk.Entry(root, textvariable=alpha_var)
alpha_entry.pack()

threshold_label = tk.Label(root, text="Threshold:")
threshold_label.pack()
threshold_var = tk.StringVar(root, value="0.1")
threshold_entry = tk.Entry(root, textvariable=threshold_var)
threshold_entry.pack()

root.mainloop()
