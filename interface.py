import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import edge_detection

def open_image(root, img_label, threshold_entry, sigma_entry, kernel_size_entry, alpha_entry):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = edge_detection.load_image(file_path)
        img_label.config(image=ImageTk.PhotoImage(Image.fromarray((img * 255).astype(np.uint8))))
        img_label.image = ImageTk.PhotoImage(Image.fromarray((img * 255).astype(np.uint8)))
        
        edge_detection.edge_detection(file_path, 
                       alpha=float(alpha_entry.get()), 
                       sigma=float(sigma_entry.get()), 
                       threshold=float(threshold_entry.get()), 
                       kernel_size=int(kernel_size_entry.get()))  

root = tk.Tk()
root.title("Edge Detection")
root.geometry("800x600")
img_label = tk.Label(root)
img_label.pack()

open_button = tk.Button(root, text="Open Image", command=lambda: open_image(root, img_label, threshold_entry, sigma_entry, kernel_size_entry, alpha_entry))
open_button.pack()

alpha_label = tk.Label(root, text="Alpha:")
alpha_label.pack()
alpha_var = tk.StringVar(root, value="0.4")
alpha_entry = tk.Entry(root, textvariable=alpha_var)
alpha_entry.pack()

sigma_label = tk.Label(root, text="Sigma:")
sigma_label.pack()
sigma_var = tk.StringVar(root, value="2")
sigma_entry = tk.Entry(root, textvariable=sigma_var)
sigma_entry.pack()

threshold_label = tk.Label(root, text="Threshold:")
threshold_label.pack()
threshold_var = tk.StringVar(root, value="0.1")
threshold_entry = tk.Entry(root, textvariable=threshold_var)
threshold_entry.pack()

kernel_size_label = tk.Label(root, text="Kernel Size:")
kernel_size_label.pack()
kernel_size_var = tk.StringVar(root, value="41")
kernel_size_entry = tk.Entry(root, textvariable=kernel_size_var)
kernel_size_entry.pack()

root.mainloop()
