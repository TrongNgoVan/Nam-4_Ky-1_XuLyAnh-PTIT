import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np


def weighted_average_filter(image_array, kernel):
    height, width = image_array.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    result = np.zeros_like(image_array)

    padded_image = np.pad(image_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + k_height, j:j + k_width]
            result[i, j] = np.sum(region * kernel)

    return result


def load_image():
    global img, img_array
    file_path = filedialog.askopenfilename()

    if file_path:
        img = Image.open(file_path).convert('L')  # Chuyển ảnh thành thang độ xám
        img_array = np.array(img, dtype=np.float32)

        # Hiển thị ảnh gốc
        img_tk = ImageTk.PhotoImage(img)
        label_original.config(image=img_tk)
        label_original.image = img_tk


def apply_filter():
    if img_array is not None:
        # Áp dụng bộ lọc
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16

        filtered_img_array = weighted_average_filter(img_array, kernel)
        filtered_img = Image.fromarray(filtered_img_array.astype(np.uint8))

        # Hiển thị ảnh đã xử lý
        filtered_img_tk = ImageTk.PhotoImage(filtered_img)
        label_filtered.config(image=filtered_img_tk)
        label_filtered.image = filtered_img_tk


# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Image Filtering App")
root.geometry("800x400")

# Tạo các label để hiển thị ảnh
label_original = Label(root)
label_original.grid(row=0, column=0, padx=20, pady=20)

label_filtered = Label(root)
label_filtered.grid(row=0, column=1, padx=20, pady=20)

# Nút để tải ảnh
btn_load = Button(root, text="Load Image", command=load_image)
btn_load.grid(row=1, column=0)

# Nút để áp dụng bộ lọc
btn_filter = Button(root, text="Apply Filter", command=apply_filter)
btn_filter.grid(row=1, column=1)

# Chạy giao diện
root.mainloop()
