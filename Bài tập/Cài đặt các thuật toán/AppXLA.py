import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np

# Biến toàn cục để lưu ảnh gốc và ảnh đã lọc
img = None
img_array = None

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
        img = Image.open(file_path)
        
        # Hiển thị ảnh gốc
        img_tk = ImageTk.PhotoImage(img)
        label_original.config(image=img_tk)
        label_original.image = img_tk
        label_original.photo = img_tk  # Lưu giữ tham chiếu tới ảnh để tránh bị thu nhỏ

def apply_filter():
    global img, img_array
    if img is not None:
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16

        filtered_img_array = weighted_average_filter(img_array, kernel)
        filtered_img = Image.fromarray(filtered_img_array.astype(np.uint8))

        # Hiển thị ảnh đã lọc
        filtered_img_tk = ImageTk.PhotoImage(filtered_img)
        label_filtered.config(image=filtered_img_tk)
        label_filtered.image = filtered_img_tk
        label_filtered.photo = filtered_img_tk  # Lưu giữ tham chiếu tới ảnh đã lọc

# Tạo cửa sổ Tkinter với thiết kế
root = tk.Tk()
root.title("Ứng dụng Xử lý ảnh cơ bản")
root.geometry("900x600")  # Kích thước cửa sổ chính
root.configure(bg="#F0F0F0")

# Tiêu đề ứng dụng
title_label = Label(root, text="Trực quan các thuật toán XLA - NVT", font=("Arial", 24, "bold"), bg="#F0F0F0", fg="#333333")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Tạo các label để hiển thị ảnh
label_original = Label(root, text="Ảnh gốc", font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
label_original.grid(row=1, column=0, padx=20, pady=10)

label_filtered = Label(root, text="Ảnh sau khi xử lý", font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
label_filtered.grid(row=1, column=1, padx=20, pady=10)

# Nút để tải ảnh
btn_load = Button(root, text="Tải ảnh lên", command=load_image, font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
btn_load.grid(row=2, column=0, pady=20)

# Nút để áp dụng bộ lọc
btn_filter = Button(root, text="Áp bộ lọc", command=apply_filter, font=("Arial", 12), bg="#008CBA", fg="white", width=15)
btn_filter.grid(row=2, column=1, pady=20)

# Chạy giao diện
root.mainloop()
