import tkinter as tk
from tkinter import filedialog, Label, Button, ttk, Scale
from PIL import Image, ImageTk
import numpy as np
from collections import defaultdict
import heapq
import io

class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý ảnh nâng cao")
        self.root.geometry("1000x700")
        self.root.configure(bg="#F0F0F0")

        self.img = None
        self.img_array = None
        self.threshold_value = 128
        
        self.setup_gui()

    def setup_gui(self):
        # Tiêu đề
        title_label = Label(self.root, text="Trực quan các thuật toán XLA - NVT", 
                          font=("Arial", 24, "bold"), bg="#F0F0F0", fg="#333333")
        title_label.grid(row=0, column=0, columnspan=3, pady=20)

        # Labels để hiển thị ảnh
        self.label_original = Label(self.root, text="Ảnh gốc", 
                                  font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
        self.label_original.grid(row=1, column=0, padx=20, pady=10)

        self.label_filtered = Label(self.root, text="Ảnh sau khi xử lý", 
                                  font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
        self.label_filtered.grid(row=1, column=1, padx=20, pady=10)

        # Frame cho các controls
        control_frame = tk.Frame(self.root, bg="#F0F0F0")
        control_frame.grid(row=2, column=0, columnspan=2, pady=20)

        # Nút tải ảnh
        btn_load = Button(control_frame, text="Tải ảnh lên", 
                         command=self.load_image, font=("Arial", 12),
                         bg="#4CAF50", fg="white", width=15)
        btn_load.pack(side=tk.LEFT, padx=10)

        # Dropdown cho lựa chọn bộ lọc
        self.filter_var = tk.StringVar()
        self.filter_choices = {
            "Lọc trung bình có trọng số": self.weighted_average_filter,
            "Phép giãn (Dilation)": self.dilation_filter,
            "Phép co (Erosion)": self.erosion_filter,
            "Cân bằng histogram": self.histogram_equalization_filter,
            "Mã hóa Huffman": self.huffman_encoding_filter,
            "Lọc trung vị": self.median_filter,
            "Ảnh âm bản": self.negative_image_filter,
            "Phép mở (Opening)": self.opening_filter,
            "Toán tử Prewitt": self.prewitt_operator_filter,
            "Toán tử Roberts": self.roberts_operator_filter,
            "Toán tử Sobel": self.sobel_operator_filter,
            "Phân ngưỡng": self.thresholding_filter
        }
        
        self.filter_dropdown = ttk.Combobox(control_frame, 
                                          textvariable=self.filter_var,
                                          values=list(self.filter_choices.keys()),
                                          font=("Arial", 12),
                                          width=25)
        self.filter_dropdown.set("Chọn phép toán xử lý")
        self.filter_dropdown.pack(side=tk.LEFT, padx=10)

        # Thanh trượt cho ngưỡng
        self.threshold_frame = tk.Frame(control_frame, bg="#F0F0F0")
        self.threshold_frame.pack(side=tk.LEFT, padx=10)
        self.threshold_scale = Scale(self.threshold_frame, 
                                   from_=0, to=255, 
                                   orient=tk.HORIZONTAL,
                                   label="Ngưỡng",
                                   command=self.update_threshold)
        self.threshold_scale.set(128)
        self.threshold_frame.pack_forget()  # Ẩn ban đầu

        # Nút áp dụng bộ lọc
        btn_apply = Button(control_frame, text="Áp dụng", 
                          command=self.apply_selected_filter,
                          font=("Arial", 12), bg="#008CBA", fg="white", width=15)
        btn_apply.pack(side=tk.LEFT, padx=10)

        # Bind event khi chọn filter
        self.filter_dropdown.bind('<<ComboboxSelected>>', self.on_filter_select)

    def on_filter_select(self, event=None):
        # Hiển thị thanh trượt ngưỡng chỉ khi chọn phép toán phân ngưỡng
        if self.filter_var.get() == "Phân ngưỡng":
            self.threshold_frame.pack(side=tk.LEFT, padx=10)
        else:
            self.threshold_frame.pack_forget()

    def update_threshold(self, value):
        self.threshold_value = int(value)
        if self.filter_var.get() == "Phân ngưỡng":
            self.apply_selected_filter()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            # Hiển thị ảnh gốc
            img_tk = ImageTk.PhotoImage(self.img)
            self.label_original.config(image=img_tk)
            self.label_original.image = img_tk
            self.label_original.photo = img_tk

    def apply_selected_filter(self):
        if self.img is not None and self.filter_var.get() in self.filter_choices:
            filter_func = self.filter_choices[self.filter_var.get()]
            filter_func()

    # Các phương thức xử lý ảnh
    def weighted_average_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16
        result = self.apply_kernel_filter(img_array, kernel)
        self.display_result(result)

    def histogram_equalization_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        
        histogram, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        cdf = histogram.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
        
        result = cdf[img_array]
        self.display_result(result)

    def huffman_encoding_filter(self):
        # Chuyển ảnh thành bytes
        img_byte_arr = io.BytesIO()
        self.img.save(img_byte_arr, format=self.img.format)
        image_data = img_byte_arr.getvalue()
        
        # Mã hóa
        encoded_data, tree = self.huffman_encode(image_data)
        # Giải mã
        decoded_data = self.huffman_decode(encoded_data, tree)
        
        # Chuyển về ảnh
        bytes_io = io.BytesIO(decoded_data)
        decoded_img = Image.open(bytes_io)
        
        # Hiển thị
        img_tk = ImageTk.PhotoImage(decoded_img)
        self.label_filtered.config(image=img_tk)
        self.label_filtered.image = img_tk
        self.label_filtered.photo = img_tk

    def median_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        result = self.apply_median_filter(img_array, 3)
        self.display_result(result)

    def negative_image_filter(self):
        img_array = np.array(self.img)
        result = 255 - img_array
        self.display_result(result)
    def dilation_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        kernel = np.ones((5, 5), np.uint8)
        height, width = img_array.shape
        result = np.zeros_like(img_array)
        
        pad = kernel.shape[0] // 2
        padded_image = np.pad(img_array, pad, mode='constant')
        
        for i in range(height):
            for j in range(width):
                window = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = np.max(window * kernel)
        
        self.display_result(result)

    def erosion_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        kernel = np.ones((5, 5), np.uint8)
        height, width = img_array.shape
        result = np.zeros_like(img_array)
        
        pad = kernel.shape[0] // 2
        padded_image = np.pad(img_array, pad, mode='constant')
        
        for i in range(height):
            for j in range(width):
                window = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = np.min(window * kernel)
        
        self.display_result(result)

    def opening_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        kernel = np.ones((5, 5), np.uint8)
        
        # Thực hiện phép co trước
        eroded = self.apply_filter(img_array, kernel, 'erosion')
        # Sau đó thực hiện phép giãn
        result = self.apply_filter(eroded, kernel, 'dilation')
        
        self.display_result(result)

    def prewitt_operator_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        
        kernel_x = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])
        
        gradient_x = self.apply_kernel_filter(img_array, kernel_x)
        gradient_y = self.apply_kernel_filter(img_array, kernel_y)
        
        result = np.sqrt(gradient_x**2 + gradient_y**2)
        self.display_result(result)

    def roberts_operator_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        height, width = img_array.shape
        result = np.zeros((height-1, width-1))
        
        for i in range(height-1):
            for j in range(width-1):
                gx = img_array[i+1, j+1] - img_array[i, j]
                gy = img_array[i+1, j] - img_array[i, j+1]
                result[i, j] = np.sqrt(gx**2 + gy**2)
        
        self.display_result(result)

    def sobel_operator_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        
        kernel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
        
        gradient_x = self.apply_kernel_filter(img_array, kernel_x)
        gradient_y = self.apply_kernel_filter(img_array, kernel_y)
        
        result = np.sqrt(gradient_x**2 + gradient_y**2)
        self.display_result(result)

    def thresholding_filter(self):
        img_gray = self.img.convert('L')
        img_array = np.array(img_gray)
        result = np.where(img_array > self.threshold_value, 255, 0)
        self.display_result(result)

    # Các phương thức hỗ trợ
    def apply_kernel_filter(self, image_array, kernel):
        height, width = image_array.shape
        k_height, k_width = kernel.shape
        pad_h = k_height // 2
        pad_w = k_width // 2
        
        padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        result = np.zeros_like(image_array)
        
        for i in range(height):
            for j in range(width):
                region = padded_image[i:i+k_height, j:j+k_width]
                result[i, j] = np.sum(region * kernel)
        
        return result

    def apply_median_filter(self, image_array, kernel_size):
        height, width = image_array.shape
        pad = kernel_size // 2
        padded_image = np.pad(image_array, pad, mode='constant')
        result = np.zeros_like(image_array)
        
        for i in range(height):
            for j in range(width):
                window = padded_image[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = np.median(window)
        
        return result

    def huffman_encode(self, data):
        freq = defaultdict(int)
        for symbol in data:
            freq[symbol] += 1
            
        heap = [HuffmanNode(symbol, freq) for symbol, freq in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merge = HuffmanNode(None, left.freq + right.freq)
            merge.left = left
            merge.right = right
            heapq.heappush(heap, merge)
            
        return self.build_codes(heap[0]), heap[0]

    def build_codes(self, root):
        codes = {}
        
        def generate_codes(node, code):
            if node.symbol is not None:
                codes[node.symbol] = code
                return
            generate_codes(node.left, code + '0')
            generate_codes(node.right, code + '1')
            
        generate_codes(root, '')
        return ''.join(codes.get(symbol, '') for symbol in range(256))

    def huffman_decode(self, encoded_data, tree):
        decoded_data = []
        current_node = tree
        
        for bit in encoded_data:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
                
            if current_node.symbol is not None:
                decoded_data.append(current_node.symbol)
                current_node = tree
                
        return bytes(decoded_data)

    def display_result(self, result_array):
        """
        Hiển thị kết quả xử lý ảnh
        Params:
            result_array: Mảng numpy chứa ảnh đã xử lý
        """
        # Normalize kết quả về range [0, 255]
        if result_array.dtype == np.float32 or result_array.dtype == np.float64:
            result_array = np.clip(result_array, 0, 255)
        
        # Chuyển về kiểu uint8
        result_array = result_array.astype(np.uint8)
        
        # Tạo ảnh PIL từ array
        filtered_img = Image.fromarray(result_array)
        
        # Resize ảnh để hiển thị phù hợp
        display_size = (400, 400)  # Kích thước hiển thị cố định
        filtered_img = filtered_img.resize(display_size, Image.Resampling.LANCZOS)
        
        # Chuyển đổi sang PhotoImage để hiển thị trong Tkinter
        filtered_img_tk = ImageTk.PhotoImage(filtered_img)
        
        # Cập nhật label
        self.label_filtered.config(image=filtered_img_tk)
        self.label_filtered.image = filtered_img_tk
        self.label_filtered.photo = filtered_img_tk  # Giữ tham chiếu


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()