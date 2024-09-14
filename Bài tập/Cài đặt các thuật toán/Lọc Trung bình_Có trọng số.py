import numpy as np
from PIL import Image


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


img = Image.open('input/weighted_average_filter.jpg').convert('L')
img_array = np.array(img, dtype=np.float32)

kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16

filtered_img = weighted_average_filter(img_array, kernel)

filtered_image = Image.fromarray(filtered_img.astype(np.uint8))
filtered_image.show()
