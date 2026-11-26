import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title, is_gray=False):
    """Display an image using Matplotlib with proper color conversion."""
    plt.figure(figsize=(6, 6))
    if is_gray:
        plt.imshow(image, cmap="gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # Ensure output directory exists
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the image from the original_images folder
    image_path = "original_images/example.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from '{image_path}'. Make sure the file exists.")
        return

    display_image(image, "Original Image")

    # 2 & 3. Convert to grayscale and display
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray_image, "Grayscale Image", is_gray=True)

    # 4. Crop the image (rows 100:300, columns 200:400)
    # Clamping bounds safely to image dimensions in case the sample image is smaller
    h, w = image.shape[:2]
    y1, y2 = min(100, h), min(300, h)
    x1, x2 = min(200, w), min(400, w)
    cropped_image = image[y1:y2, x1:x2]
    display_image(cropped_image, "Cropped Image")

    # 5. Rotate the original image by 45 degrees
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    display_image(rotated_image, "Rotated Image (45°)")

    # 6. Increase brightness by adding 50 to all pixel values
    brightness_matrix = np.ones(image.shape, dtype="uint8") * 50
    brighter_image = cv2.add(image, brightness_matrix)
    display_image(brighter_image, "Brighter Image (+50)")

    # 7. Save all transformed images to the output_images folder
    cv2.imwrite(f"{output_dir}/grayscale_image.jpg", gray_image)
    cv2.imwrite(f"{output_dir}/cropped_image.jpg", cropped_image)
    cv2.imwrite(f"{output_dir}/rotated_image.jpg", rotated_image)
    cv2.imwrite(f"{output_dir}/brighter_image.jpg", brighter_image)

    print(f"All transformed images have been saved to '{output_dir}/'.")


if __name__ == "__main__":
    main()