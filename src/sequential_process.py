import os
import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)   # ADDED: force OpenCV to use one thread, so the baseline is truly single-core

# CONFIGURATION
INPUT_DIR = "images_dataset"      # Input dataset folder (class folders inside)
OUTPUT_DIR = "sample_output/output_seq"           # Output folder: resized + watermarked images
EDGES_DIR = "sample_output/output_seq_edges"      # Output folder: Sobel edge images
WATERMARK_TEXT = "ARTI 503 Parallel Project"      # Watermark text
RESIZE_SHAPE = (128, 128)                         # Resize dimensions (width, height)
FONT_PATH = None
BLUR_KERNEL = (5, 5)                              # Gaussian blur window size


def add_watermark(image_pil, text, font_path=None):
    """Add a simple white watermark at the bottom-right corner."""
    draw = ImageDraw.Draw(image_pil)

    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 14)
    else:
        font = ImageFont.load_default()

    # Compatible with both old and new Pillow versions
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    x, y = image_pil.width - text_w - 8, image_pil.height - text_h - 8
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return image_pil


def detect_edges(img_bgr):
    """grayscale -> Gaussian blur -> Sobel edge detection. Returns an 8-bit edge image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def sequential_preprocess():
    """Sequential image processing: grayscale, blur, Sobel, resize, watermark."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EDGES_DIR, exist_ok=True)

    start_time = time.time()

    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        edges_class_path = os.path.join(EDGES_DIR, class_name)
        os.makedirs(edges_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping invalid image: {img_path}")
                    continue

                edges = detect_edges(img)
                cv2.imwrite(os.path.join(edges_class_path, img_name), edges)

                img_resized = cv2.resize(img, RESIZE_SHAPE)
                img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                img_pil = add_watermark(img_pil, WATERMARK_TEXT, FONT_PATH)
                final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(output_class_path, img_name), final_img)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    end_time = time.time()
    print(f"Sequential Processing Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    print("OpenCV threads:", cv2.getNumThreads())   # should print 1
    sequential_preprocess()
