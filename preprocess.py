import cv2
import numpy as np
import os
import yaml

# Load the parameters from the params.yaml file
with open("params.yaml", 'r') as file:
    config = yaml.safe_load(file)

paths = config['paths']
params = config["preprocess"]

KERNEL_SIZE = params["kernel_size"]
PREPROCESSED_ID_CARD_PATH = paths["preprocessed_id_image"]
IMAGE_PATH = paths["raw_id_image"]

def read_image(image_path, is_uploaded=False):
    if is_uploaded:
        try:
            image_bytes = image_path.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            print("Error reading image:", e)
            return None
    else:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            print("Error reading image:", e)
            return None

def extract_id_card(img):
    """
    Extracts the ID card from an image containing other backgrounds.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The cropped image containing the ID card, or None if no ID card is detected.
    """
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray_img, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    # Select the largest contour (assuming the ID card is the largest object)
    largest_contour = None
    largest_area = 0
    for cnt in contours[2:]:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_contour = cnt
            largest_area = area

    # If no large contour is found, assume no ID card is present
    if largest_contour is None:
        return None, None

    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    filename = os.path.join(PREPROCESSED_ID_CARD_PATH)
    contour_id = img[y:y+h, x:x+w]

    if os.path.exists(filename):
        # Remove the existing file
        os.remove(filename)

    cv2.imwrite(filename, contour_id)

    return contour_id, filename


def save_image(image, filename, path="."):
    # Construct the full path
    full_path = os.path.join(path, filename)
    if os.path.exists(full_path):
        # Remove the existing file
        os.remove(full_path)

    # Save the image using cv2.imwrite
    cv2.imwrite(full_path, image)
    return full_path

image = read_image(IMAGE_PATH)
if image is not None:
    extracted_id_card, saved_path = extract_id_card(image)
    if extracted_id_card is not None:
        print(f"ID card extracted and saved to {saved_path}")
    else:
        print("No ID card detected in the image.")
else:
    print("Failed to read the image.")
