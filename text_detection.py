import yaml
import easyocr

with open("params.yaml", 'r') as file:
    config = yaml.safe_load(file)

paths = config['paths']
params = config["text_detection"]


IMAGE_PATH = paths["preprocessed_id_image"]
CONFIDENCE_THRESHOLD = params["confidence_threshold"]

def extract_text(image_path, confidence_threshold, languages=['en']):
    """
    Extracts and filters text from an image using OCR, based on a confidence threshold.

    Parameters:
    - image_path (str): Path to the image file.
    - confidence_threshold (float): Minimum confidence for text inclusion. Default is 0.3.
    - languages (list): OCR languages. Default is ['en'].

    Returns:
    - str: Filtered text separated by '|' if confidence is met, otherwise an empty string.

    Raises:
    - Exception: Outputs error message if OCR processing fails.
    """
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages)
    
    try:
        # Read the image and extract text
        result = reader.readtext(image_path)
        filtered_text = ""  # Initialize an empty string to store filtered text
        for text in result:
            bounding_box, recognized_text, confidence = text
            if confidence > confidence_threshold:
                filtered_text += recognized_text + "|"  # Append filtered text with newline

        return filtered_text 
    except Exception as e:
        print("An error occurred during text extraction:", e)
        return ""


    # Filter the extracted text based on confidence score


print(extract_text(IMAGE_PATH,CONFIDENCE_THRESHOLD))