import easyocr

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

files = ["numberplate1_OCR_output.png", "numberplate2_OCR_output.png", "numberplate3_OCR_output.png", "numberplate4_OCR_output.png", "numberplate5_OCR_output.png", "numberplate6_OCR_output.png"]

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

for filename in files:
    try:
        file = "OCR_output_images/" + filename
        print(reader.readtext(file, detail=0))
    except:
        print(f"Error occurred during process the img {filename}")
