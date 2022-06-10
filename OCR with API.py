import re
import requests

url = "https://api.mindee.net/v1/products/Jinhua/template_ocr/v1/predict"
files = ["numberplate1_OCR_output.png", "numberplate2_OCR_output.png", "numberplate3_OCR_output.png",
         "numberplate4_OCR_output.png", "numberplate5_OCR_output.png", "numberplate6_OCR_output.png"]

for filename in files:
    file = "OCR_output_images/" + filename
    try:
        with open(file, "rb") as myfile:
            files = {"document": myfile}
            headers = {"Authorization": "Token fc1b51c52af7f507ce549b50df41a61e"}
            response = requests.post(url, files=files, headers=headers)
            find_content = re.compile(r'"content": "(.*?)",')
            result = re.findall(find_content, response.text)
            # print(f'The letters and numbers in the number plate is: {" ".join(result[:len(result)//2])}')
            print(result[:len(result) // 2])
    except:
        print(f"Error occurred during process the img {filename}")

