import base64

with open("TermoPy_logo.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

print(encoded_image)