from PIL import Image
import numpy as np

# Load the image (share)
image_path = "secret_3.png"
img = Image.open(image_path)
img_array = np.array(img)

# Modify some pixels (tampering)
img_array[0:10, 0:10] = np.random.randint(0, 256, (10, 10, 3))  

# Save the tampered image
tampered_image_path = "secret_3.png"
tampered_img = Image.fromarray(img_array)
tampered_img.save(tampered_image_path)
