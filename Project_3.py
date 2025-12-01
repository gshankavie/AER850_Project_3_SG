import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt



# Step 1: object masking
motherboard_image_path = r"/Users/shankavie/Documents/GitHub/AER850_Project_3_SG/Project 3 Data/motherboard_image.JPEG"


img = cv2.imread(motherboard_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)

# detect edges
img_edges = cv2.Canny(blurred_img, 45, 140)

kernel = np.ones((5, 5), np.uint8)
edges_dilated = cv2.dilate(img_edges, kernel, iterations=1)
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)


# detect contours
contours, hierarchy = cv2.findContours(
    edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


large_contours = [c for c in contours if cv2.contourArea(c) > 4500]
pcb_contour = max(large_contours, key=cv2.contourArea)

# create the binary mask
mask = np.zeros_like(gray_image)
cv2.drawContours(mask, [pcb_contour], contourIdx=-1, color=255, thickness=-1)


extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# generate plots
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title("Original Motherboard Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(blurred_img, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(img_edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask (Largest Contour)")
plt.axis("off")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(extracted)
plt.title("Extracted PCB via Mask")
plt.axis("off")
plt.show()



# Step 2: Yolo V11
model = YOLO("yolo11n.pt")

model.train(data=r"/Users/shankavie/Documents/GitHub/AER850_Project_3_SG/Project 3 Data/data/data.yaml" , epochs=75, imgsz = 1200, batch = 2, name = 'project_3_model', device=0) 



# Step 3: model evaluation

model.predict(source=r"/Users/shankavie/Documents/GitHub/AER850_Project_3_SG/Project 3 Data/data/evaluation", conf=0.2, save=True, project=r"/Users/shankavie/Documents/GitHub/AER850_Project_3_SG",
             name="evaluation_results")




