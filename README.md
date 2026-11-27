# Exp-12-Face-Detection-with-Haar-Cascades
## Name: AYSHWARIYA J
## Reg no: 212224230030
## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)


## Algorithm

1. Import the required libraries such as OpenCV, NumPy, Matplotlib and OS.

2. Read the input image using cv2.imread().

3. Convert the original image from BGR to RGB and display it.

4. Extract the Region of Interest (ROI) using array slicing.

5. Create a mask and perform image segmentation using cv2.bitwise_and().

6. Convert the image into grayscale using cv2.cvtColor().

7. Apply Gaussian Blur to reduce noise in the image.

8. Perform Canny Edge Detection and find contours using OpenCV functions.

9. Load the MobileNet-SSD model and detect objects with confidence greater than 0.5.

10. Draw bounding boxes with object labels and display the final output images using Matplotlib.

  

## Program

## I) ROI Segmentation in an Image using Bitwise AND
``` 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread('Qno.1.jpg')

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()

# Step 3: Define the Region of Interest (ROI)
roi = image[100:420, 200:550]

# Step 4: Create a blank mask
mask = np.zeros_like(image)

# Step 5: Place ROI on the mask
mask[100:420, 200:550] = roi

# Step 6: Perform bitwise AND
segmented_roi = cv2.bitwise_and(image, mask)

# Step 7: Display the segmented ROI
segmented_roi_rgb = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2RGB)

plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()
```

## II) Handwriting Detection in an Image
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read image
image = cv2.imread('your_image_1.jpg')

if image is None:
    raise FileNotFoundError("Image not found. Check image path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Original Image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Step 2: Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Apply Canny edge detection
edges = cv2.Canny(blurred_image, 50, 150)

# Display Canny Edge Detection
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

# Step 5: Find contours
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Step 6: Filter contours and draw bounding boxes
result_image = image.copy()

for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

# Step 7: Display handwriting detection result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
plt.show()
```

## III) Object Detection with Labels using MobileNet-SSD
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set configuration and weights files
config_file = 'deploy.prototxt'
weights = 'mobilenet_iter_73000.caffemodel'

# Step 2: Load pretrained DNN model
net = cv2.dnn.readNetFromCaffe(config_file, weights)

# Step 3: Define class labels
class_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep',
                18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Step 4: Read image
image = cv2.imread('object.jpg')

if image is None:
    raise FileNotFoundError("Image not found. Check image path.")

(h, w) = image.shape[:2]

# Step 5: Convert image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 6: Create blob for DNN processing
blob = cv2.dnn.blobFromImage(
    image, 0.007843, (300, 300), 127.5
)
# Step 7: Flatten the index, confidence
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:  # Confidence threshold
        index = int(detections[0, 0, i, 1])  # Get class index
        label = class_labels[index]  # Get label name
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
# Step 8: Draw rectangles and labels on the image
        cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image_rgb, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        

# Step 9: Display output
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()


```
## Output
<img width="752" height="660" alt="image" src="https://github.com/user-attachments/assets/e39e94f6-be2d-4162-b94b-e23efebf36bb" />
<img width="658" height="637" alt="image" src="https://github.com/user-attachments/assets/36b3749b-44ea-4b7f-b7c4-5481f3ca451e" />
<img width="846" height="397" alt="image" src="https://github.com/user-attachments/assets/87702725-2ab7-4e4c-97e1-67d713224574" />
<img width="887" height="385" alt="image" src="https://github.com/user-attachments/assets/fdbe2639-3f0c-4e58-ac2e-930dcb84e8b3" />
<img width="867" height="393" alt="image" src="https://github.com/user-attachments/assets/93f1a26b-2581-4da5-8ca7-9c7164ccfb46" />
<img width="832" height="635" alt="image" src="https://github.com/user-attachments/assets/ca9414e6-79e2-4e09-807c-13af8fd022e0" />





## Result :
Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
