# Object-Detection-System
---------------------------------------------------------------------------------------
#Demo

![](https://github.com/developer-venish/Object-Detection-System/blob/main/1.png)

![](https://github.com/developer-venish/Object-Detection-System/blob/main/2.png)

![](https://github.com/developer-venish/Object-Detection-System/blob/main/3.png)

---------------------------------------------------------------------------------------


```python
import cv2 
import matplotlib.pyplot as plt
```

1. **Import Libraries:**
   - Import the OpenCV library (`cv2`) for computer vision tasks.
   - Import the `matplotlib.pyplot` library for plotting images.

```python
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = 'frozen_inference_graph.pb'
```

2. **Specify Model Files:**
   - Define the paths for the configuration file (`ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`) and the frozen model file (`frozen_inference_graph.pb`).

```python
model = cv2.dnn_DetectionModel(frozen_model, config_file)
```

3. **Load Object Detection Model:**
   - Use OpenCV's `dnn_DetectionModel` to load the pre-trained object detection model.

```python
classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
```

4. **Load Class Labels:**
   - Read class labels from the "labels.txt" file and store them in the `classLabels` list.

```python
print(classLabels)
print(len(classLabels))
```

5. **Print Class Labels:**
   - Print the loaded class labels and the total number of classes.

```python
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127, 5, 127.5))
model.setInputSwapRB(True)
```

6. **Configure Model Input:**
   - Set input size, scale, mean, and swap Red and Blue channels for the loaded model.

```python
img = cv2.imread('boy.jpg')
plt.imshow(img)
```

7. **Read and Display an Image:**
   - Read an image ('boy.jpg') using OpenCV's `imread` and display it using `plt.imshow`.

```python
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
print(ClassIndex)
```

8. **Object Detection on Image:**
   - Perform object detection on the image using the loaded model with a confidence threshold of 0.5.
   - Print the detected class indices.

```python
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
```

9. **Define Font for Annotations:**
   - Set the font scale and font type for annotating detected objects.

```python
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
```

10. **Annotate Detected Objects on Image:**
    - Iterate through detected objects, draw rectangles around them, and annotate with class labels.

```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

11. **Display Annotated Image:**
    - Display the image with annotated objects using `plt.imshow` after converting the color format from BGR to RGB.

```python
import cv2

cap = cv2.VideoCapture("pexels-george-norina-5330833.mp4")

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()
```

12. **Open Video Capture:**
    - Open a video file ("pexels-george-norina-5330833.mp4") using OpenCV's `VideoCapture`.

```python
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
```

13. **Define Font for Video Annotations:**
    - Set the font scale and font type for annotating detected objects in video frames.

```python
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the frame.")
        break
```

14. **Read Video Frames:**
    - Read frames from the video file in a loop. If reading fails, print an error message and break the loop.

```python
ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
print(ClassIndex)
```

15. **Object Detection on Video Frames:**
    - Perform object detection on the video frame using the loaded model with a confidence threshold of 0.55.
    - Print the detected class indices.

```python
if len(ClassIndex) != 0:
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(frame, boxes, (255, 0, 0), 2)
        cv2.putText(frame, classLabels[ClassInd - 1], (int(boxes[0]) + 10, int(boxes[1]) + 40),
                    font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
```

16. **Annotate Detected Objects in Video Frame:**
    - If objects are detected, iterate through them, draw rectangles, and annotate with class labels.

```python
cv2.imshow("Object Detection by OpenCV", frame)

if cv2.waitKey(2) & 0xFF == ord('q'):
    break
```

17. **Display Video Frame with Annotations:**
    - Display the video frame with annotated objects using `cv2.imshow`.
    - Break the

 loop if the 'q' key is pressed.

```python
cap.release()
cv2.destroyAllWindows()
```

18. **Release Video Capture and Close Windows:**
    - Release the video capture and close all OpenCV windows when the script is finished.
