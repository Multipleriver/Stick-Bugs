from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

#cap = cv2.VideoCapture(0)

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model

black_image = np.zeros((360, 640, 3), dtype=np.uint8)

# Predict with the model
results = model("0", stream=True)  # predict on an image

# Access the results
#for result in results:
#    xy = result.keypoints.xy  # x and y coordinates
#    xyn = result.keypoints.xyn  # normalized
#    kpts = result.keypoints.data  # x, y, visibility (if available)
#    print(xy,xyn,kpts)
for i, r in enumerate(results):
    # Plot results image
    
    im_bgr = r.plot()#img=black_image,labels=False,boxes=False,masks=False)  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    xy = r.keypoints.xy  # x and y coordinates
    print(xy)
    # Show results to screen (in supported environments)
    cv2.imshow("Results", im_bgr)
    cv2.waitKey(1)
    # Save results to disk
    # im_rgb.save(f"results{i}.jpg")
