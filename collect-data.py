import cv2
import numpy
import time
import os

labels=['hello','thank you','i love you','yes','no']

# Create the directory structure
if not os.path.exists("images"):
    os.makedirs("images")
    os.makedirs("images/train")
    os.makedirs("images/train/"+labels[0])
    os.makedirs("images/train/" + labels[1])
    os.makedirs("images/train/" + labels[2])
    os.makedirs("images/train/" + labels[3])
    os.makedirs("images/train/" + labels[4])
parent_dir='images/train/'

# wcam.set(3, 640)  # setting width
# wcam.set(4, 480)  # setting height
# wcam.set(10, 100)  # setting brightness
# count=0

wcam = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    imgnum = 1
    while True:        
        success,frame= wcam.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, label + ':' + str(len(os.listdir(parent_dir+'/'+label))), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            # Coordinates of the ROI
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
            #     # Drawing the ROI
            #     # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            #Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 64))
        cv2.imshow("Frame", frame)
            # do the processing after capturing the image!
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("ROI", roi)
        if imgnum > 100:
            break
        if cv2.waitKey(10) == 13:
            print('Collecting image {}'.format(imgnum))
            img_path = os.path.join(parent_dir,label+'/'+ str(imgnum) + '.jpg')
            cv2.imwrite(img_path, roi)
            imgnum=imgnum+1
        elif cv2.waitKey(10)==27:
            break
    interrupt = cv2.waitKey(0)
    if interrupt & 0xFF == 27:
        break


wcam.release()
cv2.destroyAllWindows()