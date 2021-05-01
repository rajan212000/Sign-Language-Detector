import tkinter as tk
from PIL import ImageTk
from tkinter import *
from keras.models import model_from_json
import operator
import cv2
import os

def predict():
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
     # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)

        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]

        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))

        prediction = {'Hello': result[0][0],
                        'I love you': result[0][1],
                        'No': result[0][2],
                        'Thank you': result[0][3],
                        'Yes': result[0][4], }

        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

        # Displaying the predictions
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        interrupt = cv2.waitKey(10)

        if interrupt & 0xFF == 27:  # esc key
            break
    cap.release()
    cv2.destroyAllWindows()


root=Tk()
root.title("Sign language to text ")
root.geometry("1300x690+20+0")
logo_icon = ImageTk.PhotoImage(file="download.jfif")

# ======BG Image====
bg = ImageTk.PhotoImage(file="a.jpg")
bg_image = Label(root, image=bg).place(x=0, y=0, relwidth=1, relheight=1)
        # ============================title Frame=========================================
title = Label(root, text="Sign Language To Text Conversion ", padx=40, image=logo_icon,
                      compoun=LEFT, bd=10,
                      relief=GROOVE, font=("impact", 40,), bg="#154360", fg="white", anchor="w")
title.pack(side=TOP, fill=X)
# ====== Login Frame=====
Frame_login = Frame(root, bg="yellow")
Frame_login.place(x=5, y=300, height=150, width=2000)
title = Label(Frame_login, text="Convert Here", font=("Impact", 25, "bold","underline"), fg="#d77337", bg="yellow").place(x=600,
                                                                                                                   y=30)

forget_btn = Button(Frame_login, command=predict, cursor="hand2", text="convert", bg="black",
                            fg="#d77337", bd=4, relief="ridge", font=("times new roman", 25)).place(x=640, y=80)
root.mainloop()

root.mainloop()
