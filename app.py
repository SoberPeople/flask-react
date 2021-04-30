from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response
import ssl
import cv2
from yolo import YOLO
import json
import uuid
import os
import base64
import numpy as np
from gaze_tracking import GazeTracking

app = Flask(__name__)


@app.route('/')
def web():
    return render_template('index.html')


@app.route('/test/host')
def host():
    return render_template('host.html')


@app.route('/test/guest')
def guest():
    return render_template('guest.html')


network = 'prn'
size = 416
confidence = 0.25
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

# print("loading yolo-tiny-prn...")
# yolo = YOLO("models/hand/cross-hands-tiny-prn.cfg",
#             "models/hand/cross-hands-tiny-prn.weights", ["hand"])
# yolo.size = int(size)

gaze = GazeTracking()

if network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/hand/cross-hands.cfg",
                "models/hand/cross-hands.weights", ["hand"])
elif network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg",
                "models/cross-hands-tiny-prn.weights", ["hand"])
elif network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg",
                "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg",
                "models/cross-hands-tiny.weights", ["hand"])

yolo.confidence = float(confidence)


@app.route('/api/detection/', methods=['GET', 'POST'])
def detection():
    if (request.method == 'POST'):
        mat = request.form.get('file')
        userId = request.form.get('id')
        #opencv에서 읽기 위해 8비트 애들을 아스키로 변환
        img_data = np.frombuffer(base64.b64decode(mat.replace('data:image/png;base64,','')), np.uint8)
        mat = cv2.imdecode(img_data,cv2.IMREAD_ANYCOLOR)
        width, height, inference_time, results = yolo.inference(mat)

        print("%s seconds: %s classes found!" %
              (round(inference_time, 2), len(results)))

        # 여기에 사진 저장(값 0이면 캡쳐)
        if len(results) < 1:
            return json.dumps({"hand": 0})

        for detection in results:
            id, name, confidence, x, y, w, h = detection

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, color, 1)

            print("%s with %s confidence" % (name, round(confidence, 2)))

        ### gaze_Tracking ###
        # while True:
        # We get a new frame from the webcam
        # _,
        frame = mat

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
        elif gaze.is_up():
            text = "Looking up"

        cv2.putText(frame, text, (90, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil),
                    (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil),
                    (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # if cv2.waitKey(1) == 27:
        #     break
        output_path = os.path.join(output_dir, str(userId) + str(uuid.uuid4()) + ".jpg")
        # cv2.imwrite(output_path, frame)
        return json.dumps({"hand": len(results), "output_path": output_path})


# @app.route('/', defaults={'path': ''})
# @app.route('/<path>')
# def web(path):
#     if(path == 'first' or path == 'second'):
#         return render_template('index.html')
#     elif(path == 'host'):
#         return render_template('host.html')
#     elif(path == 'guest'):
#         return render_template('guest.html')
#     else:
#         return "Hello Page"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True, ssl_context=(
        './ssl/localhost.crt', './ssl/localhost.key'))
