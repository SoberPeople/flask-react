from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response
import ssl
import cv2
from yolo import YOLO
import json
import uuid
import os
import base64
import numpy as np
from flask_cors import CORS, cross_origin
# from flask.ext.cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app, resources={r"/api/detection/": {"origins": "*"}})

# @app.route('/', defaults={'path': ''})


@app.route('/<path>')
def web(path):
    return render_template('index.html')


@app.route('/test/host')
def host():
    return render_template('host.html')


@app.route('/test/guest')
def guest():
    return render_template('guest.html')


network = 'normal'
size = 416
confidence = 0.25
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

print("loading yolo-tiny-prn...")
yolo = YOLO("models/hand/cross-hands-tiny-prn.cfg",
            "models/hand/cross-hands-tiny-prn.weights", ["hand"])
yolo.size = int(size)
yolo.confidence = float(confidence)

# if network == "normal":
#     print("loading yolo...")
#     yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
# elif network == "prn":
#     print("loading yolo-tiny-prn...")
#     yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
# elif network == "v4-tiny":
#     print("loading yolov4-tiny-prn...")
#     yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
# else:
#     print("loading yolo-tiny...")
#     yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])


@app.route('/api/detection/', methods=['GET', 'POST'])
def detection():
    mat = request.form.get('file')
    img_data = np.frombuffer(base64.b64decode(
        mat.replace('data:image/png;base64,', '')), np.uint8)
    mat = cv2.imdecode(img_data, cv2.IMREAD_ANYCOLOR)

    if (request.method == 'POST'):

        # print(mat[:30])
        # print(request.form)
        # img = cv2.imread(mat)
        # cv2.imwrite('yo.png',img)
        # print(mat)

        # opencv에서 읽기 위해 8비트 애들을 아스키로 변환

        # cv2.imwrite('mm.png', mat)

        width, height, inference_time, results = yolo.inference(mat)

        print("%s seconds: %s classes found!" %
              (round(inference_time, 2), len(results)))

        if len(results) < 1:
            return json.dumps({"nums_of_hand": 0})

        for detection in results:
            id, name, confidence, x, y, w, h = detection

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, color, 1)

            print("%s with %s confidence" % (name, round(confidence, 2)))

# @app.route('/', defaults={'path': ''})
# @app.route('/<path>')
# def web(path):
#     if(path == 'index'):
#         return render_template('index.html')
#     elif(path == 'host'):
#         return render_template('')
#     elif(path == 'guest'):
#         return render_template('guest.html')
#     else:
#         return "Hello Page"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True, ssl_context=(
        './ssl/localhost.crt', './ssl/localhost.key'))
