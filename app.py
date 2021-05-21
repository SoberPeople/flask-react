from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response
import ssl
import cv2
from yolo import YOLO
import json
# import uuid
import os
import base64
import numpy as np
from gaze_tracking import GazeTracking
import datetime
# from sqlalchemy import create_engine

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


network = 'normal'
size = 416
confidence = 0.25
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)


gaze = GazeTracking()

if network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg",
                "models/cross-hands.weights", ["hand"])
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

# 추가
# engine = create_engine('mssql+pymssql://username:passwd@host/database', echo=True)


@app.route('/api/detection/', methods=['GET', 'POST'])
def detection():
    frame = request.form.get('file')

    # 이미지 받아오는 형식에 따라 cv에서 읽을 수 있는 데이터로 변환
    img_data = ""
    if "image/jpeg" in frame:
        img_data = np.frombuffer(base64.b64decode(frame.replace('data:image/jpeg;base64,', '')), np.uint8)
    else:
        img_data = np.frombuffer(base64.b64decode(frame.replace('data:image/png;base64,', '')), np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_ANYCOLOR)

    if (request.method == 'POST'):
        userId = request.form.get('id')
        now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        cheat = 0  # 부정행위이면 1, 아니면 0
        stid = userId[:7]
        device = userId[8:]
        print(now+": "+userId+"의 "+device+"이미지를 받았습니다.")

        output_path = os.path.join(output_dir, str(
            userId) + str(now) + ".jpg")
        #+ ".hand."

        ### 손 detect ###
        if((device == "PHONE") | (device == "phone")):
            width, height, inference_time, results = yolo.inference(frame)

            print("%s seconds: %s classes found!" %
                  (round(inference_time, 2), len(results)))

            # 여기에 사진 저장(값 0이면 캡쳐)
            if len(results) < 2:
                cheat = 1
                # return json.dumps({"cheat": 1})

            if cheat:
                for detection in results:
                    id, name, confidence, x, y, w, h = detection

                    # draw a bounding box rectangle and label on the image
                    color = (255, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    text = "%s (%s)" % (name, round(confidence, 2))
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.25, color, 1)

                    print("%s with %s confidence" %
                          (name, round(confidence, 2)))

                cv2.putText(frame, stid, (20, 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 0), 2)
                #147, 58, 31

                cv2.imwrite(output_path, frame)
                return json.dumps({"cheat": 1})

            return json.dumps({"cheat": 0})

        ### gaze_Tracking ###
        
        elif((device == "COM") | (device == "com")):  # 노트북 화면일 경우
            eye_text = ""
           
            gaze.refresh(frame)     # gaze tracking에 분석위해 frame 보내기
            frame = gaze.annotated_frame()  # 동공 십자가 표시

            if gaze.is_right():
                cheat = True
                eye_text = "right"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_left():
                cheat = True
                eye_text = "left"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_up():
                cheat = True
                eye_text = "up"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_center():
                cheat = False
                eye_text = "center"
            else:
                cheat = False

            cv2.putText(frame, stid + " - " + eye_text, (90, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

            if cheat:
                # left_pupil = gaze.pupil_left_coords()
                # right_pupil = gaze.pupil_right_coords()
                # cv2.putText(frame, "Left pupil:  " + str(left_pupil),
                #         (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
                # cv2.putText(frame, "Right pupil: " + str(right_pupil),
                #         (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

                cv2.imwrite(output_path, frame)
                return json.dumps({"cheat": 1})

            else:
                return json.dumps({"cheat": 0})

        else:
            print("id를 '학번_PHONE/COM' 형식으로 입력하지 않았습니다!")

            cv2.putText(frame, stid + "  ID Error", (20, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 0), 2)

            cv2.imwrite(output_path, frame)
            return json.dumps({"cheat": 1})

        #     return json.dumps({"cheat": 1, "output_path": output_path})

        # return json.dumps({"cheat": 1})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True, ssl_context=(
        './ssl/localhost.crt', './ssl/localhost.key'))
