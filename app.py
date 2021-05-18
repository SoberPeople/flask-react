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

# print("loading yolo-tiny-prn...")
# yolo = YOLO("models/hand/cross-hands-tiny-prn.cfg",
#             "models/hand/cross-hands-tiny-prn.weights", ["hand"])
# yolo.size = int(size)

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

### 추가
# engine = create_engine('mssql+pymssql://username:passwd@host/database', echo=True)

@app.route('/api/detection/', methods=['GET', 'POST'])
def detection():
    frame = request.form.get('file')
    # opencv에서 읽기 위해 8비트 애들을 아스키로 변환
    img_data = np.frombuffer(base64.b64decode(
        frame.replace('data:image/png;base64,', '')), np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_ANYCOLOR)

    if (request.method == 'POST'):
        userId = request.form.get('id')
        print(userId)

        now = datetime.datetime.now().strftime("_%y-%m-%d_%H-%M-%S")

        cheat = 0 # 손 또는 눈 detect 여부
        stid = userId[:7]
        device = userId[8:] # 핸드폰인지 노트북인지 판별
        # print(device)

        output_path = os.path.join(output_dir, str(userId) + str(now) + str(uuid.uuid4()) + ".jpg")
        #+ ".hand."

        ### 손 detect ###
        if((device == "PHONE") | (device == "phone")):    # 핸드폰 화면일 경우
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

                    print("%s with %s confidence" % (name, round(confidence, 2)))
                    
                cv2.putText(frame, stid, (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 0), 2)
                    #147, 58, 31
                
                cv2.imwrite(output_path, frame)
                return json.dumps({"cheat": 1, "output_path": output_path})

            return json.dumps({"cheat": 0, "output_path": output_path})
            
        
        
        ### gaze_Tracking ###

        elif((device == "COM") | (device == "com")) : # 노트북 화면일 경우 
            # frame = mat
            eye_text = ""

            # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame() # 동공 십자가 표시            

            if gaze.is_right():
                cheat = True
                # eye_text = "Looking right"
                eye_text = "right"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_left():
                cheat = True
                # eye_text = "Looking left"   
                eye_text = "left"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_up():
                cheat = True
                # eye_text = "Looking up"
                eye_text = "up"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_center():
                cheat = False
                # eye_text = "Looking center"
                eye_text = "center"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            elif gaze.is_down():
                cheat = False
                # eye_text = "Looking down"
                eye_text = "down"
                # print(eye_text + "  horizontal: " + str(round(gaze.horizontal_ratio(),2))
                #  + "  vertical: " + str(round(gaze.vertical_ratio(),2)))
            else :
                cheat = True

            cv2.putText(frame, stid+ "  " + eye_text, (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 0), 2)

            if cheat:
                cv2.imwrite(output_path, frame)
                return json.dumps({"cheat": 1, "output_path": output_path})
            
            else:
                return json.dumps({"cheat": 0, "output_path": output_path})

        else:
            print("id를 '학번_PHONE/COM' 형식으로 입력하지 않았습니다!")
            
            cv2.putText(frame, stid + "  ID Error", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 0), 2)

            cv2.imwrite(output_path, frame)
            return json.dumps({"cheat": 1, "output_path": output_path})


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
