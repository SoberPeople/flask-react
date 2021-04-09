
from flask import Flask      # 설치한 Flask 패키지에서 Flask 모듈을 import 하여 사용

app = Flask(__name__)       # 플라스크를 생성하고 app 변수에 flask 초기화 하여 실행

 

@app.route("/")               # 사용자에게 ( ) 에 있는 경로를 안내 해준다고 생각하면 쉬움

def test():

  return "test"                  # 즉 위에 누군가가 / 경로를 요청하여 test 란 함수를 실행하고

                                    # test 라는 결과 값을 보여줌

 

if __name__ == "__main__": 

   app.run()