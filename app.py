from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def web(path):
    return render_template('index.html');

if __name__ == "__main__":
  # app.run(debug=True)
  app.run(host="127.0.0.1", port="5000", debug=True)
