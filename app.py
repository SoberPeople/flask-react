from flask import Flask, render_template, send_from_directory
import ssl;

app = Flask(__name__)

@app.route('/')
def main():
  return "Hello page"

# @app.route('/', defaults={'path': ''})
@app.route('/index/<path:path>')
def web(path):
    return render_template('index.html');

@app.route('/host')
def host():
  return render_template('host.html')

@app.route('/guest')
def guest():
  return render_template('guest.html')


if __name__ == "__main__":
  app.run(host="127.0.0.1", port="5000", debug=True, ssl_context=(
      '/Users/gaeun/ssl/localhost.crt', '/Users/gaeun/ssl/localhost.key'))
