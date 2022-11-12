from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/sum')

def sum(x,y):
    z=x+y
    return z    

if __name__ == '__main__':
    app.run(debug=True)


