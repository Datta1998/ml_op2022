from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


@app.route("/predict", methods=['POST'])
def predict_digit():
    digit_image1 = request.json['image']
    digit_image2 = request.json['image']
    print("done loading")
    predicted_image1 = model.predict([digit_image1])
    predicted_image2 = model.predict([digit_image2])
    if predicted_image1==predicted_image2:
        return "Same Class Images are found here"
    else:
        return "Same Class Images are not found here"    


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001)