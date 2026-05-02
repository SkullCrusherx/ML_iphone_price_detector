from flask import Flask,render_template,request
import os
import joblib

app = Flask(__name__)

# Load model once (IMPORTANT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '2nd_hand_iphone.pkl')

reg = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/iphone', methods=['GET', 'POST'])
def iphone():

    # STEP 1: Show form
    if request.method == 'GET':
        return render_template('iphone.html')

    # STEP 2: Handle form submission
    if request.method == 'POST':
        series = int(request.form.get('Series'))
        ram = int(request.form.get('RAM_GB'))
        screen = float(request.form.get('Screen_Size'))
        battery = int(request.form.get('Battery_mAh'))
        storage = int(request.form.get('Storage_GB'))
        year = int(request.form.get('Release_Year'))

        # IMPORTANT: Feature order must match training
        features = [[series, ram, screen, battery, storage, year]]

        prediction = reg.predict(features)
        prediction  = str(int(prediction[0][0]))

        return render_template('result.html',result=prediction,series=series,ram=ram,screen=screen,battery=battery,storage=storage,year=year)



if __name__ == '__main__':
    app.run()