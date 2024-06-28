from flask import *

app = Flask(__name__)

import numpy as np
import joblib
file = 'static\car_price_model.joblib'
model = joblib.load(file)

@app.route('/',methods=['POST','GET'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    if request.method == 'POST':
        year = int(request.form['year'])
        km = int(request.form['km'])
        fuel = float(request.form['fuel_type'])
        seller = float(request.form['seller'])
        transmission = float(request.form['transmission'])
        owner = float(request.form['owner'])
        mileage = float(request.form['mileage'])
        enginecc = float(request.form['enginecc'])
        seat = float(request.form['seat'])

        car_data = np.array([year,km,fuel,seller,transmission,owner,mileage,enginecc,seat]).reshape(1,-1)
        price = model.predict(car_data)
        price = price[0]
        price2 = str(np.round(price,0).astype(int))
        return render_template('home.html',price = price2)



   

if __name__ == '__main__':
    app.run(debug=True)
