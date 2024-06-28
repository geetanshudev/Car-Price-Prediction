# import importnat modules
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
#loading datasets
car = pd.read_csv('data sets MEGA\cardekho.csv',usecols=['year','selling_price','km_driven','fuel','seller_type','transmission','owner','mileage(km/ltr/kg)','engine','seats'])
print(car.columns)

#now we see all value_counts of individual columns
'''
for i in range(0,11):
    print(car.iloc[:,i:i+1].value_counts())
    print("\n--------------------------------------------\n")
'''
#processing null values 

imp = SimpleImputer(missing_values=np.nan,strategy='mean')
car['mileage(km/ltr/kg)'] = imp.fit_transform(car[['mileage(km/ltr/kg)']])
car['engine'] = imp.fit_transform(car[['engine']])
car['seats'] = imp.fit_transform(car[['seats']])

#preprocessing starts here
one1 = OrdinalEncoder(categories=[["Diesel","Petrol","CNG","LPG"]]) #0,1,2,3
car["fuel"] = one1.fit_transform(car["fuel"].values.reshape(-1, 1))

one2 = OrdinalEncoder(categories=[["Individual","Dealer","Trustmark Dealer"]]) #0,1,2
car["seller_type"] = one2.fit_transform(car["seller_type"].values.reshape(-1,1))

one3 = OrdinalEncoder(categories=[["Manual","Automatic"]]) #0,1
car["transmission"] = one3.fit_transform(car["transmission"].values.reshape(-1,1))

one4 = OrdinalEncoder(categories=[["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"]]) #0,1,2,3,4
car["owner"] = one4.fit_transform(car["owner"].values.reshape(-1,1))

#print(car.sample(5))

x = car[['year','km_driven','fuel','seller_type','transmission','owner','mileage(km/ltr/kg)','engine','seats']]
y = car['selling_price']

#train test split data
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)

#making ensemble model and training testing data 

rgr = LinearRegression()
rgr1 = DecisionTreeRegressor()
rgr2 = RandomForestRegressor()

estimators = [('lr',rgr),('dtr',rgr1),('rfr',rgr2)]

'''
#testing different model 
for est in estimators:
  x = cross_val_score(est[1],xtrain,ytrain,cv=20,scoring='r2')
  print(est[0],np.round(np.mean(x),3))
'''

vc = VotingRegressor(estimators=estimators)
x = cross_val_score(vc,xtrain,ytrain,cv=20,scoring='r2')
print(np.round(np.mean(x),3))

vc.fit(xtrain,ytrain)
y_pred = vc.predict(xtest)
print("R2 score = ",r2_score(ytest,y_pred))

#testing on sample data
test_data = xtest.sample()
print(test_data.columns)
test_data = list(test_data.values)
print("testing data list = ",test_data)
pred_data = vc.predict(test_data)
print("predicted Price = ")
print(np.round(pred_data[0],3))

#saving ML Model in joblib
import joblib
file = "car_price_model.joblib"
joblib.dump(vc,file)