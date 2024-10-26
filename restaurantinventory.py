import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
day=input("Enter day: ")
food=input("Enter food: ")
df = pd.read_csv('restaurant.csv')
label_encoder_day = LabelEncoder()
label_encoder_food = LabelEncoder()

df['day_encoded'] = label_encoder_day.fit_transform(df['day'])
df['food_encoded'] = label_encoder_food.fit_transform(df['food'])
X = df[['day_encoded', 'food_encoded']]
y = df['amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
model = RandomForestRegressor(n_estimators=800, random_state=400)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error on test set: {mse:.2f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")

def predict_food_amount(day, food):
    day_encoded = label_encoder_day.transform([day])[0]
    food_encoded = label_encoder_food.transform([food])[0]
    predicted_amount = model.predict([[day_encoded, food_encoded]])
    return predicted_amount[0]
    
day_to_predict = day.capitalize() 
food_to_predict = food.capitalize() 
predicted_amount = int(predict_food_amount(day_to_predict, food_to_predict))
print(f"Predicted amount of {food_to_predict} for {day_to_predict}: {predicted_amount:.2f}")
