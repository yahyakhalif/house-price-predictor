from flask import Flask, session, render_template, request, redirect, url_for
import pyrebase
from annmodel import ANNRegressor

app = Flask(__name__)


config = {
    'apiKey': "AIzaSyBbEzViRP   j6eSsUyECQQC6uKey7gI9M5EQ",
  'authDomain': "housepricepredictor-bd154.firebaseapp.com",
  'databaseURL': "https://housepricepredictor-bd154-default-rtdb.firebaseio.com/",
  'projectId': "housepricepredictor-bd154",
  'storageBucket': "housepricepredictor-bd154.appspot.com",
  'messagingSenderId': "817729852503",
  'appId': "1:817729852503:web:0abd362b5f59352f59ff12",
  'measurementId': "G-3M06EJDHJ0",
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()  # Reference to your database


app.secret_key = 'secret'

# Load the saved model
model_path = 'models/annmodel.pkl'  # Replace with the path to your saved model
loaded_model = ANNRegressor.load_ann_model('models/annmodel.pkl')

@app.route('/', methods=['POST', 'GET'])
def login():
    if 'user' in session:
        return render_template('pred1.html')  # Redirect to the route that renders pred1.html
    if request.method=='POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
        
        except:
            return 'Failed to Login'
            
    
    return render_template('auth/login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password == confirm_password:
            try:
                # Create a new user
                user = auth.create_user_with_email_and_password(email, password)
                return redirect(url_for('login'))
            
            except: 
                return 'Failed to create a user'
        

    return render_template('auth/register.html')
    
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    type = request.form.get('type', type=int)
    bedrooms = request.form.get('bedrooms', type=int)
    bathrooms = request.form.get('bathrooms', type=int)
    toilets = request.form.get('toilets', type=float) 
    parking = request.form.get('parking', type=int)
    location = request.form.get('location', type=int)
    furnished = request.form.get('furnished', type=int)
    shared = request.form.get('shared', type=int)
    serviced = request.form.get('serviced', type=int)


    # Prepare input data in the correct order
    input_data = [bedrooms, bathrooms, toilets, furnished, serviced, shared, parking, type, location]
    # Make a prediction
    predicted_value = ANNRegressor.value_predictor(input_data, loaded_model)

    predicted_value = predicted_value + 50000
    single_predicted_value = predicted_value[0][0]
    # Format the value as a string for display
    formatted_prediction = "{:.2f}".format(single_predicted_value)
    data_to_store = {
        'type': type,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'toilets': toilets,
        'parking': parking,
        'location': location,
        'furnished': furnished,
        'shared': shared,
        'serviced': serviced,
        'predicted_price': formatted_prediction
    }

    # Push data to Firebase
    db.child("userpredictions").push(data_to_store)
    
    # Return or render the prediction
    return render_template('result.html', prediction=formatted_prediction)


    

if __name__ == '__main__':
    app.run(debug=True, port = 1111)