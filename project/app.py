from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import or_
from flask import send_from_directory


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

#defining booking database model with its columns
class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    room_name = db.Column(db.String(100), nullable=False) 
    price = db.Column(db.Float, nullable=False)
    amount = db.Column(db.Integer, nullable=False)

    #calculating of the duration and returns it in days
    @property
    def duration(self):
        return (self.end_date - self.start_date).days
    
    def calculate_price(self):
        # Define price based on room name
        if 'Penthouse Suite' in self.room_name:
            self.price = 50000
        elif 'Suite' in self.room_name:
            self.price = 40000
        elif 'Deluxe Room' in self.room_name:
            self.price = 30000
        else:
            # Default price if room name doesn't match any condition
            self.price = 20000  
        
         # Calculate amount
        self.amount = self.duration * self.price

    #initializing new instance of booking by seting initial price and amount
    def __init__(self, **kwargs):
        super(Booking, self).__init__(**kwargs)
        self.calculate_price()

#creates all the necessary fields of the table database created above
with app.app_context():
    db.create_all()

#link or route for home page
@app.route('/')
def index():
    return render_template('index.html')

#link or route for booking page
@app.route('/booking')
def booking():
    # Assuming you want to display all rooms for booking
    return render_template('booking.html')

#defining a new booking record in the database
@app.route('/book', methods=['POST'])
def book():
    if request.method == 'POST':
        name = request.form['name']
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']
        room_name = request.form['room_name']  

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Check room availability for selected dates
        availability_response = check_availability(room_name, start_date, end_date)
        if not availability_response['available']:
            return jsonify({'success': False, 'message': 'Room not available for selected dates'})
        
        # Calculate price based on room name
        if 'Penthouse Suite' in room_name:
            price = 50000
        elif 'Suite' in room_name:
            price = 40000
        elif 'Deluxe Room' in room_name:
            price = 30000
        else:
            # Default price if room name doesn't match any condition
            price = 20000 

        #adding a new record of booking to the database
        new_booking = Booking(name=name, start_date=start_date, end_date=end_date, room_name=room_name)
        db.session.add(new_booking)
        db.session.commit()

         # Retrieve the ID of the newly created booking
        booking_id = new_booking.id

        home_url = url_for('index', _external=True)

        # Define the popup message
        popup_message = f"Booking successful! Your booking ID is: {booking_id}. "

        # Return JSON response with success status, booking ID, and redirect URL
        return jsonify({'success': True, 'booking_id': booking_id, 'popup_message': popup_message, 'Click the link': 'http://127.0.0.1:5000'})


#link or route for a page that displays the bookings made 
@app.route('/bookings')
def display_bookings():
    bookings = Booking.query.all()
    return render_template('bookings.html', bookings=bookings)




#check if a room is available to be handed to a client
def check_availability(room_name, start_date, end_date):
    # Check existing bookings for the selected room type and dates
    conflicting_bookings = Booking.query.filter_by(room_name=room_name).filter(
        (Booking.start_date <= end_date) & (Booking.end_date >= start_date)
    ).all()

    # Check if there are any conflicting bookings
    if conflicting_bookings:
        return {'available': False}
    else:
        return {'available': True}

#creating a predictive model and training it to provide a forecast of room availability  
def train_model():
    # Provided data to work with
    data = {
        'Name': ['Gabriel', 'Collins', 'Jane', 'John', 'Doe', 'Chris', 'Rehema', 'Rebecca'],
        'Room Name': ['Suite 1', 'Penthouse Suite 1', 'Suite 1', 'Deluxe Room 1', 'Penthouse Suite 1', 'Deluxe Room 2', 'Suite 1', 'Deluxe Room 1'],
        'Start Date': ['2023-12-03', '2023-12-23', '2023-12-14', '2024-01-01', '2024-01-01', '2024-02-13', '2023-12-30', '2024-02-12'],
        'End Date': ['2023-12-09', '2023-12-27', '2023-12-20', '2024-01-06', '2024-01-05', '2024-02-16', '2024-01-05', '2024-02-15'],
        'Price': [40000.0, 50000.0, 40000.0, 30000.0, 50000.0, 30000.0, 40000.0, 30000.0]
    }
    
    # Convert dates to datetime objects
    data['Start Date'] = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in data['Start Date']]
    data['End Date'] = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in data['End Date']]

    # Calculate duration from above datetime objects
    data['Duration (Days)'] = [(end_date - start_date).days for start_date, end_date in zip(data['Start Date'], data['End Date'])]

    # Calculate amount by multiplying price and duration obtained above
    data['Amount'] = [price * duration for price, duration in zip(data['Price'], data['Duration (Days)'])]

    # Create a DataFrame from the dictionary for data manipulation
    df = pd.DataFrame(data)

    # Initialize LabelEncoder for conversion of string to integer
    label_encoder = LabelEncoder()

    # Encode the 'Room Name' column i.e converting the names of the rooms to integer(binary)
    df['Room Name'] = label_encoder.fit_transform(df['Room Name'])

    # Split the data into features (X) and target variable (y) for training and testing
    X = df[['Room Name', 'Price', 'Amount']] # Features
    y = df['Duration (Days)']  # Target variable

    # Split the data into training and testing sets from above features and target variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model for checking accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    return model

# Call the train_model function to train the model
trained_model = train_model()
    
#link or route for the occupancy forecast 
@app.route('/occupancy-forecast')
def occupancy_forecast_page():
    return render_template('occupancy_forecast.html')  

# Getting occupancy forecast by the client using predictive model
@app.route('/occupancy_forecast', methods=['POST'])
def occupancy_forecast():
    # Parse request data
    data = request.get_json() # Data sent by client checking for rooms

    # Extract relevant information from the request
    room_type = data.get('room_type')
    check_in_date = datetime.strptime(data.get('check_in_date'), '%Y-%m-%d').date()
    check_out_date = datetime.strptime(data.get('check_out_date'), '%Y-%m-%d').date()

    # Calculate duration
    duration = (check_out_date - check_in_date).days

    # Calculate price based on room type
    if 'Penthouse Suite' in room_type:
        price = 50000
    elif 'Suite' in room_type:
        price = 40000
    elif 'Deluxe Room' in room_type:
        price = 30000
    else:
        price = 20000

    # Calculate amount
    amount = price * duration

    # Create a dataframe with the input data which will be used as features by the predictive model
    input_data = pd.DataFrame({
        'Room Name': [1],
        'Price' : [price],
        'Amount' : [amount]
    })

     # Query the database to check for conflicting bookings thus preventing the predictive model ...
     # ...from predicting an already booked room 
    conflicting_bookings = Booking.query.filter_by(room_name=room_type)\
                                         .filter(or_(Booking.start_date <= check_out_date, Booking.end_date >= check_in_date))\
                                         .all()

    # Determine if the room can be available based on conflicting bookings
    if conflicting_bookings:
        return jsonify({'occupancy_forecast': 'Room not available for selected dates'})
    else:
        # Make a prediction using the trained model if there are no conflicting bookings
        prediction = trained_model.predict(input_data)

        # Determine if the room can be available based on prediction gotten above
        if prediction == 0:
            return jsonify({'occupancy_forecast': 'Room not available for booking'})
        else:
            return jsonify({'occupancy_forecast': 'Room available for booking'})
        
# Define a route to serve static files such as images
@app.route('/Images/<path:filename>')
def serve_static(filename):
    return send_from_directory('Images', filename)

#Define route for room page
@app.route('/rooms')
def rooms():
    return render_template('rooms.html')

#Define route for about page
@app.route('/about')
def about():
    return render_template('about.html')

#Define route for contact us page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Debug and run the app
if __name__ == '__main__':
    app.run(debug=True)
