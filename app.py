from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
import os

from ml_model import evaluate_on_user_sequences
from ml_model import predict_next_input  # Make sure to import the function correctly

from ml_model import DEFAULT_INPUT_LENGTH

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_15input.db' #let's try only 10
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

secret_key = os.urandom(24) #secret key for the server side session
app.config['SECRET_KEY'] = secret_key

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    sequence = db.Column(db.String(120), nullable=True)
    input_date = db.Column(db.DateTime, default=datetime.utcnow)  # Stores the current time and date
    symbols_used = db.Column(db.String(120), nullable=True)  # Stores the symbols used

with app.app_context():
    db.create_all()
    # push context manually to app
    # guest_user = User.query.filter_by(username='guest').first()
    # if not guest_user:
    #     guest_user = User(username='guest', sequence='')
    #     db.session.add(guest_user)
    #     db.session.commit()


@app.route('/')
def index():
    # Query all users
    unique_usernames = User.query.with_entities(User.username).distinct().all()
    users = [user[0] for user in unique_usernames]  # Extract usernames from tuples
    print(users)

    # Generate initial predictions
    initial_predictions = predict_next_input("")  # Empty sequence for initial guess
    # Format predictions for JSON serialization
    formatted_predictions = {
        model: {
            'predicted_class': int(pred['predicted_class']),
            'confidence': float(pred['confidence'])
        } for model, pred in initial_predictions.items()
    }

    return render_template('index.html', users=users, initial_predictions=formatted_predictions)


@app.route('/update_sequence', methods=['POST'])
def update_sequence():
    data = request.json
    username = data.get('username')  # Get the username from the AJAX request
    sequence = data.get('sequence')
    symbols = data.get('symbols')  # Assuming you send this data from the client
    print("sequence is ", sequence)
    print("symbols is", symbols)

    # Find the user by the username and update their sequence
    user = User.query.filter_by(username=username).first()
    if user:
        user.sequence = sequence
        user.input_date = datetime.utcnow()  # Update input date to current time
        # Update the symbols_used with the symbols sent from the client
        user.symbols_used = symbols

        #db.session.commit()
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'User not found'})


@app.route('/handle_user', methods=['POST'])
def handle_user():
    data = request.json
    username = data['username']
    symbols = data.get('symbols')  # Dynamically receive symbols from the request

    print("user is", username)
    print("symbols are", symbols)

    user = User.query.filter_by(username=username).first()
    if not user:
        # If user does not exist, add them with the symbols
        user = User(username=username, symbols_used=symbols)
        db.session.add(user)
        db.session.commit() #think i need to commit?

        print("user added")
    else:
        # Optionally update the symbols for existing users
        user.symbols_used = symbols

    # Store the user's id in the session
    session['user_id'] = user.id
    return jsonify({'status': 'success'})

@app.route('/view_table')
def view_table():
    users = User.query.all()  # Query all users
    return render_template('view_table.html', users=users)

@app.route('/submit_clear', methods=['POST'])
def submit_clear():
    data = request.json
    username = data.get('username', 'guest')  # Default to 'guest' if not provided
    sequence = data.get('sequence', '')
    symbols = data.get('symbols', '')
    print("user name is" , username, type(username))

    #if no user selected, then we use guest
    if username == "":
        print("hi")
        username = "guest" #default user
    new_entry = User(username=username, sequence=sequence, input_date=datetime.utcnow(), symbols_used=symbols)
    db.session.add(new_entry)
    #db.session.commit()

    print("test")
    #ML STUFF HERE
    # Check if sequence is provided
    if sequence:
        print("training")
        # Save the sequence and user info
        new_entry = User(username=username, sequence=sequence, input_date=datetime.utcnow(), symbols_used=symbols)
        db.session.add(new_entry)
        db.session.commit()

        # Call the ML function to train on this sequence and get predictions
        predictions = evaluate_on_user_sequences(sequence)

        # Print results to console
        for model_name, result in predictions.items():
            print(f"{model_name}:")
            print(f"Predicted sequence: {result['predicted_sequence']}")
            print(f"Accuracy: {result['accuracy']}\n")

        # Prepare the predictions for JSON response
        response_data = {
            'status': 'success',
            'message': 'Sequence submitted and predictions are generated.',
            'predictions': predictions
        }
    else:
        print("No sequence provided.")
        response_data = {'status': 'error', 'message': 'No sequence provided.'}

    return jsonify(response_data)


@app.route('/get_ai_predictions', methods=['POST'])
def get_ai_predictions():

    data = request.json
    sequence = data.get('sequence', '')
    # Check if AI Predictions are requested after each input
    if data.get('aiPrediction') == 'each':
        print("BUTTON PRESSED")
        predictions = predict_next_input(sequence)
        # Convert numpy data types to Python native types for JSON serialization
        predictions = {model: {'predicted_class': int(pred['predicted_class']),  # Convert np.int64 to int
                               'confidence': float(pred['confidence'])}  # Convert np.float32 to float
                       for model, pred in predictions.items()}
        print("APP.PY", predictions)
        print(type(predictions))
        return jsonify({'status': 'success', 'predictions': predictions})
    else:
        return jsonify({'status': 'error', 'message': 'AI predictions not requested'})


if __name__ == '__main__':
    app.run(debug=True)



# make input length to be 50 - more reliable estimation
# acid test - actual random number, flip a coin and test
# certain patterns
# ensemble model
# in the middle of input, model is trying one of 3 models, and then switch to that model
#  maybe switch model again, etc
#  behind the scene, it always looks like one model making predictions, dispatch model
# 4. MOST IMPORTANT
    # self testing, then i need to do and ask other people, human testing project
    # ask rachel try to think a sequence of random numbers, as random as possible, but DONT FLIP A COIN
    # use your mind to think randomly, and input 50 0's and 1's, and at the end my model will predict if you are truly random
    # ask friends for this and enter 50 0's and 1's as random as possible
    # at the end, best model if consistently predicts 60% or better, then that is a winning model
    # then human minds, if they can predict randomly, think free will, creatively, randomly, is not purely random
    # human mind is partially deterministic
    # AI MODEL CAN PREDICT HUMANS AT 60% ACCURACY
    # THINK RANDOMLY
# 30 is not as reliable as 50 inputs
# at least on my phone or laptop
# need to test this on a bunch of people
# NEED TO WRECK THE OTHER MASTERS STUDENT
# ABOVE 60%


# 




