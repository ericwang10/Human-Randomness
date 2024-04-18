from tensorflow.keras.layers import Input, LSTM, SimpleRNN, GRU, Dense, Masking, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

DEFAULT_INPUT_LENGTH = 50
LEARNING_RATE = 0.02
# LSTM Model
def init_lstm_model(input_length=30, learning_rate=0.02):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = LSTM(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Simple RNN Model
def init_rnn_model(input_length=30, learning_rate=0.02):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = SimpleRNN(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# GRU Model
def init_gru_model(input_length=30, learning_rate=0.02):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = GRU(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def init_complex_rnn_model(input_length=30, learning_rate=0.02):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = SimpleRNN(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def init_attention_lstm_model(input_length=50, learning_rate=0.02):
    print("attention 1")
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = LSTM(16, return_sequences=True)(x)  # Enable return_sequences for attention
    x = SeqSelfAttention(attention_activation='sigmoid')(x)
    x = LSTM(8, return_sequences=False)(x)  # Another LSTM layer to process the output from the attention layer
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_functions = {
    "SimpleRNN": init_rnn_model,
#    "LSTM": init_lstm_model,
#    "GRU": init_gru_model,
#     "ComplexRNN": init_complex_rnn_model
#    "Attention 1": init_attention_lstm_model
}
from tensorflow.keras.utils import to_categorical
import numpy as np
import random

DEBUG = True


def generate_random_sequence(length=30):
    return [random.choice([0, 1]) for _ in range(length)]


def incremental_train_and_predict(model, sequence, input_length=30):
    predictions = []  # Store model predictions

    # Initial prediction with fully masked input
    X_initial = np.array([[-1] * input_length]).reshape(1, input_length, 1)
    pred_initial = model.predict(X_initial, verbose=0)
    predicted_class_initial = np.argmax(pred_initial, axis=1)[0]
    predictions.append(predicted_class_initial)

    if DEBUG:
        print(f"\nInitial Prediction (fully masked, no input): {predicted_class_initial}")
        print(f"Pred so far {predictions}")

    for i in range(len(sequence)):
        # Skip the last one for prediction since there's no next value to predict
        if i < len(sequence) - 1:
            # Use the sequence up to the current point for prediction
            current_sequence = sequence[:i + 1]
            X, y = update_dataset_categorical(current_sequence, input_length)

            # Train model on the observed sequence so far
            if i > 0:  # Ensure there's data to train on
                model.fit(X, y, epochs=5, verbose=0) # train for 5 epochs instead of 10... takes too long

            # Predict the next value based on the current sequence
            X_pred = np.array([current_sequence + [-1] * (input_length - len(current_sequence))]).reshape(1,
                                                                                                          input_length,
                                                                                                          1)
            pred = model.predict(X_pred, verbose=0)
            predicted_class = np.argmax(pred, axis=1)[0]
            predictions.append(predicted_class)

            if DEBUG:
                print(f"\nTraining Data up to index {i}: {current_sequence}")
                print(f"Predicting next input based on: {X_pred.flatten()}")
                print(f"Prediction at index {i + 1} (for next input): {predicted_class}")
                print(f"Pred so far {predictions}")

    return predictions


def update_dataset_categorical(sequence, input_length=30):
    X = []
    y = []
    for i in range(0, len(sequence)):
        # Pad sequence to match the input_length
        padded_sequence = sequence[:i] + [-1] * (input_length - i)
        X.append(padded_sequence)
        y.append(sequence[i])
    X = np.array(X).reshape(-1, input_length, 1)
    y = to_categorical(y, num_classes=2)
    return X, y


def calculate_accuracy(actual, predicted):
    correct = sum(a == p for a, p in zip(actual, predicted))
    return correct / len(actual)


def train_and_evaluate_model(init_model_func, sequence, input_length=30):
    model = init_model_func(input_length=input_length, learning_rate=LEARNING_RATE)
    predictions = incremental_train_and_predict(model, sequence, input_length)
    accuracy = calculate_accuracy(sequence, predictions)
    return accuracy, predictions


def evaluate_on_user_sequences(sequence):
    sequence = [int(ch) for ch in sequence]  # Convert string to list of integers
    input_length = len(sequence)  # Set input length to the current sequence length
    print(f"\nTesting user sequence: {sequence}")

    results = {}  # Initialize results dictionary

    for name, init_model_func in model_functions.items():
        accuracy, predictions = train_and_evaluate_model(init_model_func, sequence, input_length)

        # Convert predicted sequence to string format for easy reading
        predicted_sequence_str = ''.join(map(str, predictions))

        print(f"{name} model predictions: {predicted_sequence_str}")
        print(f"{name} model accuracy on current sequence: {accuracy:.4f}")

        # Save results
        results[name] = {
            'predicted_sequence': predicted_sequence_str,
            'accuracy': f"{accuracy:.4f}%"
        }

    return results


def predict_next_input(sequence_str):

    # Convert string sequence to list of integers
    sequence = [int(i) for i in sequence_str]
    input_length = DEFAULT_INPUT_LENGTH  # Account for padding

    # Dictionary to hold predictions and confidence levels for each model
    predictions_confidence = {}

    for name, init_model_func in model_functions.items():
        print("PREDICTING NEXT SEQUENCE", name, sequence_str)
        # Initialize and train model
        model = init_model_func(input_length=input_length, learning_rate=LEARNING_RATE)

        # Prepare the sequence for prediction
        if sequence:  # Check if sequence is not empty
            X, y = update_dataset_categorical(sequence, input_length)
            print("X IS")
            for x in X:
                print(x.flatten())
            for pair in y:
                print(pair)
            model.fit(X, y, epochs=10, verbose=0)
            #model.fit(X, to_categorical(sequence, num_classes=2), epochs=10, verbose=0) #hmm?

        # Prepare the input for prediction: pad the sequence to input_length
        X_pred = np.array([sequence + [-1] * (input_length - len(sequence))]).reshape(1, input_length, 1)

        #DEBUG
        print("MAKING A PREDICTION FOR")
        for x in X_pred:
            print(x.flatten())

        # Make prediction
        pred = model.predict(X_pred, verbose=0)
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred, axis=1)[0]

        #DEBUG
        print("PRED IS", pred, "predicted class is", predicted_class)
        print("\n\n")


        # Save the prediction and confidence
        predictions_confidence[name] = {'predicted_class': predicted_class, 'confidence': confidence}

    # DEBUG
    # print("Predictions and confidence levels before returning:")
    # for model_name, info in predictions_confidence.items():
    #     print(f"{model_name}: Predicted class={info['predicted_class']}, Confidence={info['confidence']:.4f}")

    return predictions_confidence
