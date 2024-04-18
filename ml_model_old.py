import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, GRU, Dense, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Initialize models
def init_lstm_model(input_length=30, learning_rate=0.01):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = LSTM(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def init_rnn_model(input_length=30, learning_rate=0.01):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = SimpleRNN(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def init_gru_model(input_length=30, learning_rate=0.01):
    inputs = Input(shape=(input_length, 1))
    x = Masking(mask_value=-1.)(inputs)
    x = GRU(16, activation='relu', return_sequences=False)(x)
    x = Dense(8, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def update_dataset_categorical(sequence, input_length=30):
    X = []
    y = []
    for i in range(0, len(sequence)): #change from 1 to 0 because this would let the model predict the first input!
        X.append(sequence[:i] + [-1] * (input_length - i))  # Pad sequence
        y.append(sequence[i])
    X = np.array(X).reshape(-1, input_length, 1)
    y = to_categorical(y, num_classes=2)
    return X, y

def train_and_evaluate_model(init_model_func, sequence, input_length=30):
    model = init_model_func(input_length=input_length, learning_rate=0.02)  # Reinitialize the model
    X, y = update_dataset_categorical(sequence, input_length)
    model.fit(X, y, epochs=10, verbose=0)
    accuracy = model.evaluate(X, y, verbose=0)[1]  # Get accuracy
    return accuracy


def train_and_predict_for_each_model(input_sequence):
    # Convert input sequence to the correct format
    sequence = [int(char) for char in input_sequence]

    # Initialize the dictionary to hold results
    results = {}

    # Preprocess the input for all models
    X, y = update_dataset_categorical(sequence)

    # Iterate over each model function, train, predict, and calculate accuracy
    for name, init_model_func in model_functions.items():
        # Initialize model
        model = init_model_func()

        # Train the model
        model.fit(X, y, epochs=10, verbose=0)

        # Evaluate the model
        accuracy = model.evaluate(X, y, verbose=0)[1] * 100  # Convert to percentage

        # Make predictions
        predictions = model.predict(X)
        predicted_sequence = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

        # Convert predicted sequence to string format for easy reading
        predicted_sequence_str = ''.join(str(i) for i in predicted_sequence)

        # Save results
        results[name] = {
            'predicted_sequence': predicted_sequence_str,
            'accuracy': f"{accuracy:.2f}%"
        }

    return results

model_functions = {
    "LSTM": init_lstm_model,
    "SimpleRNN": init_rnn_model,
    "GRU": init_gru_model,
}
