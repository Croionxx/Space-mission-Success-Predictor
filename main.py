import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def preprocess_data(df):
    """
    Preprocess the data, including encoding categorical variables and extracting features.
    """
    # Extract year from Datum
    df['Year'] = df['Datum'].apply(lambda x: int(re.search(r'\d{4}', x).group()) if pd.notna(x) else 0)

    # Create binary features from Detail
    df['is_military'] = df['Detail'].str.contains('military|Military|defense|Defense', na=False).astype(int)
    df['is_communication'] = df['Detail'].str.contains('communication|Communication|sat|Sat', na=False).astype(int)
    df['is_research'] = df['Detail'].str.contains('research|Research|test|Test', na=False).astype(int)

    # Extract numeric values from Rocket column, handle missing column
    if ' Rocket' in df.columns:
        df['Rocket_Value'] = pd.to_numeric(df[' Rocket'], errors='coerce')
    else:
        df['Rocket_Value'] = 0

    # Handle categorical variables
    encoders = {}
    categorical_cols = ['Company Name', 'Location', 'Status Rocket']

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    # Select features for model
    feature_cols = ['Year', 'is_military', 'is_communication', 'is_research', 
                    'Rocket_Value'] + [col + '_encoded' for col in categorical_cols]

    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(0)

    return df[feature_cols], encoders

def build_model(input_size):
    """
    Build a TensorFlow model for the rocket mission classification task.
    """
    model = Sequential([
        Dense(128, input_dim=input_size, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the TensorFlow model with early stopping.
    """
    input_size = X_train.shape[1]
    model = build_model(input_size)

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history

def save_model(model, encoders, scaler, filepath):
    """
    Save the TensorFlow model along with encoders and scaler.
    """
    model.save(filepath + '.h5')
    np.save(filepath + '_encoders.npy', encoders)
    np.save(filepath + '_scaler.npy', scaler)

def calculate_f1_score(model, X_val, y_val):
    """
    Calculate and return the F1 score for the validation dataset.
    """
    val_predictions = (model.predict(X_val) >= 0.5).astype(int)
    f1 = f1_score(y_val, val_predictions)
    print(f"Validation F1 Score: {f1:.4f}")
    return f1

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('Datasets/train.csv')
    test_df = pd.read_csv('Datasets/test.csv')

    # Preprocess data
    X_train_raw, encoders = preprocess_data(train_df)
    X_test_raw, _ = preprocess_data(test_df)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, 
        train_df['Status Mission'].values,
        test_size=0.2,
        random_state=42,
        stratify=train_df['Status Mission'].values
    )

    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)

    # Calculate F1 score
    f1 = calculate_f1_score(model, X_val, y_val)

    # Save model
    save_model(model, encoders, scaler, 'rocket_mission_model')

    # Evaluate model
    test_predictions = (model.predict(X_test_scaled) >= 0.5).astype(int)
    print("Test Predictions:", test_predictions)

    # Save predictions with Serial Number
    test_df['Status Mission'] = test_predictions
    output_df = test_df[['Serial Number', 'Status Mission']]
    output_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")
