import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Dataset configuration
FOLDER_PATH = "Dataset_Different_Sector"
STOCK_FILES = [
    "ACLBSL_2000-01-01_2021-12-31.csv",
    "ADBL_2000-01-01_2021-12-31.csv",
    "AHPC_2000-01-01_2021-12-31.csv",
    "NABIL_2000-01-01_2021-12-31.csv",
    "UPPER_2000-01-01_2021-12-31.csv"
]

# ==========================================================
# TASK 2: DATA COLLECTION & PREPROCESSING
# ==========================================================
def preprocess_data(file_path):
    """
    Handles data cleaning, chronological sorting, 
    and advanced feature engineering (MA7, MA21).
    """
    df = pd.read_csv(file_path)
    
    # 1. Cleaning & Sorting
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates(subset=['Date'])
    
    # 2. Feature Selection: Engineering Technical Indicators
    df['MA7'] = df['Close Price'].rolling(window=7).mean()
    df['MA21'] = df['Close Price'].rolling(window=21).mean()
    df['Returns'] = df['Close Price'].pct_change().replace([np.inf, -np.inf], 0)
    
    # 3. Defining Target: 1 for Price UP next day, 0 for DOWN/STABLE
    df['Target'] = (df['Close Price'].shift(-1) > df['Close Price']).astype(int)
    
    # Remove NaN rows created by rolling windows
    df = df.dropna()
    return df

# ==========================================================
# TASK 3 & 4: MODEL DESIGN, TRAINING & EVALUATION
# ==========================================================
def train_and_evaluate(df, stock_name):
    """
    Task 3: Algorithm - Random Forest with Class Balancing.
    Task 4: Evaluation using Accuracy, Precision, Recall, and F1-score.
    """
    # Features used for classification
    features = ['Total Transactions', 'Total Traded Shares', 'Close Price', 'MA7', 'MA21', 'Returns']
    X = df[features]
    y = df['Target']
    
    # Temporal splitting (No shuffling) to simulate real-world time sequence
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Model initialization with 'balanced' weights to fix "Always Upward" bias
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42) #n_estimator change and check 
    model.fit(X_train, y_train)
    
    # Generate predictions for evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, acc, report, X_test, y_test

# ==========================================================
# SEPARATE VISUALIZATION FUNCTION (Optional/As Needed)
# ==========================================================
def generate_visual_evidence(model, X_test, y_test, stock_name):
    """
    Creates Confusion Matrix and Feature Importance plots for Task 4 evaluation.
    Call this function only when you need to generate images for the report.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # 1. Confusion Matrix Window
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {stock_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{stock_name}_confusion_matrix.png")
    plt.show()

    # 2. Feature Importance Window
    importances = model.feature_importances_
    f_names = ['Transactions', 'Shares', 'Price', 'MA7', 'MA21', 'Returns']
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=f_names)
    plt.title(f'Feature Importance: {stock_name}')
    plt.savefig(f"{stock_name}_feature_importance.png")
    plt.show()

# ==========================================================
# TASK 5: INTERFACE DEVELOPMENT (CLI)
# ==========================================================
def start_cli(trained_models):
    """Interactive CLI allowing users to predict future trends (Task 5b)."""
    print("\n" + "="*45)
    print("NEPSE AI STOCK PREDICTION INTERFACE")
    print("="*45)
    
    choice = input(f"Select stock ({', '.join(trained_models.keys())}): ").upper()
    
    if choice in trained_models:
        # Fetch latest data point for prediction
        file_name = [f for f in STOCK_FILES if f.startswith(choice)][0]
        path = os.path.join(FOLDER_PATH, file_name) if os.path.exists(FOLDER_PATH) else file_name
        data = preprocess_data(path)
        latest_row = data.iloc[-1:] # The most recent trading day
        
        # Calculate prediction and confidence
        features_list = ['Total Transactions', 'Total Traded Shares', 'Close Price', 'MA7', 'MA21', 'Returns']
        proba = trained_models[choice].predict_proba(latest_row[features_list])[0]
        prediction = trained_models[choice].predict(latest_row[features_list])[0]
        
        result = "UPWARD" if prediction == 1 else "DOWNWARD/STABLE"
        confidence = proba[prediction] * 100
        
        print(f"\n[AI PREDICTION RESULT]")
        print(f"Stock: {choice} | Confidence: {confidence:.2f}%")
        print(f"Predicted Trend for Next Day: {result}")
        print("-" * 45)
    else:
        print("Invalid Selection.")

# ==========================================================
# MAIN PROJECT EXECUTION
# ==========================================================
if __name__ == "__main__":
    trained_models = {}
    print("--- SYSTEM INITIALIZING: TRAINING MULTI-SECTOR MODELS ---")

    for file_name in STOCK_FILES:
        path = os.path.join(FOLDER_PATH, file_name) if os.path.exists(FOLDER_PATH) else file_name
        
        if os.path.exists(path):
            stock_name = file_name.split('_')[0]
            data = preprocess_data(path)
            
            # Train and Evaluate
            model, acc, report, X_test, y_test = train_and_evaluate(data, stock_name)
            trained_models[stock_name] = model
            
            # OUTPUT ALL METRICS (Precision, Recall, F1)
            print(f"\n>> {stock_name} Classification Report:")
            print(f"Overall Accuracy: {acc:.2f}")
            print(report)
            
            #VISUAL
            # generate_visual_evidence(model, X_test, y_test, stock_name) 
        else:
            print(f"Error: {file_name} not found.")

    # Launch CLI
    start_cli(trained_models)