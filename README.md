# 🤸 Human Activity Recognition (HAR) using Smartphone Sensor Data
![Human Activity Recognition](https://github.com/user-attachments/assets/c768e545-6db5-4338-84be-12ce252f5622)

This project uses machine learning to recognize human activities like walking, sitting, or laying using smartphone sensor data. It includes model training, prediction on new sensor data, and a user-friendly Streamlit web app.

## 📊 Dataset

- **Source:** [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Activities:**  
  - 1: WALKING  
  - 2: WALKING_UPSTAIRS  
  - 3: WALKING_DOWNSTAIRS  
  - 4: SITTING  
  - 5: STANDING  
  - 6: LAYING  
- **Features:** Accelerometer and gyroscope readings preprocessed into 561 features per record.



## 🚀 How to Run

## 1. 🧠 Train the Model
      python train_model.py
      #Trains a RandomForest model
      #Saves har_model.pkl containing model and feature names

## 2. 🖥️ Run the Streamlit App
      streamlit run app.py
      #Upload a CSV file (sample_input.csv) with correct sensor features

## 3. 🧪 Sample Input Format
      X_test.head(5).to_csv("sample_input.csv", index=False)
      
## 📄 Creating Test CSV File

- Use the UCI HAR Dataset's `X_test.txt` file to generate `sample_input.csv`
- Make sure feature names match the trained model (561 columns, cleaned for duplicates)

