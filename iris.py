

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle


class IrisClassifier:
    def __init__(self):
        """Initialize the classifier and load the dataset"""
        self.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
        self.df = pd.read_csv('iris.data', names=self.columns)
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values
        self.model = SVC()

    def visualize_data(self):
        """Generate pairplot for the dataset"""
        st.subheader("Data Visualization")
        fig = sns.pairplot(self.df, hue='Class_labels')
        st.pyplot(fig)

    def train_model(self):
        """Train the model and display accuracy"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)

        # Save the model
        with open('iris_model.pkl', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self):
        """Load the trained model"""
        with open('iris_model.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, input_features):
        """Make predictions using the trained model"""
        prediction = self.model.predict([input_features])
        return prediction[0]


# Streamlit App
def main():
    st.title("🌸 Iris Classification App")

    classifier = IrisClassifier()

    # Sidebar options
    menu = ["Home", "Visualize Data", "Train Model", "Make Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("Welcome to the Iris Classification App! 🌼")
        st.write("Use this app to explore the Iris dataset, train a model, and make predictions.")

    elif choice == "Visualize Data":
        classifier.visualize_data()

    elif choice == "Train Model":
        classifier.train_model()

    elif choice == "Make Prediction":
        st.subheader("Make a Prediction")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)

        input_features = [sepal_length, sepal_width, petal_length, petal_width]

        if st.button("Predict"):
            classifier.load_model()
            result = classifier.predict(input_features)
            st.success(f"The predicted iris species is: **{result}**")

if __name__ == "__main__":
    main()
