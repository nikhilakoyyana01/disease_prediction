import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Reading the modified dataset
diseases = pd.read_csv("C:\\Users\\KAUSIK\\Desktop\\datasets\\modified_dataset.csv")

# Assuming the target variable is in the 'target_column' column
target_column = 'Disease'  # Replace with the actual target column name

# Split the dataset into features (X) and target variable (y)
X = diseases.drop(target_column, axis=1)
y = diseases[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification n report
#print("Classification Report:\n", classification_report(y_test, y_pred))

with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('decision_tree_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
new_data = pd.DataFrame({
        'Symptom_1': [7],
        'Symptom_2': [5],
        'Symptom_3': [4],
        'Symptom_4': [4],
        'Symptom_5': [0],
        'Symptom_6': [0],
        'Symptom_7': [0],
        'Symptom_8': [0],
        'Symptom_9': [0],
        'Symptom_10': [0],
        'Symptom_11': [0],
        'Symptom_12': [0],
        'Symptom_13': [0],
        'Symptom_14': [0],
        'Symptom_15': [0],
        'Symptom_16': [0],
        'Symptom_17': [0]
    })
    
    # Use the loaded model to make predictions
result = loaded_model.predict(new_data)

    # Display the predicted labels
print("Predicted Labels:", result)