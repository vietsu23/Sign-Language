import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt  # Import thư viện vẽ biểu đồ

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Training model")

    # Add and parse the arguments
    parser.add_argument("--model_name", help="Name of the model",
                        type=str, default="model")
    parser.add_argument("--dir", help="Location of the model",
                        type=str, default="models")
    args = parser.parse_args()

    # Train X, y and mapping
    X, y, mapping = [], [], dict()

    # Read in the data from data folder
    for current_class_index, pose_file in enumerate(os.scandir("data")):
        # Load pose data
        file_path = f"data/{pose_file.name}"
        pose_data = np.load(file_path)

        # Add to training data
        X.append(pose_data)
        y += [current_class_index] * pose_data.shape[0]

        # Add to mapping
        mapping[current_class_index] = pose_file.name.split(".")[0]

    # Convert to Numpy
    X, y = np.vstack(X), np.array(y)

    # Create model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Train the model and validate
    model = SVC(decision_function_shape='ovo', kernel='rbf')
    model.fit(X_train, y_train)

    # Get the train and test accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Display the train and test accuracy
    print(f"Training examples: {X.shape[0]}. Num classes: {len(mapping)}")
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}% - Test accuracy: {round(test_accuracy * 100, 2)}%")

    # Save the model to the model's folder
    model_path = os.path.join(f"{args.dir}", f"{args.model_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump((model, mapping), file)
    print(f"Saved model to {model_path}")

    # **Vẽ biểu đồ Train Accuracy vs Test Accuracy**
    labels = ["Train Accuracy", "Test Accuracy"]
    values = [train_accuracy * 100, test_accuracy * 100]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['blue', 'green'])
    plt.ylim(0, 100)  # Giới hạn trục y từ 0% - 100%
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Testing Accuracy")
    plt.text(0, values[0] + 1, f"{values[0]:.2f}%", ha='center', fontsize=12)
    plt.text(1, values[1] + 1, f"{values[1]:.2f}%", ha='center', fontsize=12)

    # Hiển thị biểu đồ
    plt.show()
