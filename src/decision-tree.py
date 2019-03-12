import data_cleaner as dc
import file_manager
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def decision_tree_using_gini(x_train, y_train):
    # Creacion del Arbol de desicion CART
    dtc = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)

    # Entrenamiento
    dtc.fit(x_train, y_train)

    return dtc


def prediction(x_test, tree):
    prediction = tree.predict(x_test)
    return prediction


def cal_accuracy(x_train, y_train, tree):
    print("Decision Tree:")
    # Accuracy
    accuracy = round(tree.score(x_train, y_train) * 100, 2)
    print("Accuracy:", round(accuracy, 2, ), "%", end = "\n"*2)

    # Validacion Cruzada
    cross_predictions = cross_val_score(tree, x_train, y_train, cv=5)
    print("Cross Validation:\n", cross_predictions, end = "\n"*2)

    # Matrix de confusion
    cross_predictions = cross_val_predict(tree, x_train, y_train, cv=5)
    print("Confusion Matrix:\n", confusion_matrix(y_train, cross_predictions), end = "\n"*2)

    # Precision and Recall
    print("Precision:", precision_score(y_train, cross_predictions))
    print("Recall:",recall_score(y_train, cross_predictions), end = "\n"*2)


def main():
    dataset_file = "./../data/titanic-dataset.csv"
    file_output = "./../data/decision_tree.csv"

    titanic_data = file_manager.readData(dataset_file)
    titanic_data = dc.dataCleaner(titanic_data)
    
    data_test, x_train, x_test, y_train = dc.splitData(titanic_data)

    dtc = decision_tree_using_gini(x_train, y_train)

    dtc_predictions = prediction(x_test, dtc) 
    cal_accuracy(x_train, y_train, dtc) 
    file_manager.writeData(file_output, data_test["PassengerId"], dtc_predictions)
    print("NOTE: To see the predicted values look for decision_tree.csv in the data folder.")


if __name__ == '__main__':
    main()
