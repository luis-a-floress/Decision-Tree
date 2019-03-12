import data_cleaner as dc
import file_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


def random_forest(x_train, y_train):
    # Creacion del Arbol de desicion CART
    dtc = RandomForestClassifier(n_estimators=50)

    # Entrenamiento
    dtc.fit(x_train, y_train)

    return dtc


def prediction(x_test, forest):
    prediction = forest.predict(x_test)
    return prediction


def cal_accuracy(x_train, y_train, forest):
    print("Random Forest:")
    # Accuracy
    accuracy = round(forest.score(x_train, y_train) * 100, 2)
    print("Accuracy:", round(accuracy, 2, ), "%", end = "\n"*2)

    # Validacion Cruzada
    cross_predictions = cross_val_score(forest, x_train, y_train, cv=5)
    print("Cross Validation:\n", cross_predictions, end = "\n"*2)

    # Matrix de confusion
    cross_predictions = cross_val_predict(forest, x_train, y_train, cv=5)
    print("Confusion Matrix:\n", confusion_matrix(y_train, cross_predictions), end = "\n"*2)


    # Precision and Recall
    print("Precision:", precision_score(y_train, cross_predictions))
    print("Recall:",recall_score(y_train, cross_predictions), end = "\n"*2)


def main():
    dataset_file = "./../data/titanic-dataset.csv"
    file_output = "./../data/random_forest.csv"

    titanic_data = file_manager.readData(dataset_file)
    titanic_data = dc.dataCleaner(titanic_data)
    
    data_test, x_train, x_test, y_train = dc.splitData(titanic_data)

    rfc = random_forest(x_train, y_train)

    rfc_predictions = prediction(x_test, rfc) 
    cal_accuracy(x_train, y_train, rfc) 
    file_manager.writeData(file_output, data_test["PassengerId"], rfc_predictions)
    print("NOTE: To see the predicted values look for random_forest.csv in the data folder.")


if __name__ == '__main__':
    main()
