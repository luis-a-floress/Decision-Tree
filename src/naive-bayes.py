import data_cleaner as dc
import file_manager
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


def naive_bayes(x_train, y_train):
    # Inicializacion de Navie Bayes Classifier
    gnb = GaussianNB()

    # Entrenamiento
    gnb.fit(x_train, y_train)

    return gnb


def prediction(x_test, bayes):
    prediction = bayes.predict(x_test)
    return prediction


def cal_accuracy(x_train, y_train, bayes):
    print("Naive Bayes:")
    # Accuracy
    accuracy = round(bayes.score(x_train, y_train) * 100, 2)
    print("Accuracy:", round(accuracy, 2, ), "%", end = "\n"*2)

    # Validacion Cruzada
    cross_predictions = cross_val_score(bayes, x_train, y_train, cv=5)
    print("Cross Validation:\n", cross_predictions, end = "\n"*2)

    # Matrix de confusion
    cross_predictions = cross_val_predict(bayes, x_train, y_train, cv=5)
    print("Confusion Matrix:\n", confusion_matrix(y_train, cross_predictions), end = "\n"*2)


    # Precision and Recall
    print("Precision:", precision_score(y_train, cross_predictions))
    print("Recall:",recall_score(y_train, cross_predictions), end = "\n"*2)


def main():
    dataset_file = "./../data/titanic-dataset.csv"
    file_output = "./../data/naive_bayes.csv"

    titanic_data = file_manager.readData(dataset_file)
    titanic_data = dc.dataCleaner(titanic_data)
    
    data_test, x_train, x_test, y_train = dc.splitData(titanic_data)

    gnb = naive_bayes(x_train, y_train)

    gnb_predictions = prediction(x_test, gnb) 
    cal_accuracy(x_train, y_train, gnb) 
    file_manager.writeData(file_output, data_test["PassengerId"], gnb_predictions)
    print("NOTE: To see the predicted values look for naive_bayes.csv in the data folder.")


if __name__ == '__main__':
    main()
