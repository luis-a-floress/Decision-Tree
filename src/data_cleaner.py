"""
    El codigo implementado en este archivo para la limpieza de datos esta basado
    en la implementacion de Niklas Donges, en 
    https://www.kaggle.com/niklasdonges/end-to-end-project-with-python?scriptVersionId=10621161

"""

import re


def nameCleaner(data):
    """ Esta implementacion fue modificada """
    honoraries = {"Capt": 1, "Col": 2, "Countess": 3, "Don": 4, "Dona": 5,
        "Dr": 6, "Jonkheer": 7, "Lady": 8, "Major": 9, "Master": 10,
        "Miss": 11, "Mlle": 12, "Mr": 13, "Mrs": 14, "Ms": 15, "Mme": 16,
        "Rev": 17, "Sir": 18}

    data['Name'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Name'] = data['Name'].map(honoraries)

    return data


def sexCleaner(data):
    sex = {"female": 0, "male": 1}
    data['Sex'] = data['Sex'].map(sex)

    return data


def ageCleaner(data):
    """ Esta implementacion fue modificada """
    # Llenamos los valores nulos con la media
    mean = data["Age"].mean()
    data['Age'] = data['Age'].fillna(mean)

    # Intervalos dados por pandas.qcut() con 8 quantiles para categorizar
    # print(pandas.qcut(titanic_data["Age"], 9, duplicates="drop"))
    # [(-0.001, 16.0] < (16.0, 21.0] < (21.0, 25.0] < (25.0, 29.0] <
    # (29.0, 31.0] < (31.0, 36.0] < (36.0, 46.0] < (46.0, 80.0]]
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 21), 'Age'] = 1
    data.loc[(data['Age'] > 21) & (data['Age'] <= 25), 'Age'] = 2
    data.loc[(data['Age'] > 25) & (data['Age'] <= 29), 'Age'] = 3
    data.loc[(data['Age'] > 29) & (data['Age'] <= 31), 'Age'] = 4
    data.loc[(data['Age'] > 31) & (data['Age'] <= 36), 'Age'] = 5
    data.loc[(data['Age'] > 36) & (data['Age'] <= 46), 'Age'] = 6
    data.loc[ data['Age'] > 46, 'Age'] = 7

    data['Age'] = data['Age'].astype(int)

    return data


def fareCleaner(data):
    """ Esta implementacion fue modificada """
    # Llenamos los valores nulos con la media
    mean = data["Fare"].mean()
    data['Fare'] = data['Fare'].fillna(mean)

    # Intervalos dados por pandas.qcut() con 8 quantiles para categorizar
    # print(pandas.qcut(titanic_data["Fare"], 9, duplicates="drop"))
    # [(-0.001, 7.0] < (7.0, 8.0] < (8.0, 13.0] < (13.0, 16.0] <
    # (16.0, 26.0] < (26.0, 35.0] < (35.0, 73.0] < (73.0, 512.0]]
    data.loc[ data['Fare'] <= 7.0, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.0) & (data['Fare'] <= 8.0), 'Fare'] = 1
    data.loc[(data['Fare'] > 8.0) & (data['Fare'] <= 13.0), 'Fare']   = 2
    data.loc[(data['Fare'] > 13.0) & (data['Fare'] <=  16.0), 'Fare']   = 3
    data.loc[(data['Fare'] >  16.0) & (data['Fare'] <= 26.0), 'Fare']   = 4
    data.loc[(data['Fare'] > 26.0) & (data['Fare'] <= 35.0), 'Fare']   = 5
    data.loc[(data['Fare'] > 35.0) & (data['Fare'] <= 73.0), 'Fare']   = 6
    data.loc[ data['Fare'] > 73.0, 'Fare'] = 7

    data['Fare'] = data['Fare'].astype(int)

    return data


def ticketCleaner(data):
    data = data.drop(['Ticket'], axis=1)

    return data


def cabinCleaner(data):
    """ Esta implementacion fue modificada """
    id_cabins = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Z": 9}

    data['Cabin'] = data['Cabin'].fillna("Z0")
    data['Cabin'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Cabin'] = data['Cabin'].map(id_cabins)
    data['Cabin'] = data ['Cabin'].fillna(0)
    data['Cabin'] = data['Cabin'].astype(int)
    
    return data


def embarkedCleaner(data):
    ports = {"C": 0, "Q": 1, "S": 2}

    mode = data["Embarked"].mode(dropna=True)[0]
    data['Embarked'] = data['Embarked'].fillna(mode)
    data['Embarked'] = data['Embarked'].map(ports)

    return data


def dataCleaner(data):
    data = nameCleaner(data)
    data = sexCleaner(data)
    data = ageCleaner(data)
    data = fareCleaner(data)
    data = ticketCleaner(data)
    data = cabinCleaner(data)
    data = embarkedCleaner(data)

    return data


def splitData(data): 
    data_train = data[:891]
    """ Esta implementacion fue modificada """
    data_test = data[891:].drop(["Survived"], axis = 1)
    data_train = data_train.drop(["PassengerId"], axis = 1)
    x_train = data_train.drop("Survived", axis=1)
    y_train = data_train["Survived"]
    x_test  = data_test.drop("PassengerId", axis=1).copy()

    return data_test, x_train, x_test, y_train
