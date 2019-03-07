import pandas

def readData(file: str): 
    data = pandas.read_csv(file, sep= ',') 
    return data 

def writeData(file: str, passengerId, survived):
    data = pandas.DataFrame({
            "PassengerId": passengerId,
            "Survived": survived
        })
    data.to_csv(file, index=False)