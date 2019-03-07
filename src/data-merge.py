def readFile(file: str):
    csv = []
    with open(file) as document:
        for line in document:
            csv.append(line.strip().split(','))
    return csv

def writeFile(file: str, data: [[str]]):
    with open(file, 'w') as document:
        for item in data:
            document.write("%s\n" % ",".join(item))

def mergeData(test_set: [[str]], submission_set: [[str]]):
    for i in range(len(submission_set)):
        test_set[i].insert(1, submission_set[i][1])
    return test_set

def main():
    
    path = "./../data/"
    train_file = "train.csv"
    test_file = "test.csv"
    submission_file = "gender_submission.csv"
    merge_file = "titanic-dataset.csv"

    train_set = readFile(path + train_file)
    test_set = readFile(path + test_file)[1:]
    submission_set = readFile(path + submission_file)[1:]
    test_set = mergeData(test_set, submission_set)
    print(test_set)
    writeFile(path + merge_file, train_set + test_set)

if __name__ == '__main__':
    main()