import pandas as pd

def load_train_data():
    train = pd.read_csv("data/train.csv")
    print("Train dataset loaded")
    print(train.head())
    return train


def load_test_data():
    test = pd.read_csv("data/test.csv")
    print("Test dataset loaded")
    print(test.head())
    return test


if __name__ == "__main__":
    train = load_train_data()
    test = load_test_data()