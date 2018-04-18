def load_subset_context(data_path='../output/', activities_to_keep=None):

    from sklearn.preprocessing import MinMaxScaler
    from sklearn import preprocessing
    import pandas as pd
    import numpy as np

    labels = pd.read_csv(data_path + '/labels', sep=",", header=None)

    labels[labels[0] == "Sleep"] = "Home"
    labels[labels[0] == "Launch Break"] = "Break"
    labels[labels[0] == "Restaurant"] = "Free time"
    labels[labels[0] == "Shopping"] = "Free time"

    x = pd.read_csv(data_path + '/data', sep=",")

    x.head()

    x["Label"] = labels.values

    if activities_to_keep is not None:
        x = x.loc[x["Label"].isin(activities_to_keep)]

    labels = x["Label"].values
    x = x.drop("Label", axis=1)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)

    x = MinMaxScaler().fit_transform(x)
    x = x.astype(np.float32)

    return x, y, labels


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    import numpy as np

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)

    return x, y, y
