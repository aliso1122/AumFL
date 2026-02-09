import medmnist


def get_datasets(train_or_test: str = "train"):
    # datasets = {
    # "PathMNIST": medmnist.PathMNIST(split=train_or_test, download=True),
    # "ChestMNIST": medmnist.ChestMNIST(split=train_or_test, download=True),
    # "DermaMNIST": medmnist.DermaMNIST(split=train_or_test, download=True),
    # "OCTMNIST": medmnist.OCTMNIST(split=train_or_test, download=True),
    # "PneumoniaMNIST": medmnist.PneumoniaMNIST(split=train_or_test, download=True),
    # "RetinaMNIST": medmnist.RetinaMNIST(split=train_or_test, download=True),
    # "BreastMNIST": medmnist.BreastMNIST(split=train_or_test, download=True),
    # "BloodMNIST": medmnist.BloodMNIST(split=train_or_test, download=True),
    # "TissueMNIST": medmnist.TissueMNIST(split=train_or_test, download=True),
    # "OrganAMNIST": medmnist.OrganAMNIST(split=train_or_test, download=True),
    # }
    a = medmnist.PathMNIST(split=train_or_test, download=True)

    a.save("data/MEDMNIST")
    return a
