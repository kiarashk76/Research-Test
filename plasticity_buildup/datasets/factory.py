from .fourier import RandomSupervisedFourierDataset
from .linear import RandomSupervisedLinearDataset
from .nonlinear import RandomSupervisedNonlinearDataset
from .polynomial import RandomSupervisedPolynomialDataset


def make_dataset(dataset_name, config):
    name = dataset_name.lower()
    common = {
        "num_samples": config["num_samples"],
        "input_dim": config["input_dim"],
        "output_dim": config["output_dim"],
    }
    if name == "linear":
        return RandomSupervisedLinearDataset(**common)
    if name == "nonlinear":
        return RandomSupervisedNonlinearDataset(**common, teacher_hidden_dim=32, activation="relu")
    if name == "polynomial":
        return RandomSupervisedPolynomialDataset(**common, include_interactions=True)
    if name == "fourier":
        return RandomSupervisedFourierDataset(**common, num_frequencies=32, frequency_scale=1.0)
    raise ValueError(f"Unknown dataset {dataset_name!r}. Choose from: linear, nonlinear, polynomial, fourier.")
