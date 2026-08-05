from dataclasses import dataclass


@dataclass
class MethodState:
    model: object
    optimizer: object
    training_method: object
    results: dict
