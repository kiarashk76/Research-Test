import gymnasium as gym


class BaseEnvironment(gym.Env):
    """Shared name for this project's Gymnasium environments.

    Gymnasium already provides the environment interface. Concrete
    environments implement its normal ``reset``, ``step``, and ``render``
    methods and define their own spaces.
    """

    pass
