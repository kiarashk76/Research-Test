from __future__ import annotations

import os
from copy import deepcopy

from agents import *
from environments import *
from llm.client import LLMClient


# ============================================================================
# LLM Configurations
# ============================================================================

LLM_CONFIGS = {
    "GEMINI": {
        "temperature": 0.7,
        "timeout": 60,
        "max_retries": 3,
        "stream": True,
    },
    "VULCAN": {
        "temperature": 0.7,
        "timeout": 60,
        "max_retries": 3,
        "stream": True,
    },
}


# ============================================================================
# Environment Configurations
# ============================================================================

ENV_CONFIGS = {
    "SimpleGridEnv": {
        "constructor": SimpleGridEnv,
        "params": {
            "max_steps": 50,
            "size": 5,
        },
        "folder_keys": ["size", "max_steps"],
    },

    "ObstacleGridEnv": {
        "constructor": ObstacleGridEnv,
        "params": {
            "max_steps": 50,
            "size": 5,
            "obstacle_density": 0.2,
        },
        "folder_keys": ["size", "max_steps", "obstacle_density"],
    },
}


# ============================================================================
# Agent Configurations
# ============================================================================

AGENT_CONFIGS = {
    "DQNAgent": {
        "constructor": DQNAgent,
        "uses_llm": False,
        "params": {
            "learning_rate": 1e-3,
            "discount": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 5_000,
            "replay_capacity": 10_000,
            "batch_size": 64,
            "num_batches": 1,
            "learning_starts": 500,
            "train_frequency": 4,
            "target_update_frequency": 250,
            "device": "cpu",
            "verbose": False,
        },
        "folder_keys": ["learning_rate", "batch_size"],
    },

    "SimpleLLMAgent": {
        "constructor": SimpleLLMAgent,
        "uses_llm": True,
        "params": {
            "n_actions": 1,
            "verbose": False,
            "device": "cpu",
        },
        "folder_keys": ["n_actions"],
    },

    "ProgrammaticLLMAgent": {
        "constructor": ProgrammaticLLMAgent,
        "uses_llm": True,
        "params": {
            "n_actions": 10,
            "verbose": False,
            "device": "cpu",
        },
        "folder_keys": ["n_actions"],
    },
}


# ============================================================================
# Factory Functions
# ============================================================================

def make_llm_client(
    llm_name: str = "GEMINI",
    overrides: dict | None = None,
) -> LLMClient:
    if llm_name not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM configuration: {llm_name}")

    params = deepcopy(LLM_CONFIGS[llm_name])

    if overrides:
        params.update(overrides)

    return LLMClient(
        model=os.environ[f"{llm_name}_MODEL"],
        api_key=os.environ[f"{llm_name}_API_KEY"],
        base_url=os.environ[f"{llm_name}_BASE_URL"],
        **params,
    )


def make_env(
    env_name: str,
    overrides: dict | None = None,
):
    """Build an environment. Returns ``(env, resolved_params)``.

    ``resolved_params`` is the config's defaults merged with ``overrides``
    and is JSON-serializable, so callers can use it to name/document runs.
    """
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")

    config = ENV_CONFIGS[env_name]
    params = deepcopy(config["params"])

    if overrides:
        params.update(overrides)

    env = config["constructor"](**params)

    return env, params


def make_agent(
    env,
    agent_name: str,
    llm_name: str = "GEMINI",
    overrides: dict | None = None,
    llm_overrides: dict | None = None,
):
    """Build an agent. Returns ``(agent, resolved_params)``.

    ``resolved_params`` is the config's defaults merged with ``overrides``
    (plus ``llm_name``/``llm_overrides`` for LLM-backed agents) and is
    JSON-serializable, so callers can use it to name/document runs. It
    excludes the constructed ``client``, which isn't serializable.
    """
    if agent_name not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent: {agent_name}")

    config = AGENT_CONFIGS[agent_name]
    params = deepcopy(config["params"])

    if overrides:
        params.update(overrides)

    constructor_kwargs = dict(params)

    if config["uses_llm"]:
        constructor_kwargs["client"] = make_llm_client(
            llm_name,
            overrides=llm_overrides,
        )
        params["llm_name"] = llm_name
        if llm_overrides:
            params["llm_overrides"] = llm_overrides

    agent = config["constructor"](
        env.observation_space,
        env.action_space,
        **constructor_kwargs,
    )

    return agent, params