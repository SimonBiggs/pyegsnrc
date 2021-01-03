from typing import Dict
from typing_extensions import Literal

import jax.numpy as jnp

ParticleKeys = Literal["position", "direction", "energy"]
T = Dict[ParticleKeys, jnp.DeviceArray]


def zeros(num_particles: int) -> T:
    return {
        "position": jnp.zeros((3, num_particles)),
        "direction": jnp.zeros((3, num_particles)),
        "energy": jnp.zeros((1, num_particles)),
    }
