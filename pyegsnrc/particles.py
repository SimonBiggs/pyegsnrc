from typing import Dict

import jax.numpy as jnp
from typing_extensions import Literal

ParticleKeys = Literal["position", "direction", "energy"]
Type = Dict[ParticleKeys, jnp.DeviceArray]


def zeros(num_particles: int) -> Type:
    return {
        "position": jnp.zeros((3, num_particles)),
        "direction": jnp.zeros((3, num_particles)),
        "energy": jnp.zeros((1, num_particles)),
    }
