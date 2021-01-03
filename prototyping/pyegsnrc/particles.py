from typing import Dict
from typing_extensions import Literal

import jax.numpy as jnp

ParticleKeys = Literal["position", "direction", "energy"]
ParticlesBase = Dict[ParticleKeys, jnp.DeviceArray]


class Particles(ParticlesBase):
    @classmethod
    def zeros(cls, num_particles: int):
        return cls(
            {
                "position": jnp.zeros((3, num_particles)),
                "direction": jnp.zeros((3, num_particles)),
                "energy": jnp.zeros((1, num_particles)),
            }
        )
