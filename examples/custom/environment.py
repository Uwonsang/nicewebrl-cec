from typing import Optional
from flax import struct
from enum import IntEnum
import jax.random
import jax.numpy as jnp

from nicewebrl import Timestep


# these are parameters that are used to define the environment but don't change
@struct.dataclass
class StaticEnvParams:
  grid_size: int


# these are parameters that change across reset
# e.g. you might have a train parameter that is False during evaluation
@struct.dataclass
class EnvParams:
  train: bool


class Actions(IntEnum):
  right = 0
  down = 1
  left = 2
  up = 3


class Environment:
  def __init__(self, static_params: Optional[StaticEnvParams] = None):
    static_params = static_params or StaticEnvParams()
    self.static_params = static_params

  def reset(self, rng: jax.random.PRNGKey, params: Optional[struct.PyTreeNode] = None):
    """Environment reset logic"""
    return Timestep()

  def step(
    self,
    rng: jax.random.PRNGKey,
    prior_timestep: Timestep,
    action: int,
    params: Optional[struct.PyTreeNode] = None,
  ):
    """Environment dynamics logic."""
    return Timestep()

  @property
  def num_actions(self) -> int:
    """Number of actions possible in environment."""
    return 4


def render_fn(timestep: Timestep):
  return jnp.array


###### Usage
static_params = StaticEnvParams()
train_params = EnvParams(train=True)
eval_params = EnvParams(train=False)
rng = jax.random.PRNGKey(42)

env = Environment(static_params)
timestep = env.reset(rng, train_params)
next_timestep = env.step(rng, timestep, 3, train_params)


env = Environment(static_params)
timestep = env.reset(rng, eval_params)
next_timestep = env.step(rng, timestep, 3, eval_params)
