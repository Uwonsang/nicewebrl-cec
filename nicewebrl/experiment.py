from nicegui import app
from typing import List, Union, Dict
import dataclasses
import uuid
import jax.numpy as jnp
import jax.random

from nicewebrl.stages import Block, Stage
from nicewebrl.container import Container
from nicewebrl.nicejax import new_rng
from nicewebrl.logging import get_logger
from nicewebrl.utils import get_user_lock, get_progress

logger = get_logger(__name__)



@dataclasses.dataclass
class Experiment(Container):
  blocks: List[Union[Block, Stage]] = dataclasses.field(default_factory=list)
  randomize: Union[bool, List[bool]] = False
  name: str = None

  def __post_init__(self):
    super().__post_init__()
    if self.name is None:
      self.name = f"experiment_{uuid.uuid4().hex[:8]}"

    # Convert any Stage objects to single-member Blocks
    converted_blocks = []
    for item in self.blocks:
      if isinstance(item, Stage):
        # Create a Block with this single stage
        block = Block(
          stages=[item],
          name=item.name if item.name else f"auto_block_{item.unique_id}",
          randomize=False
        )
        converted_blocks.append(block)
      else:
        converted_blocks.append(item)

    self.blocks = converted_blocks

    stage_idx = 0
    for idx, block in enumerate(self.blocks):
      block.unique_id = f"{idx}_{block.unique_id}"
      for stage in block.stages:
        stage.unique_id = f"{stage_idx}_{stage.unique_id}"
        stage_idx += 1

  @property
  def num_stages(self):
    return sum(len(block.stages) for block in self.blocks)

  @property
  def num_blocks(self):
    return len(self.blocks)

  async def initialize(self):
    app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
    app.storage.user["block_idx"] = app.storage.user.get("block_idx", 0)
    app.storage.user["block_name"] = "undefined"
    app.storage.user["stage_name"] = "undefined"
    app.storage.user["num_blocks"] = self.num_blocks
    app.storage.user["num_stages"] = self.num_stages
    block_order = await self.get_block_order()
    block_names_in_order = [self.blocks[i].name for i in block_order]
    logger.info(f"Block order: {block_names_in_order}")

  async def get_blocks(self, ordered: bool = True):
    if ordered:
      order = await self.get_block_order()
      return [self.blocks[i] for i in order]
    else:
      return self.blocks

  async def get_block_order(self):
    if not self.randomize:
      return list(range(len(self.blocks)))
    block_order = self.get_user_data("block_order")
    if block_order is not None:
      return block_order

    import ipdb; ipdb.set_trace()
    indices = jnp.arange(len(self.blocks))
    mask = jnp.array(self.randomize)

    # Get randomizable indices
    random_indices = indices[mask]

    # Permute the randomizable indices
    rng_key = new_rng()
    rng_key, subkey = jax.random.split(rng_key)
    random_indices = jax.random.permutation(subkey, random_indices)

    # Combine back together
    permuted = indices.at[mask].set(random_indices)

    block_order = [int(i) for i in permuted]
    await self.set_user_data(block_order=block_order)
    return block_order

  def get_experiment_stage_idx(self):
    stage_idx = app.storage.user["stage_idx"]
    if stage_idx is None:
      stage_idx = 0
      app.storage.user["stage_idx"] = stage_idx
    return stage_idx

  def get_block_idx(self):
    block_idx = app.storage.user["block_idx"]
    if block_idx is None:
      block_idx = 0
      app.storage.user["block_idx"] = block_idx
    return block_idx

  async def get_block(self):
    """

    First, get the block_idx. if
    """
    # first get the block
    block_idx = self.get_block_idx()
    block_order = await self.get_block_order()
    if block_idx >= len(block_order):
      logger.info("Defaulting to final block")
      block_idx = len(block_order) - 1

    block = self.blocks[block_order[block_idx]]
    app.storage.user["block_name"] = block.name
    return block

  async def get_stage(self):
    """

    First, get the block_idx. if
    """
    # first get the block
    block: Block = await self.get_block()

    # then get the stage
    stage: Stage = await block.get_stage()
    app.storage.user["stage_name"] = stage.name
    return stage

  async def advance(self):
    self.advance_stage()

    block: Block = await self.get_block()
    if block.finished():
      self.advance_block()

  async def advance_block(self):
    block_idx = self.get_block_idx()
    async with get_user_lock():
      app.storage.user["block_idx"] = block_idx + 1
    get_progress()

  async def advance_stage(self):
    # advance stage within the block
    block = await self.get_block()
    await block.advance_stage()

    # advance experiment stage idx
    stage_idx = self.get_experiment_stage_idx()
    async with get_user_lock():
      app.storage.user["stage_idx"] = stage_idx + 1
    get_progress()

  def not_finished(self):
    block_idx = self.get_block_idx()
    return block_idx < len(self.blocks)

  def finished(self):
    block_idx = self.get_block_idx()
    finished = app.storage.user.get("experiment_finished", False)
    return block_idx >= len(self.blocks) or finished

  def force_finish(self):
    app.storage.user["stage_idx"] = self.num_stages
    app.storage.user["block_idx"] = self.num_blocks


SimpleExperiment = Experiment


@dataclasses.dataclass
class ExperimentSet(Container):
  experiments: Dict[str, Experiment] = dataclasses.field(default_factory=list)

  def set_experiment(self, experiment: str):
    app.storage.user['experiment'] = experiment

  def get_experiment(self):
    name = app.storage.user['experiment']
    return self.experiments[name]

  # enable dict-like access
  def __getitem__(self, key):
    return self.experiments[key]
