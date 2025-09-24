# experiment_structure.py
from enum import IntEnum
from types import SimpleNamespace
from typing import Optional

import asyncio
import jax
import jax.numpy as jnp
import numpy as np

from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl.stages import Stage, FeedbackStage, LLMEnvStage
from nicewebrl.logging import get_logger

import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.experimental.img_obs import RGBImgObservationWrapper
from xminigrid.rendering.text_render import _text_encode_rule, _encode_tile

from datetime import datetime, timezone

import nicewebrl.stages as _stages_mod

_stages_mod.timezone = timezone

import config

logger = get_logger(__name__)

# =========================
# Experiment constants
# =========================
MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


# =========================
# Minigrid env wiring
# =========================
class PlaygroundTimestepWrapper(TimestepWrapper):
  def reset(self, key: jax.random.PRNGKey, params=None):
    timestep = self._env.reset(key=key, params=params)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)

  def step(self, key, state, action, params=None):
    if isinstance(state, TimeStep):
      state = state.state
    timestep = self._env.step(params=params, timestep=state, action=action)
    resized_obs = jax.image.resize(
      timestep.observation, shape=(256, 256, 3), method="bilinear"
    ).astype(jnp.uint8)
    return timestep.replace(observation=resized_obs)


class Actions(IntEnum):
  forward = 0
  right = 1
  left = 2
  pick_up = 3
  put_down = 4
  toggle = 5


actions = jnp.array([0, 1, 2, 3, 4, 5])
# NOTE: no ArrowDown on purpose (env has 6 actions mapped below)
action_keys = ["ArrowUp", "ArrowRight", "ArrowLeft", "p", "d", "t"]
action_to_name = ["Forward", "Right", "Left", "Pick Up", "Drop", "Toggle"]


def text_encode_goal(goal: list[int]) -> str:
  goal_id = goal[0]
  if goal_id == 1:
    return f"Agent_Hold({_encode_tile(goal[1:3])})"
  elif goal_id == 3:
    return f"Agent_Near({_encode_tile(goal[1:3])})"
  elif goal_id == 4:
    return f"Tile_Near({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id in range(7, 11):
    dir_map = ["Up", "Right", "Down", "Left"]
    return f"Tile_Near_{dir_map[goal_id - 7]}_Goal({_encode_tile(goal[1:3])}, {_encode_tile(goal[3:5])})"
  elif goal_id in range(11, 15):
    dir_map = ["Up", "Right", "Down", "Left"]
    return f"Agent_Near_{dir_map[goal_id - 11]}_Goal({_encode_tile(goal[1:3])})"
  else:
    raise RuntimeError(f"Unknown goal id: {goal_id}")


def describe_ruleset(ruleset) -> str:
  s = "GOAL:\n" + text_encode_goal(ruleset.goal.tolist()) + "\n\nRULES:\n"
  for rule in ruleset.rules.tolist():
    if rule[0] != 0:
      s += _text_encode_rule(rule) + "\n"
  s += "\nINIT TILES:\n"
  for tile in ruleset.init_tiles.tolist():
    if tile[0] != 0:
      s += _encode_tile(tile) + "\n"
  return s


def get_object_name(object_type: int, object_color: int):
  object_types = {
    0: "EMPTY",
    1: "FLOOR",
    2: "WALL",
    3: "BALL",
    4: "SQUARE",
    5: "PYRAMID",
    6: "GOAL",
    7: "KEY",
    8: "DOOR_LOCKED",
    9: "DOOR_CLOSED",
    10: "DOOR_OPEN",
    11: "HEX",
    12: "STAR",
  }
  colors = {
    0: "EMPTY",
    1: "red",
    2: "green",
    3: "blue",
    4: "purple",
    5: "yellow",
    6: "grey",
    7: "black",
    8: "orange",
    9: "white",
    10: "brown",
    11: "pink",
  }
  if object_type == 0:
    return "EMPTY"
  return f"{colors[object_color]} {object_types[object_type]}"


def convert_state_to_text(timestep: TimeStep, rule_text: str):
  gamestate = timestep.state
  agent_position = gamestate.agent.position.tolist()
  dir_map = ["UP", "RIGHT", "DOWN", "LEFT"]
  agent_direction = dir_map[gamestate.agent.direction]
  cur_reward = timestep.reward
  cur_grid = gamestate.grid.tolist()
  state_text = f"Agent Position: {agent_position}, Agent Direction: {agent_direction}\n"
  state_text += f"Current Reward: {cur_reward}\n"
  state_text += (
    f"\nThe current rule description is:\n{rule_text}\n"
    '"GOAL" describes the goal of the current episode. "RULES" describe how objects can be transformed together this episode. '
    '"INIT TILES" describe the initial objects in the environment.\n'
    "Note that goals are described with relative positions (e.g. to the right of). Make sure to note this.\n"
    "We now give the current grid state:\n"
  )
  for i, row in enumerate(cur_grid):
    for j, (obj_type, color) in enumerate(row):
      state_text += f"({i}, {j}) -> {get_object_name(obj_type, color).lower()}\n"
  return state_text + "\n"


def create_env_with_ruleset(ruleset_key):
  env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
  benchmark = xminigrid.load_benchmark(name="trivial-1m")
  rule = benchmark.sample_ruleset(jax.random.key(ruleset_key))
  rule_text = describe_ruleset(rule)
  env_params = env_params.replace(ruleset=rule, max_steps=50, view_size=11)
  env = GymAutoResetWrapper(env)
  env = RGBImgObservationWrapper(env)
  return env, benchmark, env_params, rule_text


num_envs = 3
env, benchmark, env_params, rule_text = create_env_with_ruleset(0)
jax_env = PlaygroundTimestepWrapper(env, autoreset=True, use_params=True)
jax_web_env = JaxWebEnv(env=jax_env, actions=actions)
jax_web_env.precompile(dummy_env_params=env_params)
render_fn = jax.jit(lambda t: t.observation.astype(jnp.uint8))
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, env_params)


async def call_llm_api(prompt: str, env_text: str, stage: Stage) -> str:
  # default to Gemini
  model_name = app.storage.user.get("selected_model", "gemini")

  # Helper to record & return
  async def _finish(text: str) -> str:
    await stage.handle_llm_prompt_submission(prompt, text)
    return text

  try:
    # ----- GEMINI -----
    if model_name == "gemini":
      try:
        import google.generativeai as genai

        genai.configure(api_key=config.GEMINI_API_KEY)
        gem_model = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash")

        def _gen():
          model = genai.GenerativeModel(gem_model)
          return model.generate_content([env_text, "\n\nUser question:\n", prompt])

        result = await asyncio.to_thread(_gen)
        text = getattr(result, "text", None) or "(no text)"
        return await _finish(text)
      except Exception as e:
        logger.error(f"[GEMINI Error] {e}")
        # fall back to ChatGPT if available
        model_name = "chatgpt"

    # ----- CHATGPT (OpenAI) -----
    if model_name == "chatgpt":
      try:
        from openai import OpenAI

        client = OpenAI(api_key=config.CHATGPT_API_KEY)
        openai_model = getattr(config, "OPENAI_MODEL", "gpt-4o-mini")

        def _chat():
          return client.chat.completions.create(
            model=openai_model,
            messages=[
              {
                "role": "system",
                "content": "You give concise, step-by-step hints for a gridworld.",
              },
              {"role": "user", "content": f"{env_text}\n\nUser question:\n{prompt}"},
            ],
          )

        resp = await asyncio.to_thread(_chat)
        text = resp.choices[0].message.content
        return await _finish(text)
      except Exception as e:
        logger.error(f"[OPENAI Error] {e}")
        return "The assistant is temporarily unavailable."

    # ----- CLAUDE (Anthropic) -----
    if model_name == "claude":
      try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        claude_model = getattr(config, "CLAUDE_MODEL", "claude-3-5-sonnet-latest")

        def _claude():
          return client.messages.create(
            model=claude_model,
            max_tokens=400,
            messages=[
              {"role": "user", "content": f"{env_text}\n\nUser question:\n{prompt}"}
            ],
          )

        msg = await asyncio.to_thread(_claude)
        parts = getattr(msg, "content", []) or []
        text = "".join(getattr(p, "text", "") for p in parts)
        return await _finish(text or "(empty Claude response)")
      except Exception as e:
        logger.error(f"[CLAUDE Error] {e}")
        return "The assistant is temporarily unavailable."

    return f"Unsupported model: {model_name}"
  except Exception as e:
    logger.error(f"[LLM Error] {e}")
    return "The assistant is temporarily unavailable."


# =========================
# Stage UI
# =========================
def make_image_html(src):
  return f"""
    <div id="stateImageContainer" style="display:flex;justify-content:center;align-items:center;">
        <img id="stateImage" src="{src}" style="width:400px;height:400px;object-fit:contain;">
    </div>"""


async def env_stage_display_fn(stage: Stage, container: ui.element, timestep: TimeStep):
  rendered_img = stage.render_fn(timestep)
  new_obs_base64 = base64_npimage(rendered_img)
  stage_state = stage.get_user_data("stage_state")
  if stage_state is None:
    await stage.set_user_data(stage_state=SimpleNamespace(nsuccesses=0, nepisodes=0))
    stage_state = stage.get_user_data("stage_state")

  # New ruleset text each (re)render
  rule = benchmark.sample_ruleset(nicewebrl.new_rng())
  current_rule_text = describe_ruleset(rule)

  # Default focus: ENV is active
  await stage.set_user_data(active_container="env", rule_text=current_rule_text)

  nicewebrl.clear_element(container)
  with container.style("align-items: center;"):
    # Allow focus & mark env active when the container gets focus/clicked
    container.props("tabindex=0")
    container.on(
      "focusin",
      lambda e: asyncio.create_task(stage.set_user_data(active_container="env")),
    )
    container.on(
      "mousedown",
      lambda e: asyncio.create_task(stage.set_user_data(active_container="env")),
    )

    # Header stats
    with ui.row():
      ui.label(f"Success: {stage_state.nsuccesses}/{stage.min_success}")
      ui.label().bind_text_from(
        stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
      )

    # LLM selector (inline, default to gemini)
    model_list = ["gemini", "chatgpt", "claude"]
    with ui.row().classes("items-center gap-3"):
      ui.label("LLM:")
      sel = ui.select(
        model_list,
        value=app.storage.user.get("selected_model", "gemini"),
      ).props("dense outlined")

      def _set_llm(e):
        app.storage.user["selected_model"] = e.value
        ui.notify(f"LLM: {e.value}", type="positive")

      sel.on("update:model-value", _set_llm)

    ui.markdown("Ask the AI for help below.")
    ui.html(make_image_html(src=new_obs_base64))

    # Prompt box + focus guards
    prompt_box = ui.textarea(label="Ask the AI:").classes("w-full")
    await stage.set_user_data(prompt_box_id=prompt_box.id)
    prompt_box.on(
      "focus",
      lambda e: asyncio.create_task(stage.set_user_data(active_container="llm")),
    )
    prompt_box.on(
      "blur", lambda e: asyncio.create_task(stage.set_user_data(active_container="env"))
    )

    # Response label
    llm_response_output = ui.label().classes("text-sm italic text-gray-600")

    async def submit_prompt(_=None):
      prompt_text = (prompt_box.value or "").strip()
      if not prompt_text:
        return
      env_text = convert_state_to_text(timestep, current_rule_text)
      text = await call_llm_api(prompt_text, env_text, stage)
      llm_response_output.text = f"AI: {text}"

    ui.button("Submit", on_click=submit_prompt).classes("mt-2")


# =========================
# Success check
# =========================
def evaluate_success_fn(timestep: TimeStep, params: Optional[object] = None):
  return timestep.last() and timestep.reward > 0


# =========================
# Build stages
# =========================
env_stages = [
  LLMEnvStage(
    name=f"Environment {i + 1}",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    msg_display_time=2,
    metadata=dict(stage_number=i + 1),
  )
  for i in range(num_envs)
]


async def instruction_display_fn(stage, container):
  nicewebrl.clear_element(container)
  with container.style("align-items: center;"):
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      "Press the arrow keys to move the agent.\n\n"
      "Press:\n- p to pick up an object\n- d to drop an object\n- t to transform an object"
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)


async def feedback_display_fn(stage, container):
  nicewebrl.clear_element(container)
  with container.style("align-items: center;"):
    ui.markdown(f"## {stage.name}")
    ui.markdown("Please answer the following:")
    questions = ["How helpful was the AI?", "How human-like was the AI?"]
    answers, done = {}, asyncio.Event()

    def record_answer(q, a):
      answers[q] = a
      if all(x is not None for x in answers.values()):
        done.set()

    for q in questions:
      ui.label(q)
      ui.radio(
        [1, 2, 3, 4, 5], on_change=lambda e, q=q: record_answer(q, e.value)
      ).props("inline")
      answers[q] = None
    await done.wait()
    return answers


feedback_stage = FeedbackStage(name="Feedback", display_fn=feedback_display_fn)

all_stages = [instruction_stage, *env_stages, feedback_stage]
experiment = nicewebrl.SimpleExperiment(
  stages=all_stages,
  randomize=[False] + [True] * (len(all_stages) - 1),
)
