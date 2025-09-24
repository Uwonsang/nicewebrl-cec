# experiment_structure.py
from enum import IntEnum
from typing import Optional

import asyncio
import jax
import jax.numpy as jnp
from flax import struct
from nicegui import ui

import navix as nx
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimeStep, TimestepWrapper
from nicewebrl.stages import Stage, LLMEnvStage
from nicewebrl import get_logger

# Optional: only needed if your web_app attaches last two frames to Gemini
try:
  from shared_state import previous_obs_base64, current_obs_base64

  _HAS_SHARED_STATE = True
except Exception:
  previous_obs_base64 = None
  current_obs_base64 = None
  _HAS_SHARED_STATE = False

import config  # must define GEMINI_API_KEY; optional GEMINI_MODEL (default: gemini-2.0-flash)

logger = get_logger(__name__)

# =========================
# Experiment constants
# =========================
MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 10_000
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1


# =========================
# Actions & keys
# =========================
class Actions(IntEnum):
  left = 0
  right = 1
  forward = 2
  unused1 = 3
  unused2 = 4
  unused3 = 5


actions = jnp.array([0, 1, 2])  # left, right, forward
action_keys = ["ArrowLeft", "ArrowRight", "ArrowUp"]
action_to_name = ["Left", "Right", "Forward"]

# =========================
# Minigrid (Navix) env
# =========================
rows = 8
cols = 8
jax_env_raw = nx.make("Navix-Empty-8x8-v0", observation_fn=nx.observations.rgb)


@struct.dataclass
class EnvParams:
  max_steps_in_episode: int = MAX_EPISODE_TIMESTEPS


default_params = EnvParams()


class NavixTimestepWrapper(TimestepWrapper):
  """Wrap Navix timestep -> nicewebrl.TimeStep and resize obs in the env loop."""

  def reset(self, key: jax.random.PRNGKey, params=None):
    t = self._env.reset(key)
    resized = jax.image.resize(
      t.observation, shape=(128, 128, 3), method="bilinear"
    ).astype(jnp.uint8)
    return TimeStep(
      state=t.replace(observation=resized),
      observation=resized,
      discount=jnp.ones((), dtype=jnp.float32),
      reward=jnp.zeros((), dtype=jnp.float32),
      step_type=jnp.array(0, dtype=jnp.uint8),
    )

  def step(self, key, state, action, params=None):
    if isinstance(state, TimeStep):
      state = state.state
    t = self._env.step(state, action)
    resized = jax.image.resize(
      t.observation, shape=(128, 128, 3), method="bilinear"
    ).astype(jnp.uint8)
    return TimeStep(
      state=t.replace(observation=resized),
      observation=resized,
      discount=jnp.ones((), dtype=jnp.float32),
      reward=t.reward,
      step_type=jnp.where(
        t.is_done(), jnp.array(2, dtype=jnp.uint8), jnp.array(1, dtype=jnp.uint8)
      ),
    )


# Wrap + web wrapper
jax_env = NavixTimestepWrapper(jax_env_raw, autoreset=True, use_params=True)
jax_web_env = JaxWebEnv(env=jax_env, actions=actions)

# Precompile env ops
jax_web_env.precompile(dummy_env_params=default_params)


# Render 128->256 only for display
def _render_img(t: nicewebrl.TimeStep):
  return jax.image.resize(t.observation, shape=(256, 256, 3), method="nearest").astype(
    jnp.uint8
  )


render_fn = jax.jit(_render_img)
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, default_params)


# =========================
# Simple text summary for the LLM
# =========================
def summarize_timestep_for_llm(timestep: TimeStep) -> str:
  # Minimal prompt context for the empty 8x8 gridworld.
  return (
    "You are assisting a human in an empty 8x8 gridworld.\n"
    "They can move Left, Right, or Forward. Provide concise navigation hints."
  )


# =========================
# LLM call (Gemini only, OPENAI fallback)
# =========================
async def call_llm_api(prompt: str, env_text: str, stage: Stage) -> str:
  async def _finish(text: str) -> str:
    await stage.handle_llm_prompt_submission(prompt, text)
    return text

  try:
    # --- Gemini primary ---
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    gem_model = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash")

    def _gen():
      model = genai.GenerativeModel(gem_model)
      return model.generate_content([env_text, "\n\nUser question:\n", prompt])

    result = await asyncio.to_thread(_gen)
    text = getattr(result, "text", None) or "(no text from Gemini)"
    return await _finish(text)

  except Exception as e:
    logger.error(f"[GEMINI Error] {e}, falling back to OpenAI...")

    try:
      # --- OpenAI fallback ---
      from openai import AsyncOpenAI

      client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

      resp = await client.chat.completions.create(
        model=getattr(config, "OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
          {"role": "system", "content": env_text},
          {"role": "user", "content": prompt},
        ],
      )
      text = resp.choices[0].message.content.strip()
      return await _finish(text)

    except Exception as oe:
      logger.error(f"[OpenAI Error] {oe}")
      return "The assistant is temporarily unavailable (both Gemini and OpenAI failed)."


# =========================
# Stages
# =========================
all_stages = []


# ---- Instructions ----
async def instruction_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(
      "Use the arrow keys to move:\n\n"
      "- ⬅️ Left\n- ➡️ Right\n- ⬆️ Forward\n\n"
      "Ask the built-in AI (Gemini) for concise navigation hints."
    )


instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)

# ---- Environment (LLM-enabled) ----
env_params = default_params.replace(max_steps_in_episode=MAX_EPISODE_TIMESTEPS)


def make_image_html(src: str) -> str:
  return f"""
    <div id="stateImageContainer" style="display:flex;justify-content:center;align-items:center;">
        <img id="stateImage" src="{src}" style="width:400px;height:400px;object-fit:contain;">
    </div>"""


async def env_stage_display_fn(
  stage: LLMEnvStage, container: ui.element, timestep: TimeStep
):
  """Display the env + Gemini chat UI (xland-style layout) with robust typing guards."""
  # Render the current observation
  rendered_img = stage.render_fn(timestep)
  new_obs_base64 = base64_npimage(rendered_img)

  # If you have shared_state.py, keep last two frames for the side-panel chat
  if _HAS_SHARED_STATE:
    global previous_obs_base64, current_obs_base64
    previous_obs_base64 = current_obs_base64
    current_obs_base64 = new_obs_base64

  # Ensure we have a stage_state to bind labels against (same shape as xland)
  stage_state = stage.get_user_data("stage_state")
  if stage_state is None:
    from types import SimpleNamespace

    await stage.set_user_data(stage_state=SimpleNamespace(nsuccesses=0, nepisodes=0))
    stage_state = stage.get_user_data("stage_state")

  # Default: ENV is active so arrow keys work unless a text box is focused
  await stage.set_user_data(active_container="env")

  # Build UI: focusable container, stats, image, prompt box, submit
  nicewebrl.clear_element(container)
  with container.style("align-items: center;"):
    # Make the container focusable, and mark ENV active when focused/clicked
    container.props("tabindex=0")
    container.on(
      "focusin",
      lambda e: asyncio.create_task(stage.set_user_data(active_container="env")),
    )
    container.on(
      "mousedown",
      lambda e: asyncio.create_task(stage.set_user_data(active_container="env")),
    )

    # Header stats (bind like xland)
    with ui.row():
      ui.label(f"Success: {stage_state.nsuccesses}/{stage.min_success}")
      ui.label().bind_text_from(
        stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
      )

    # Fixed LLM (Gemini-only example)
    ui.label("LLM: Gemini (gemini-2.0-flash)").classes("text-sm text-gray-600")

    ui.markdown("Ask the AI for help below.")
    ui.html(make_image_html(src=new_obs_base64))

    # Prompt box + focus guards (so keypresses don't leak while typing)
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
      user_text = (prompt_box.value or "").strip()
      if not user_text:
        return
      env_text = summarize_timestep_for_llm(timestep)
      answer = await call_llm_api(user_text, env_text, stage)
      llm_response_output.text = f"AI: {answer}"

    ui.button("Submit", on_click=submit_prompt).classes("mt-2")


def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
  return timestep.last() and (timestep.reward > 0)


environment_stage = LLMEnvStage(
  name="Environment",
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
  metadata=dict(
    desc="Navix Empty 8x8 with Gemini LLM hints",
    grid_size=f"{rows}x{cols}",
  ),
)

all_stages.append(environment_stage)
