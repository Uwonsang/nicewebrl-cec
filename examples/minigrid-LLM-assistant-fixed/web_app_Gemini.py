# web_app.py
import os
import asyncio
from asyncio import Lock
from types import SimpleNamespace
from fastapi import Request
from nicegui import app, ui
from tortoise import Tortoise

import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl.stages import EnvStage, LLMEnvStage

import experiment_structure as experiment  # uses all_stages + environment_stage

# ------------------------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------------------------
DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"
setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
logger = get_logger("main")

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)

# Per-user locks
_user_locks = {}


def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


async def experiment_not_finished():
  async with get_user_lock():
    return app.storage.user.get("stage_idx", 0) < len(experiment.all_stages)


# ------------------------------------------------------------------------------
# Key handling (container-scoped, slot-safe)
# ------------------------------------------------------------------------------
from types import SimpleNamespace
from nicegui import app


async def global_handle_key_press(e, container):
  logger.info("global_handle_key_press")

  # current stage
  stage_idx = app.storage.user.get("stage_idx", 0)
  if stage_idx >= len(experiment.all_stages):
    return
  stage = experiment.all_stages[stage_idx]
  if stage.get_user_data("finished", False):
    return

  # ---------- Typing guard (use the event's client; never ui.run_javascript here) ----------
  try:
    is_typing = await e.client.run_javascript("""
          (function(){
            const el = document.activeElement;
            if (!el) return false;
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable) return true;
            const closest = el.closest('input, textarea, [contenteditable=""], [contenteditable="true"], .q-field__native, .q-field__input');
            return !!closest;
          })()
        """)
    if is_typing:
      return
  except Exception as ex:
    logger.warning(f"typing-guard JS failed: {ex}")

  # Only accept keys when the env is the active container
  if stage.get_user_data("active_container") != "env":
    return

  # If the focused element is exactly the prompt box, skip
  try:
    active_id = await e.client.run_javascript(
      'document.activeElement ? document.activeElement.id : ""'
    )
    prompt_box_id = stage.get_user_data("prompt_box_id")
    if prompt_box_id and str(active_id) == str(prompt_box_id):
      return
  except Exception:
    pass

  # ---------- Timestamp stamping (still via event client) ----------
  try:
    image_ts = await e.client.run_javascript(
      "window.imageSeenTime ? new Date(window.imageSeenTime).toISOString() : null"
    )
  except Exception:
    image_ts = None

  try:
    key_ts = await e.client.run_javascript("new Date().toISOString()")
  except Exception:
    key_ts = None

  # Wrap original event args and add timestamps (don’t mutate NiceGUI’s e)
  base_args = dict(getattr(e, "args", {}) or {})
  try:
    if "key" not in base_args and hasattr(e, "args") and hasattr(e.args, "get"):
      base_args["key"] = e.args.get("key")
  except Exception:
    pass
  base_args["imageSeenTime"] = image_ts
  base_args["keydownTime"] = key_ts
  wrapped_event = SimpleNamespace(args=base_args)

  # ---------- Call the stage INSIDE the container slot ----------
  with container:
    await stage.handle_key_press(wrapped_event, container)

  # Optional local hook
  local = stage.get_user_data("local_handle_key_press")
  if local:
    await local()


# ------------------------------------------------------------------------------
# DB init / shutdown
# ------------------------------------------------------------------------------
async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)


# ------------------------------------------------------------------------------
# Stage execution helpers (slot-safe)
# ------------------------------------------------------------------------------
async def run_stage(stage, container):
  """Run one stage while keeping all UI operations inside the container's slot."""
  stage_over_event = asyncio.Event()

  async def local_handle_key_press():
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via key press")
        stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  async def handle_button_press():
    if stage.get_user_data("finished", False):
      return
    await stage.handle_button_press(container)
    async with get_user_lock():
      if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage.name} via button press")
        stage_over_event.set()

  # Keep UI work in the container slot
  with container:
    with container.style("align-items: center;"):
      await stage.activate(container)

    if stage.get_user_data("finished", False):
      logger.info(f"Finished {stage.name} immediately after activation")
      stage_over_event.set()

    if stage.next_button:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)  # still inside slot!
      await handle_button_press()

  await stage_over_event.wait()


async def finish_experiment(container):
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")


# ------------------------------------------------------------------------------
# Experiment flow
# ------------------------------------------------------------------------------
async def start_experiment(container):
  # prevent double-starts
  if app.storage.user.get("experiment_running", False):
    return
  app.storage.user["experiment_running"] = True

  try:
    # Container-scoped key handling; make container focusable and focused
    with container:
      container.props("tabindex=0")
      # bind directly to the container (like xland) so slot context is valid
      container.on(
        "keydown",
        lambda e, c=container: asyncio.create_task(global_handle_key_press(e, c)),
      )
      ui.timer(
        0.01,
        lambda cid=container.id: ui.run_javascript(
          f'document.getElementById("{cid}").focus()'
        ),
        once=True,
      )

    logger.info("Starting experiment")
    app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)

    # Iterate stages (you have Instructions then the single LLMEnvStage)
    while await experiment_not_finished():
      stage_idx = app.storage.user["stage_idx"]
      stage = experiment.all_stages[stage_idx]

      logger.info("=" * 30)
      logger.info(f"Began stage '{stage.name}'")

      # Clear any stale flags
      await stage.set_user_data(finished=False, stage_finished=False, started=False)

      await run_stage(stage, container)

      logger.info(f"Finished stage '{stage.name}'")

      if isinstance(stage, EnvStage):
        await stage.finish_saving_user_data()
        logger.info(f"Saved data for stage '{stage.name}'")

      # advance
      async with get_user_lock():
        app.storage.user["stage_idx"] = stage_idx + 1

    await finish_experiment(container)

  finally:
    app.storage.user["experiment_running"] = False


# ------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------
@ui.page("/")
async def index(request: Request):
  # initialize per-user state
  app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
  nicewebrl.initialize_user(request=request)

  # include basic helper JS (heartbeat, etc.)
  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  # Layout: single gameplay column (chat lives inside the env stage)
  with ui.row().style("width: 100vw; height: 100vh; overflow: hidden;"):
    with ui.column().style(
      "flex: 1; padding: 16px; overflow-y: auto;"
    ) as gameplay_container:
      asyncio.create_task(start_experiment(gameplay_container))


# ------------------------------------------------------------------------------
# Run app
# ------------------------------------------------------------------------------
ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload=True,
  title="Minigrid Web App",
)
