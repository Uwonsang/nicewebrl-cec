import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import jax
import example_config as config
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import TimeStep
import time
import json
from upload_google_data import save_to_gcs_with_retries, GOOGLE_CREDENTIALS
from experiment_structure import experiment  

# --- Data saving toggles (module-level constants) ---
KEEP_LOCAL_COPIES = True                   # keep .msgpack and metadata locally
ENABLE_GCS_UPLOAD = False                  # turn on only when you actually want to push to GCS
GCS_KEY_PATH = "./google-cloud-key.json"   # path to your creds file (if using GCS)

DATA_DIR = "data"
DATABASE_FILE = "db.sqlite"

_user_locks = {}

def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]

# --------------------------------------------------------------------
# Key handling: container-scoped + active_container guard
# --------------------------------------------------------------------
from types import SimpleNamespace

async def global_handle_key_press(e, container):
    logger.info("global_handle_key_press")
    if experiment.finished():
        logger.info("Experiment finished")
        return

    stage = await experiment.get_stage()
    if stage.get_user_data("finished", False):
        return

    # Hard guard: ignore all keys while the user is typing anywhere.
    is_typing = await ui.run_javascript("""
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

    # Optional guard: if the focused element is exactly the LLM prompt box, ignore.
    active_id = await ui.run_javascript('document.activeElement ? document.activeElement.id : ""')
    prompt_box_id = stage.get_user_data("prompt_box_id")
    if prompt_box_id and str(active_id) == str(prompt_box_id):
        return

    # Only pass keys to the env when the env is the active container.
    if stage.get_user_data("active_container") != "env":
        return

    # --- Attach timestamps so they get saved with the action ---
    try:
        image_ts = await ui.run_javascript(
            "window.imageSeenTime ? new Date(window.imageSeenTime).toISOString() : null"
        )
    except Exception:
        image_ts = None

    try:
        key_ts = await ui.run_javascript("new Date().toISOString()")
    except Exception:
        key_ts = None

    # Wrap the event with augmented args (don’t mutate NiceGUI’s event in place)
    base_args = dict(getattr(e, "args", {}) or {})
    base_args.setdefault("key", getattr(getattr(e, "args", None), "get", lambda *_: None)("key"))
    base_args["imageSeenTime"] = image_ts
    base_args["keydownTime"] = key_ts
    wrapped_event = SimpleNamespace(args=base_args)

    await stage.handle_key_press(wrapped_event, container)

    local_handle_key_press = stage.get_user_data("local_handle_key_press")
    if local_handle_key_press is not None:
        await local_handle_key_press()

# --------------------------------------------------------------------
# Setup logging & local data dir
# --------------------------------------------------------------------
setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
logger = get_logger("main")

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)

# --------------------------------------------------------------------
# DB lifecycle
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# Minimal consent + demographics 
# --------------------------------------------------------------------
async def make_consent_form(container):
  consent_given = asyncio.Event()
  with container:
    ui.markdown("## Consent Form")
    with open("consent.md", "r") as consent_file:
      consent_text = consent_file.read()
    ui.markdown(consent_text)

    def on_change():
      consent_given.set()

    ui.checkbox("I agree to participate.", on_change=on_change)
  await consent_given.wait()

async def collect_demographic_info(container):
  nicewebrl.clear_element(container)
  collected_demographic_info_event = asyncio.Event()
  with container:
    ui.markdown("## Demographic Info")
    ui.markdown("Please fill out the following information.")

    with ui.column():
      with ui.column():
        ui.label("Biological Sex")
        sex_input = ui.radio(["Male", "Female"], value="Male").props("inline")
      age_input = ui.input("Age")

    async def submit():
      age = age_input.value
      sex = sex_input.value
      if not age.isdigit() or not (0 < int(age) < 100):
        ui.notify("Please enter a valid age between 1 and 99.", type="warning")
        return
      app.storage.user["age"] = int(age)
      app.storage.user["sex"] = sex
      logger.info(f"age: {int(age)}, sex: {sex}")
      collected_demographic_info_event.set()

    button = ui.button("Submit", on_click=submit)
    await button.clicked()

# --------------------------------------------------------------------
# Experiment flow (containers for stage/meta only; chat is inside env stages)
# --------------------------------------------------------------------
async def start_experiment(meta_container, stage_container):
  if not (app.storage.user.get("experiment_started", False)):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

  # Container-scoped key handling; make sure container is focusable
  stage_container.props('tabindex=0')
  stage_container.on('keydown', lambda e, sc=stage_container: global_handle_key_press(e, sc))
  ui.timer(0.01, lambda sc=stage_container: ui.run_javascript(f'document.getElementById("{sc.id}").focus()'), once=True)

  logger.info("Starting experiment")

  while not experiment.finished():
    stage = await experiment.get_stage()
    await run_stage(stage, stage_container)
    await stage.finish_saving_user_data()
    await experiment.advance_stage()

  await finish_experiment(meta_container)

async def finish_experiment(container):
  nicewebrl.clear_element(container)
  with container:
    ui.markdown("# Experiment over")

  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    status_container = None
    with container:
      nicewebrl.clear_element(container)
      ui.markdown("## Your data is being saved. Please do not close or refresh the page.")
      status_container = ui.markdown("Saving local files...")

    try:
      save_task = asyncio.create_task(save_data(feedback=feedback))
      start_time = time.time()

      while not save_task.done():
        elapsed_seconds = int(time.time() - start_time)
        status_container.content = (
          f"Still saving... ({elapsed_seconds}s elapsed). This may take 5-10 minutes."
        )
        try:
          await asyncio.wait_for(asyncio.shield(save_task), timeout=2.0)
        except asyncio.TimeoutError:
          continue
        except Exception as e:
          logger.error(f"Error during save: {e}")
          status_container.content = "⚠️ Error saving data. Please contact the experimenter."
          raise

      elapsed_seconds = int(time.time() - start_time)
      status_container.content = (
        f"Save complete in {elapsed_seconds}s! Moving to next screen..."
      )
      app.storage.user["data_saved"] = True

    except Exception as e:
      logger.error(f"Save failed: {e}")
      status_container.content = "⚠️ Error saving data. Please contact the experimenter."
      raise

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with container:
      nicewebrl.clear_element(container)
      ui.markdown("Please provide feedback on the experiment here.")
      text = ui.textarea().style("width: 80%;")
      button = ui.button("Submit")
      await button.clicked()
      await submit(text.value)

  with container:
    nicewebrl.clear_element(container)
    ui.markdown("# Experiment over")
    ui.markdown("## Data saved")
    ui.markdown("### Please record the following code for compensation")
    ui.markdown("### 'carvalho.assistants 3'")
    ui.markdown("#### You may close the browser")

async def save_data(feedback=None, **kwargs):
  user_data_file = nicewebrl.user_data_file()
  user_metadata_file = nicewebrl.user_metadata_file()

  # Always write a fresh metadata JSON alongside the msgpack
  user_storage = nicewebrl.make_serializable(dict(app.storage.user))
  metadata = dict(
    finished=True,
    feedback=feedback,
    user_storage=user_storage,
    **kwargs,
  )
  import json, os
  with open(user_metadata_file, "w") as f:
    json.dump(metadata, f)

  files_to_save = [user_data_file, user_metadata_file]

  uploaded = False
  try:
    if ENABLE_GCS_UPLOAD:
      from upload_google_data import save_to_gcs_with_retries, GOOGLE_CREDENTIALS
      # Require both a non-empty GOOGLE_CREDENTIALS and an existing key file path
      key_ok = isinstance(GOOGLE_CREDENTIALS, str) and os.path.exists(GCS_KEY_PATH)
      if key_ok:
        logger.info(f"Saving to bucket: {config.BUCKET_NAME}")
        await save_to_gcs_with_retries(
          files_to_save,
          max_retries=5,
          bucket_name=config.BUCKET_NAME,
        )
        uploaded = True
      else:
        logger.info("Skipping GCS upload (credentials disabled or key file missing).")
    else:
      logger.info("GCS upload disabled; keeping local files.")
  except Exception as e:
    logger.warning(f"GCS upload failed: {e}; keeping local files.")

  # Clean up DB records (optional; doesn’t touch the msgpack)
  try:
    from nicewebrl.stages import StageStateModel
    logger.info(f"Deleting DB stage rows for user {app.storage.browser.get('id')}")
    await StageStateModel.filter(session_id=app.storage.browser.get("id")).delete()
  except Exception as e:
    logger.warning(f"Could not clear StageStateModel rows: {e}")

  # Delete local files ONLY if upload succeeded AND you explicitly opt in
  if uploaded and not KEEP_LOCAL_COPIES:
    for local_file in files_to_save:
      try:
        os.remove(local_file)
        logger.info(f"Deleted local file after upload: {local_file}")
      except Exception as e:
        logger.warning(f"Failed to delete local file {local_file}: {e}")
  else:
    logger.info(f"Keeping local files: {files_to_save}")

async def run_stage(stage, container):
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

  with container.style("align-items: center;"):
    await stage.activate(container)

  if stage.get_user_data("finished", False):
    logger.info(f"Finished {stage.name} immediately after activation")
    stage_over_event.set()

  if stage.next_button:
    with container:
      button = ui.button("Next page")
      await wait_for_button_or_keypress(button)
      await handle_button_press()

  await stage_over_event.wait()

# --------------------------------------------------------------------
# Main page: minimal containers only; chat UI lives inside env stages
# --------------------------------------------------------------------
@ui.page("/")
async def index(request: Request):
    nicewebrl.initialize_user(request=request)
    await experiment.initialize()

    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html("<script>" + f.read() + "</script>")

    card = (
        ui.card(align_items=["center"])
        .classes("fixed-center")
        .style(
            "width: 80vw; max-height: 90vh; overflow: auto; display: flex; flex-direction: column; justify-content: flex-start; align-items: center; padding: 1rem;"
        )
    )

    with card:
        meta_container = ui.column()
        with meta_container.style("align-items: center;"):
            display_container = ui.row()
            with display_container.style("align-items: center;"):
                stage_container = ui.column()
                # optional heartbeat/timer
                ui.timer(interval=10, callback=lambda: None)

            footer_container = ui.row()

        with meta_container.style("align-items: center;"):
            await footer(footer_container)
            with display_container.style("align-items: center;"):
                await start_experiment(display_container, stage_container)

async def footer(footer_container):
  with footer_container:
    with ui.row():
      ui.label().bind_text_from(app.storage.user, "user_id", lambda v: f"user id: {v}.")
      def text_display(v):
        stage_idx = max(experiment.num_stages, int(v) + 1)
        return f"stage: {stage_idx}/{experiment.num_stages}."
      ui.label().bind_text_from(app.storage.user, "stage_idx", text_display)
      ui.label().bind_text_from(app.storage.user, "session_duration", lambda v: f"minutes passed: {int(v)}.")
    ui.linear_progress(value=nicewebrl.get_progress()).bind_value_from(app.storage.user, "stage_progress")
    ui.button("Toggle fullscreen", icon="fullscreen", on_click=nicewebrl.utils.toggle_fullscreen).props("flat")

ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload="FLY_ALLOC_ID" not in os.environ,
  title="Minigrid Web App",
  port=8080,
)
