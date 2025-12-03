from typing import Optional, Tuple, Union, Any
import random
from nicegui import ui, app
from asyncio import Lock

# This is used to ensure that each user has a unique lock
_user_locks = {}
global_lock = Lock()


def get_user_lock():
  """A function that returns a lock for the current user using their unique seed"""
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]

def args_from_room(args):
  user_room = str(app.storage.user.get("room_id", None))
  if user_room != str(args["called_by_room_id"]):
    return False
  return True

async def get_room_stage_object(key: str, default: Optional[Any] = None) -> Any:
  """Retrieves a value from the room-stage specific storage system.

  This function retrieves values from a hierarchical storage system that maintains
  separate states for different combinations of rooms and stages.

  Args:
      key (str): The identifier for the value to retrieve.
      default (Optional[Any]): Value to return if the key doesn't exist. Defaults to None.

  Returns:
      Any: The stored value for the given key, or the default value if not found.

  Example:
      >>> # Get a dictionary tracking user completion
      >>> checked = get_room_stage_object('checked', {})
      >>> print(checked)  # {'user1': True, 'user2': False}

      >>> # Get a value with a default
      >>> score = get_room_stage_object('current_score', default=0)
  """
  async with get_user_lock():
    room_id = app.storage.user["room_id"]
    stage_idx = app.storage.user["stage_idx"]
    room_stage = f"{room_id}_{stage_idx}"
    all_room_objects = app.storage.general.get("room_objects", {})
    room_stage_objects = all_room_objects.get(room_stage, {})
    return room_stage_objects.get(key, default)

async def set_room_stage_object(key: str, value: Any) -> None:
  """Stores a value in the room-stage specific storage system.

  This function implements a hierarchical storage system that maintains separate
  states for different combinations of rooms and stages. It's primarily used
  for maintaining game state in a multi-room, multi-stage environment.

  Storage Structure:
      app.storage.general['room_objects'] = {
          'room1_stage0': {
              'key1': value1,
              'key2': value2
          },
          'room2_stage1': {
              'key1': value3,
              ...
          }
      }

  Args:
      key (str): The identifier for the value being stored.
      value (Any): The value to store. Can be any serializable object.

  Example:
      >>> # Store a dictionary tracking user completion
      >>> checked = {'user1': True, 'user2': False}
      >>> set_room_stage_object('checked', checked)

      >>> # Store a simple value
      >>> set_room_stage_object('current_score', 100)

  Note:
      - The room_id and stage_idx are automatically retrieved from app.storage.user
      - Values are stored in app.storage.general['room_objects']
      - Previous values for the same key will be overwritten
  """
  async with get_user_lock():
    room_id = app.storage.user["room_id"]
    stage_idx = app.storage.user["stage_idx"]
    room_stage = f"{room_id}_{stage_idx}"
    all_room_objects = app.storage.general.get("room_objects", {})
    room_stage_objects = all_room_objects.get(room_stage, {})
    room_stage_objects[key] = value
    all_room_objects[room_stage] = room_stage_objects
    app.storage.general["room_objects"] = all_room_objects

def get_room_users():
  room_id = app.storage.user["room_id"]
  return app.storage.general["rooms"][str(room_id)]


def add_user_to_room(max_users_per_room: int) -> Tuple[str, list]:
  """_summary_

  First person in room is leader. In charge of creating environments.
  Returns:
      _type_: _description_
  """
  # get user id
  user_id = str(app.storage.user["seed"])

  # get rooms
  rooms = app.storage.general.get("rooms", {})
  user_to_action_idx = app.storage.general.get("user_to_action_idx", {})
  print("Available rooms:")
  print(rooms)

  # if no rooms, create one
  app.storage.general["latest_room_id"] = app.storage.general.get(
      "latest_room_id", None
  )
  no_rooms = len(rooms) == 0

  def new_room():
    # get new room and new room ID
    latest_room_id = str(random.getrandbits(32))
    rooms[latest_room_id] = [user_id]
    latest_room = rooms[latest_room_id]
    app.storage.general["latest_room_id"] = latest_room_id
    app.storage.user["leader"] = True
    user_to_action_idx[user_id] = 0

    return latest_room_id, latest_room

  if no_rooms:
    latest_room_id, latest_room = new_room()
  else:
    latest_room_id = str(app.storage.general["latest_room_id"])
    latest_room = rooms[latest_room_id]

    # if the latest room is full, create a new one
    if len(latest_room) >= max_users_per_room:
      # create new id
      latest_room_id, latest_room = new_room()
    else:
      app.storage.user["leader"] = False
      user_to_action_idx[user_id] = len(latest_room)
      latest_room.append(user_id)

  app.storage.general["rooms"] = rooms
  app.storage.general["user_to_action_idx"] = user_to_action_idx
  return latest_room_id, latest_room


async def setup_room():
  await ui.run_javascript(f"window.room = '{app.storage.user['room_id']}';", timeout=10)
  await ui.run_javascript(f"window.user_id = '{app.storage.user['seed']}';", timeout=10)