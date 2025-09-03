import os

from dotenv import load_dotenv
import nicewebrl

load_dotenv()

GOOGLE_CREDENTIALS = "./google-cloud-key.json"
BUCKET_NAME = "jaxmaze"
DATA_DIR = "./data"
DATABASE_FILE = "db.sqlite"

nicewebrl.run(
    storage_secret="private key to secure the browser session cookie",
    experiment_file="experiment_structure.py",
    host="0.0.0.0",
    port=8081,
    title="Custom Web App",
    data_dir=DATA_DIR,
    database_file=DATABASE_FILE,
    reload="FLY_ALLOC_ID" not in os.environ,
)
