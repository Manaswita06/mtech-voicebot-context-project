import json
from pathlib import Path
from datetime import datetime

from demo.utils.config import CONTEXT_DB


class ContextMemory:

    def __init__(self):

        if not CONTEXT_DB.exists():

            CONTEXT_DB.parent.mkdir(parents=True, exist_ok=True)

            with open(CONTEXT_DB, "w") as f:
                json.dump({}, f, indent=2)

    def _load(self):

        with open(CONTEXT_DB) as f:
            return json.load(f)

    def _save(self, db):

        with open(CONTEXT_DB, "w") as f:
            json.dump(db, f, indent=2)

    def save_context(self, customer_id, context):

        db = self._load()

        if customer_id not in db:
            db[customer_id] = []

        db[customer_id].append({

            "timestamp": datetime.now().isoformat(),

            "context": context

        })

        self._save(db)

    def get_customer_history(self, customer_id):

        db = self._load()

        return db.get(customer_id, [])

    def latest_context(self, customer_id):

        history = self.get_customer_history(customer_id)

        if not history:

            return None

        return history[-1]["context"]

    def customer_exists(self, customer_id):

        db = self._load()

        return customer_id in db

    def list_customers(self):

        db = self._load()

        return list(db.keys())