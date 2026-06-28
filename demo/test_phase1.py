from demo.memory.context_memory import ContextMemory
from demo.utils.printer import *

section("PHASE 1 TEST")

db = ContextMemory()

customer_id = "cust002"

context = {

    "primary_intent": "PAYMENT_ISSUE",

    "secondary_intents": ["STATEMENT_REQUEST"],

    "tool_failure": True,

    "failure_reasons": ["TIMEOUT"],

    "ambiguity_level": 0.34,

    "sentiment_overall": -0.61

}

success("Saving customer context...")

db.save_context(customer_id, context)

success("Done.")

section("Latest Context")

latest = db.latest_context(customer_id)

print(latest)

section("Customer History")

history = db.get_customer_history(customer_id)

print(history)

section("Existing Customers")

print(db.list_customers())