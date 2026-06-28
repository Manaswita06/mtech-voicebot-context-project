import json

from demo.generators.customers_profiles import generate_customer
from demo.generators.conversation_templates import generate_scenario
from demo.generators.conversation_engine import ConversationEngine

from demo.utils.config import TRANSCRIPT_DIR


customer = generate_customer()

intent, opening, clarification, answer = generate_scenario()

engine = ConversationEngine()

transcript = engine.generate(

    customer=customer,

    intent=intent,

    opening=opening,

    clarification=clarification,

    customer_answer=answer

)

conversation_id = transcript["conversation_id"]

outfile = TRANSCRIPT_DIR / f"{conversation_id}.json"

with open(outfile, "w", encoding="utf8") as f:

    json.dump(transcript, f, indent=2, ensure_ascii=False)

print(outfile)