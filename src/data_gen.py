#!/usr/bin/env python3
"""
data_gen_adversarial_finance.py

A highly complicated synthetic conversation generator designed to break TF-IDF baselines.
It maintains the strict schema and structure requested, while injecting adversarial
linguistic patterns like shared vocabulary, negations, and false starts.

Usage:
  python data_gen_adversarial_finance.py --out data/synthetic_adversarial --count 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Shared vocabulary designed to pollute the TF-IDF vectors
FINANCIAL_NOISE = ["account", "transaction", "payment", "card", "system", "charge", "balance", "money"]

INTENTS: Dict[str, Dict] = {
    "CARD_REPLACEMENT": {
        "core_signals": [
            "replace my physical plastic",
            "issue a new chip",
            "plastic is broken",
            "need a fresh piece of plastic",
        ],
        "paraphrases": [
            "I can't make a payment because my card snapped.",  # Cross-pollinating 'payment'
            "The transaction declined because the chip is dead.",  # Cross-pollinating 'transaction'
            "I need something reissued for this account.",
            "The physical item I use for charges was stolen.",
        ],
        "supporting_details": [
            "same account number",
            "current address",
            "mailing address",
            "expedited shipping",
        ],
        "tool": "CARD_SERVICE",
        "resolution": [
            "I have queued the replacement request.",
            "The new plastic is in progress.",
        ],
    },
    "PAYMENT_DUE_DATE": {
        "core_signals": [
            "due date",
            "when I need to pay",
            "next payment",
            "statement due",
        ],
        "paraphrases": [
            "When does the card charge actually hit?",  # Cross-pollinating 'card' and 'charge'
            "I am trying to figure out the account timing.",
            "Can you check the billing cycle for my card?",
            "I want to know when this balance is payable.",
        ],
        "supporting_details": [
            "minimum amount",
            "full balance",
            "latest cycle",
            "billing date",
        ],
        "tool": "ACCOUNT_LOOKUP",
        "resolution": [
            "The billing timing has been checked.",
            "I found the latest due date.",
        ],
    },
    "TRANSACTION_DISPUTE": {
        "core_signals": [
            "charge I don't recognize",
            "unauthorized transaction",
            "dispute a charge",
            "suspicious activity",
        ],
        "paraphrases": [
            "Someone used my card for a payment I didn't make.",  # Cross-pollinating 'card' and 'payment'
            "There is an issue with a charge on my account.",
            "I need help reviewing a posted payment on the card.",
            "I think a transaction needs to be reversed.",
        ],
        "supporting_details": [
            "transaction date",
            "amount",
            "merchant name",
            "reference number",
        ],
        "tool": "TRANSACTION_LOOKUP",
        "resolution": [
            "I have opened a review case.",
            "A dispute workflow has been started.",
        ],
    },
    "ADDRESS_UPDATE": {
        "core_signals": [
            "update my address",
            "change my mailing address",
            "new shipping location",
            "delivery address",
        ],
        "paraphrases": [
            "I need my contact details adjusted before you send the card.",  # Cross-pollinating 'card'
            "I moved and need the account payment info updated.",  # Cross-pollinating 'account' and 'payment'
            "My delivery info for statements is no longer correct.",
            "I need a profile change for my current residence.",
        ],
        "supporting_details": [
            "billing address",
            "street name",
            "zip code",
            "proof of residency",
        ],
        "tool": "PROFILE_UPDATE",
        "resolution": [
            "The profile update is complete.",
            "The new location is now on file.",
        ],
    },
    "PAYMENT_ISSUE": {
        "core_signals": [
            "payment didn't go through",
            "transfer failed",
            "missing payment",
            "remittance problem",
        ],
        "paraphrases": [
            "I used my card but the transaction is stuck.",  # Cross-pollinating 'card' and 'transaction'
            "Something went wrong when I tried to clear the charge.",  # Cross-pollinating 'charge'
            "The account is not reflecting the money I sent.",
            "I am checking an authorization that failed.",
        ],
        "supporting_details": [
            "posted status",
            "pending status",
            "declined code",
            "settlement date",
        ],
        "tool": "PAYMENT_LOOKUP",
        "resolution": [
            "The posting status has been confirmed.",
            "The update is now visible.",
        ],
    },
}

INTENT_NAMES = list(INTENTS.keys())

SCENARIOS = [
    "ambiguous_single",
    "multi_intent_related",
    "multi_intent_unrelated",
    "topic_drift",
    "clarify_then_resolve",
    "tool_failure_retry",
    "escalate_after_failure",
    "noisy_short",
    "noisy_long",
]

FILLERS = ["um", "uh", "well", "actually", "kind of", "maybe", "just checking", "one sec", "I mean", "sort of"]

CONFUSION_PHRASES = [
    "I am not sure", "this is confusing", "something seems off",
    "I cannot tell", "that does not look right", "I need help understanding this"
]

DECOY_INTENTS = {
    "CARD_REPLACEMENT": ["TRANSACTION_DISPUTE", "ADDRESS_UPDATE"],
    "PAYMENT_DUE_DATE": ["PAYMENT_ISSUE", "CARD_REPLACEMENT"],
    "TRANSACTION_DISPUTE": ["PAYMENT_ISSUE", "CARD_REPLACEMENT"],
    "ADDRESS_UPDATE": ["CARD_REPLACEMENT", "PAYMENT_DUE_DATE"],
    "PAYMENT_ISSUE": ["PAYMENT_DUE_DATE", "TRANSACTION_DISPUTE"],
}

TYPO_MAP = {"please": "plz", "account": "acnt", "payment": "pymnt", "statement": "stmt", "address": "adress",
            "replace": "replce", "update": "updte", "card": "crd"}


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def mk_event(event_name: str, role: str, text: str, event_data: Optional[dict] = None) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "event_name": event_name,
        "event_timestamp": utc_now_iso(),
        "participant": {
            "participant_id": f"{role}_{random.randint(1000, 9999)}",
            "role": role,
        },
        "text": text,
        "event_data": event_data or {},
    }


def noisy(text: str, severity: float) -> str:
    if not text: return text
    words = text.split()
    if random.random() < severity:
        words.insert(random.randint(0, len(words)), random.choice(FILLERS))
    out = []
    for w in words:
        core = w.strip(".,!?")
        lw = core.lower()
        new_w = w
        if random.random() < severity * 0.35 and lw in TYPO_MAP:
            repl = TYPO_MAP[lw]
            if core[:1].isupper(): repl = repl.capitalize()
            new_w = repl + w[len(core):]
        elif random.random() < severity * 0.08 and len(core) > 4:
            chars = list(core)
            idx = random.randint(1, len(chars) - 2)
            if random.random() < 0.5:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            else:
                chars.pop(idx)
            new_w = "".join(chars) + w[len(core):]
        out.append(new_w)
    if random.random() < severity * 0.2 and len(out) > 6:
        out.insert(random.randint(2, len(out) - 2), random.choice(["...", "--", ";"]))
    return " ".join(out)


def pick_intent() -> str:
    return random.choice(INTENT_NAMES)


def pick_secondary(primary: str, k: int = 1) -> List[str]:
    choices = [i for i in INTENT_NAMES if i != primary]
    return random.sample(choices, k=min(k, len(choices)))


def choose_scenario() -> str:
    return random.choices(SCENARIOS, weights=[15, 14, 12, 13, 14, 14, 10, 4, 4], k=1)[0]


def opening_text(intent: str, ambiguity: float, noise: float) -> str:
    """Injected Adversarial Logic: Negations and extreme brevity to break TF-IDF."""
    bp = INTENTS[intent]

    # ADV 1: Negation Logic. "I am NOT calling about [decoy]."
    if ambiguity > 0.65 and random.random() < 0.4:
        decoy = random.choice([i for i in INTENT_NAMES if i != intent])
        decoy_signal = random.choice(INTENTS[decoy]["core_signals"])
        primary_signal = random.choice(bp["paraphrases"])
        return noisy(f"I am not calling about a {decoy_signal}. I actually need help because {primary_signal}", noise)

    # ADV 2: Brevity with high noise. Just an elliptical sentence.
    if ambiguity > 0.75 and random.random() < 0.3:
        return noisy(f"My {random.choice(FINANCIAL_NOISE)} is having a {random.choice(bp['core_signals'])} issue.",
                     noise)

    pieces = []
    if ambiguity > 0.35:
        pieces.append(random.choice([
            "I have a question about my account.",
            "Something seems off with my account.",
            "I need help with a couple of things.",
            "Can you check something for me?",
        ]))

    base = random.choice(bp["paraphrases"])
    if random.random() < 0.7:
        pieces.append(base)
    else:
        pieces.append(random.choice(bp["core_signals"]))

    if random.random() < 0.6:
        pieces.append(" ".join(random.sample(bp["supporting_details"], k=2)))

    return noisy(" ".join(pieces), noise)


def agent_prompt(intent: str, stage: str, noise: float) -> str:
    if stage == "opening":
        txt = random.choice([
            "I can help with that. Let me verify your identity first.",
            "Sure, I can look into this.",
            "I will check that for you.",
            "Let me take a look at the account details.",
        ])
    elif stage == "clarify":
        txt = random.choice([
            "Can you clarify which part you need help with?",
            "Could you give me a little more detail?",
            "Do you want me to check the latest record or a previous one?",
            "Can you confirm the date and amount if relevant?",
        ])
    elif stage == "resolve":
        txt = random.choice(INTENTS[intent]["resolution"])
    else:
        txt = "Okay, let me check that."
    return noisy(txt, noise * 0.2)


def sentiment_curve(kind: str, n: int) -> List[float]:
    if n <= 1: return [0.0]
    vals = []
    for i in range(n):
        t = i / (n - 1)
        if kind == "frustrated_to_ok":
            v = -0.7 + 1.4 * t
        elif kind == "ok_to_frustrated":
            v = 0.6 - 1.2 * t
        elif kind == "volatile":
            v = random.uniform(-0.8, 0.8)
        elif kind == "stable_negative":
            v = random.uniform(-0.7, -0.2)
        elif kind == "stable_positive":
            v = random.uniform(0.2, 0.7)
        else:
            v = random.uniform(-0.15, 0.15)
        vals.append(max(-1.0, min(1.0, v + random.uniform(-0.08, 0.08))))
    return vals


def tool_name_for(intent: str) -> str:
    return INTENTS[intent]["tool"]


def tool_event(intent: str, conv_id: str, fail_rate: float = 0.2) -> Tuple[dict, dict, str]:
    tool = tool_name_for(intent)
    tcid = str(uuid.uuid4())
    tool_call = mk_event(
        "AGENT_TOOL_CALLED", "agent", text=f"Calling {tool}",
        event_data={
            "tool_name": tool,
            "tool_call_id": tcid,
            "arguments": {
                "conversation_id": conv_id,
                "intent_hint": intent,
                "request_id": str(uuid.uuid4()),
            },
        },
    )
    status = "fail" if random.random() < fail_rate else "success"
    response = {"tool_name": tool, "tool_call_id": tcid, "status": status, "latency_ms": random.randint(80, 2400)}
    if status == "success":
        response["response"] = {"message": "ok", "confidence": round(random.uniform(0.82, 0.99), 3)}
    else:
        response["error_code"] = random.choice(
            ["INVALID_PHONE", "TIMEOUT", "NO_RECORD_FOUND", "DOWNSTREAM_ERROR", "AUTH_FAIL"])

    tool_resp = mk_event("TOOL_RESPONSE_RECEIVED", "system", text="Tool response", event_data=response)
    return tool_call, tool_resp, status


def make_transcript(idx: int) -> dict:
    scenario = choose_scenario()
    primary = pick_intent()
    secondary: List[str] = []

    if scenario in {"multi_intent_related", "multi_intent_unrelated"}:
        secondary = pick_secondary(primary, k=1 if random.random() < 0.8 else 2)
    elif scenario == "topic_drift":
        secondary = pick_secondary(primary, k=1)
    elif random.random() < 0.14:
        secondary = pick_secondary(primary, k=1)

    ambiguity = {
        "ambiguous_single": random.uniform(0.55, 0.95),
        "multi_intent_related": random.uniform(0.35, 0.85),
        "multi_intent_unrelated": random.uniform(0.2, 0.75),
        "topic_drift": random.uniform(0.3, 0.7),
        "clarify_then_resolve": random.uniform(0.25, 0.55),
        "tool_failure_retry": random.uniform(0.2, 0.5),
        "escalate_after_failure": random.uniform(0.2, 0.55),
        "noisy_short": random.uniform(0.1, 0.35),
        "noisy_long": random.uniform(0.2, 0.5),
    }[scenario]

    noise = {
        "ambiguous_single": random.uniform(0.05, 0.22),
        "multi_intent_related": random.uniform(0.07, 0.25),
        "multi_intent_unrelated": random.uniform(0.08, 0.28),
        "topic_drift": random.uniform(0.08, 0.3),
        "clarify_then_resolve": random.uniform(0.05, 0.18),
        "tool_failure_retry": random.uniform(0.05, 0.22),
        "escalate_after_failure": random.uniform(0.06, 0.25),
        "noisy_short": random.uniform(0.22, 0.5),
        "noisy_long": random.uniform(0.2, 0.45),
    }[scenario]

    conv_id = f"{uuid.uuid4()}@0.0.0.0"
    tenant_id = str(uuid.uuid4())

    events: List[dict] = []

    # Opening user turn.
    events.append(mk_event("USER_MESSAGE_RECEIVED", "user", text=opening_text(primary, ambiguity, noise)))

    if scenario in {"ambiguous_single", "clarify_then_resolve", "multi_intent_related", "multi_intent_unrelated",
                    "topic_drift"}:
        events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=agent_prompt(primary, "opening", noise)))
        events.append(mk_event(
            "USER_MESSAGE_RECEIVED", "user",
            text=noisy(random.choice([
                random.choice(INTENTS[primary]["supporting_details"]),
                random.choice(CONFUSION_PHRASES),
                "Can you be more specific?",
                "I'm not sure which one applies.",
            ]), noise * 0.8),
        ))

    fail_rate = 0.18
    if scenario in {"tool_failure_retry", "escalate_after_failure"}:
        fail_rate = 0.6
    elif scenario == "noisy_long":
        fail_rate = 0.28

    # Primary intent flow
    events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=agent_prompt(primary, "opening", noise)))
    tc, tr, status = tool_event(primary, conv_id, fail_rate=fail_rate)
    events.extend([tc, tr])

    if status == "fail":
        events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=noisy(random.choice([
            "Let me try a different way.", "I will check again.", "I need to retry that request."
        ]), noise * 0.4)))
        tc2, tr2, status2 = tool_event(primary, conv_id, fail_rate=0.45)
        events.extend([tc2, tr2])
        if status2 == "fail":
            events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=noisy(random.choice([
                "I will escalate this for manual review.", "I am transferring this to a specialist."
            ]), noise * 0.3)))

    # Secondary intent / drift.
    if secondary:
        sec = secondary[0]
        drift_lead = random.choice(["Also,", "By the way,", "On another note,", "One more thing,"])
        sec_text = opening_text(sec, ambiguity * 0.7, noise * 0.9)
        events.append(mk_event("USER_MESSAGE_RECEIVED", "user", text=noisy(f"{drift_lead} {sec_text}", noise)))
        events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=agent_prompt(sec, "opening", noise)))

        if random.random() < 0.75:
            tcs, trs, sec_status = tool_event(sec, conv_id,
                                              fail_rate=0.22 if scenario != "multi_intent_unrelated" else 0.32)
            events.extend([tcs, trs])
            if sec_status == "success":
                events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=agent_prompt(sec, "resolve", noise)))
            else:
                events.append(mk_event("AGENT_MESSAGE_SENT", "agent",
                                       text=noisy("I will escalate this second request as well.", noise * 0.3)))
        else:
            events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=agent_prompt(sec, "clarify", noise)))
            events.append(
                mk_event("USER_MESSAGE_RECEIVED", "user", text=noisy("Yes, that's the one I meant.", noise * 0.5)))

    # Optional decoy mention.
    if random.random() < 0.28:
        decoy = random.choice(DECOY_INTENTS[primary])
        events.append(mk_event(
            "USER_MESSAGE_RECEIVED", "user",
            text=noisy(f"Not sure if this matters, but it is kind of like a {decoy.lower()} issue too.", noise * 0.8),
        ))

    events.append(mk_event("AGENT_MESSAGE_SENT", "agent", text=noisy(random.choice([
        "Is there anything else I can help with?",
        "Would you like me to check anything else?",
    ]), noise * 0.25)))

    sentiment_kind = random.choice(
        ["stable_positive", "stable_negative", "volatile", "frustrated_to_ok", "ok_to_frustrated"])
    sentiments = sentiment_curve(sentiment_kind, len(events))
    for ev, s in zip(events, sentiments):
        ev["sentiment_score"] = s

    user_scores = [ev["sentiment_score"] for ev in events if ev["participant"]["role"] == "user"]
    sentiment_overall = sum(user_scores) / len(user_scores) if user_scores else 0.0

    tool_failures = [ev for ev in events if
                     ev["event_name"] == "TOOL_RESPONSE_RECEIVED" and ev["event_data"].get("status") == "fail"]

    return {
        "tenant_id": tenant_id,
        "conversation_id": conv_id,
        "correlation_id": str(uuid.uuid4()),
        "event_month": date.today().replace(day=1).isoformat(),
        "event_timestamp": events[0]["event_timestamp"],
        "events": events,
        "schema_version": 3,
        "ingestion_timestamp": utc_now_iso(),
        "gt_primary_intent": primary,
        "gt_secondary_intents": secondary,
        "gt_multi_intent": len(secondary) > 0,
        "gt_ambiguity_level": round(ambiguity, 3),
        "gt_tool_failure": len(tool_failures) > 0,
        "gt_failure_count": len(tool_failures),
        "gt_sentiment_trajectory": sentiment_kind,
        "gt_sentiment_overall": round(sentiment_overall, 4),
        "gt_turn_count": len(events),
        "gt_scenario_family": scenario,
        "tool_summary": [
            {
                "tool_name": ev["event_data"].get("tool_name"),
                "tool_call_id": ev["event_data"].get("tool_call_id"),
                "status": ev["event_data"].get("status"),
                "error_code": ev["event_data"].get("error_code"),
                "latency_ms": ev["event_data"].get("latency_ms"),
            }
            for ev in events if ev["event_name"] == "TOOL_RESPONSE_RECEIVED"
        ],
    }


def main(out: str, count: int, seed: int):
    random.seed(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(count):
        t = make_transcript(i)
        fp = out_dir / f"transcript_{i:05d}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2, ensure_ascii=False)

        manifest.append({
            "file": fp.name,
            "conversation_id": t["conversation_id"],
            "primary_intent": t["gt_primary_intent"],
            "secondary_intents": t["gt_secondary_intents"],
            "scenario_family": t["gt_scenario_family"],
            "ambiguity": t["gt_ambiguity_level"],
            "tool_failure": t["gt_tool_failure"],
            "turn_count": t["gt_turn_count"],
        })

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated {count} adversarial synthetic transcripts in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_adversarial")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.out, args.count, args.seed)