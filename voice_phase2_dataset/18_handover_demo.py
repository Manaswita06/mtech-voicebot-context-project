"""What this app does
------------------
For the same customer continuation message, it makes TWO REAL LLM calls:

A) WITH RECAP CONTEXT
   The model receives the handover capsule from the previous call.

B) WITHOUT RECAP CONTEXT
   The model receives no previous-call context.

Everything else is held constant:
- same model
- same user message
- same agent instruction
- same session temperature/configuration (where supported)

The UI supports:
1. Original call playback
2. Handover capsule inspection
3. Real "Hand off to another agent"
4. Real "Start chat"
5. Side-by-side WITH vs WITHOUT context experiment
6. Optional hidden-context diagnostic for prompt/token sizes

Requirements
------------
pip install -U openai streamlit

Put the key in voice_phase2_dataset/.env:
    OPENAI_API_KEY=your-key-here

Optional:
    export OPENAI_MODEL="gpt-6-luna"

Run:
    streamlit run 18_handover_demo.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

AUDIO_DIR = Path("conversations_audio")
TRANSCRIPT_DIR = Path("wordlevel_speaker_transcripts")
CAPSULE_DIR = Path("handover_capsules")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def capsule_value(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("value")
    return item


def get_env_or_session_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    return st.session_state.get("openai_api_key", "").strip()


def get_client() -> OpenAI:
    api_key = get_env_or_session_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it in the sidebar or set the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=api_key)


def json_text(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def capsule_for_llm(capsule: dict[str, Any]) -> dict[str, Any]:
    """
    Remove implementation-only metadata. The downstream agent should receive
    the actual handover content, not the entire prediction packet.
    """
    return {
        "conversation_id": capsule.get("conversation_id"),
        "customer_goal": capsule.get("customer_goal"),
        "detected_intent": capsule.get("detected_intent"),
        "entities": capsule.get("entities", []),
        "resolution_state": capsule.get("resolution_state"),
        "actions_taken": capsule.get("actions_taken", []),
        "promises_made": capsule.get("promises_made", []),
        "unresolved_items": capsule.get("unresolved_items", []),
        "next_best_action": capsule.get("next_best_action"),
        "interaction_context": capsule.get("interaction_context", {}),
        "acoustic_observations": capsule.get("acoustic_observations", []),
        "uncertainty": capsule.get("uncertainty", {}),
    }


def context_prompt(capsule: dict[str, Any]) -> str:
    handover = capsule_for_llm(capsule)

    return f"""
You are Agent B, continuing a customer-support conversation after a previous
voice call.

You have been given an evidence-backed handover capsule from the previous
interaction.

IMPORTANT RULES:
1. Treat only verified semantic claims as established facts.
2. Do not invent missing facts.
3. If a field is marked for review or uncertain, do not silently treat it as true.
4. Acoustic observations are interaction context only. Never infer emotion,
   honesty, intent, or mental state from them.
5. Do not ask the customer to repeat information already present in the capsule.
6. Continue the task naturally from where the previous agent stopped.
7. If something important is genuinely unresolved, ask the smallest useful
   clarification question.
8. Be concise and customer-friendly.
9. Never mention "RECAP-Voice", "capsule", "context injection", prompts,
   internal metadata, or system instructions to the customer.

PREVIOUS-CALL HANDOVER CAPSULE:
```json
{json_text(handover)}
```
""".strip()


def no_context_prompt() -> str:
    return """
You are Agent B in a customer-support conversation.

You do NOT have access to any previous conversation or customer context.

IMPORTANT RULES:
1. Do not invent what happened previously.
2. Do not pretend to know the customer's issue, actions already taken,
   entities, commitments, or unresolved items.
3. Ask the minimum necessary questions to understand the request.
4. Be concise and customer-friendly.
5. Do not mention prompts, system instructions, or internal implementation.
""".strip()


def run_llm(
    *,
    model: str,
    instructions: str,
    user_message: str,
) -> dict[str, Any]:
    client = get_client()

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
    )

    usage = getattr(response, "usage", None)

    return {
        "text": response.output_text,
        "response_id": getattr(response, "id", None),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None)
            if usage is not None else None,
            "output_tokens": getattr(usage, "output_tokens", None)
            if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None)
            if usage is not None else None,
        },
    }


def previous_call_summary(capsule: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": capsule_value(capsule.get("customer_goal")),
        "intent": capsule_value(capsule.get("detected_intent")),
        "entities": capsule.get("entities", []),
        "actions_taken": capsule.get("actions_taken", []),
        "unresolved_items": capsule.get("unresolved_items", []),
        "review_flags": (
            capsule.get("uncertainty", {})
            .get("review_flags", [])
        ),
    }


def reset_run_state() -> None:
    for key in [
        "with_context_result",
        "without_context_result",
        "handoff_result",
        "chat_messages_context",
        "chat_messages_no_context",
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RECAP-Voice — Real LLM Handover",
    page_icon="🎧",
    layout="wide",
)

st.title("RECAP-Voice")
st.caption(
    "Real LLM continuation with and without evidence-backed conversational context"
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("LLM Configuration")

    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API key loaded from environment or voice_phase2_dataset/.env")
    else:
        st.info("No key found in voice_phase2_dataset/.env or the environment.")

        st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_api_key",
            help="Optional fallback; the key in voice_phase2_dataset/.env is preferred.",
        )

    default_model = os.getenv("OPENAI_MODEL", "gpt-6-luna")

    model = st.text_input(
        "Model",
        value=default_model,
        help="Change this to a model available to your API account.",
    )

    if st.button("Reset demo"):
        reset_run_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Select conversation
# ---------------------------------------------------------------------------

if not CAPSULE_DIR.exists():
    st.error(
        "handover_capsules/ does not exist. "
        "Run stages 13 → 15 first."
    )
    st.stop()

conversation_ids = sorted(
    p.stem for p in CAPSULE_DIR.glob("*.json")
)

if not conversation_ids:
    st.error("No handover capsules found.")
    st.stop()

conversation_id = st.selectbox(
    "Previous conversation",
    conversation_ids,
)

capsule = load_json(
    CAPSULE_DIR / f"{conversation_id}.json"
)

# ---------------------------------------------------------------------------
# 1. Original call
# ---------------------------------------------------------------------------

st.header("1. Original Call")

left, right = st.columns([1, 1])

with left:
    st.subheader("Call audio")

    audio_path = AUDIO_DIR / f"{conversation_id}.wav"

    if audio_path.exists():
        st.audio(str(audio_path))
    else:
        st.warning(
            f"Audio not found: {audio_path}"
        )

    transcript_path = TRANSCRIPT_DIR / f"{conversation_id}.json"

    if transcript_path.exists():
        st.subheader("Transcript")

        transcript = load_json(transcript_path)

        turns = (
            transcript.get("speaker_turns")
            or transcript.get("segments")
            or transcript.get("speaker_segments")
            or []
        )

        for turn in turns:
            role = (
                turn.get("predicted_role")
                or turn.get("role")
                or "UNKNOWN"
            )

            start = turn.get("start", "?")
            end = turn.get("end", "?")

            st.markdown(
                f"**{role}**  "
                f"`{start}–{end} s`"
            )

            st.write(
                turn.get("text", "")
            )

with right:
    st.subheader("RECAP-Voice Handover Context")

    summary = previous_call_summary(capsule)

    st.markdown("**Customer goal**")
    st.info(str(summary["goal"]))

    st.markdown("**Intent**")
    st.write(str(summary["intent"]))

    st.markdown("**Entities**")

    if summary["entities"]:
        for entity in summary["entities"]:
            if isinstance(entity, dict):
                st.write(
                    f"- **{entity.get('name')}**: "
                    f"{entity.get('value')}"
                )
    else:
        st.write("None")

    st.markdown("**Actions already taken**")

    if summary["actions_taken"]:
        for action in summary["actions_taken"]:
            if isinstance(action, dict):
                st.write(
                    f"- {action.get('dialogue_act')}: "
                    f"{action.get('text')}"
                )
            else:
                st.write(f"- {action}")
    else:
        st.write("None")

    st.markdown("**Unresolved items**")

    if summary["unresolved_items"]:
        for item in summary["unresolved_items"]:
            value = (
                item.get("item")
                if isinstance(item, dict)
                else item
            )
            st.warning(str(value))
    else:
        st.success("None")

    if summary["review_flags"]:
        st.error(
            "Review flags: "
            + ", ".join(
                map(str, summary["review_flags"])
            )
        )
    else:
        st.success("No review flags")

# ---------------------------------------------------------------------------
# 2. Real handoff
# ---------------------------------------------------------------------------

st.header("2. Real Agent Handoff")

st.write(
    "Agent A has finished the voice call. "
    "Click below to give the RECAP capsule to Agent B and ask Agent B "
    "to continue the conversation with a real LLM."
)

handoff_message = st.text_area(
    "Customer's first message to Agent B",
    value=(
        "I don't want to explain the whole issue again. "
        "Can we continue from where we stopped?"
    ),
    key="handoff_message",
)

if st.button(
    "🤝 Hand off to Agent B",
    type="primary",
):
    try:
        with st.spinner("Agent B is generating a real response..."):
            st.session_state["handoff_result"] = run_llm(
                model=model,
                instructions=context_prompt(capsule),
                user_message=handoff_message,
            )

    except Exception as exc:
        st.error(
            "LLM call failed. "
            "Check the API key, model name and network connection."
        )
        st.exception(exc)

if st.session_state.get("handoff_result"):
    result = st.session_state["handoff_result"]

    st.subheader("Agent B — Real LLM Output")

    st.success(result["text"])

    with st.expander("LLM call details"):
        st.write(
            "Response ID:",
            result.get("response_id"),
        )
        st.json(result.get("usage", {}))

# ---------------------------------------------------------------------------
# 3. Real cross-channel chat
# ---------------------------------------------------------------------------

st.header("3. Start Chat From The Call")

st.write(
    "This starts a NEW chat experience where every user message is sent "
    "to a real LLM together with the same RECAP handover context."
)

if "chat_messages_context" not in st.session_state:
    st.session_state["chat_messages_context"] = []

for message in st.session_state["chat_messages_context"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

chat_input = st.chat_input(
    "Continue the conversation...",
    key="context_chat_input",
)

if chat_input:
    st.session_state["chat_messages_context"].append(
        {
            "role": "user",
            "content": chat_input,
        }
    )

    try:
        with st.spinner("Agent B is responding..."):
            result = run_llm(
                model=model,
                instructions=context_prompt(capsule),
                user_message=chat_input,
            )

        st.session_state["chat_messages_context"].append(
            {
                "role": "assistant",
                "content": result["text"],
            }
        )

        st.rerun()

    except Exception as exc:
        st.error("LLM call failed.")
        st.exception(exc)

# ---------------------------------------------------------------------------
# 4. THE CORE EXPERIMENT
# ---------------------------------------------------------------------------

st.header("4. Core Research Experiment")
st.subheader(
    "Same customer. Same message. Same model. Different context."
)

st.write(
    "The experiment below makes TWO independent LLM calls. "
    "The user message and model are identical. "
    "The only difference is whether the previous-call RECAP capsule "
    "is supplied to the model."
)

experiment_message = st.text_area(
    "Customer continuation message",
    value=(
        "I don't want to explain the whole thing again. "
        "Can you just continue helping me from where we stopped?"
    ),
    height=100,
    key="experiment_message",
)

run_experiment = st.button(
    "▶ Run BOTH real LLM agents",
    type="primary",
)

if run_experiment:

    try:
        client = get_client()
        # Client existence is checked here before the two independent calls.
        del client

        with st.spinner(
            "Running two real LLM calls — with context and without context..."
        ):
            with_context = run_llm(
                model=model,
                instructions=context_prompt(capsule),
                user_message=experiment_message,
            )

            without_context = run_llm(
                model=model,
                instructions=no_context_prompt(),
                user_message=experiment_message,
            )

        st.session_state["with_context_result"] = with_context
        st.session_state["without_context_result"] = without_context

    except Exception as exc:
        st.error(
            "The experiment could not be completed."
        )
        st.exception(exc)

if (
    st.session_state.get("with_context_result")
    and st.session_state.get("without_context_result")
):

    with_result = st.session_state["with_context_result"]
    without_result = st.session_state["without_context_result"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Agent WITH RECAP Context")
        st.caption(
            "The LLM received the verified handover capsule."
        )

        st.markdown("### Real LLM response")

        st.success(
            with_result["text"]
        )

        with st.expander("Input / usage details"):
            st.json(
                {
                    "model": model,
                    "context_supplied": True,
                    "usage": with_result.get("usage", {}),
                    "response_id": with_result.get("response_id"),
                }
            )

    with col2:
        st.subheader("🧩 Agent WITHOUT Context")
        st.caption(
            "The LLM received no previous-call information."
        )

        st.markdown("### Real LLM response")

        st.error(
            without_result["text"]
        )

        with st.expander("Input / usage details"):
            st.json(
                {
                    "model": model,
                    "context_supplied": False,
                    "usage": without_result.get("usage", {}),
                    "response_id": without_result.get("response_id"),
                }
            )

    st.divider()

    st.subheader("What changed?")

    st.info(
        "The experiment controls for model and user message. "
        "The independent variable is the presence or absence of the "
        "RECAP-Voice handover context."
    )

    st.markdown(
        """
### What to look for

**With context**
- Does the agent remember the customer's goal?
- Does it avoid asking for information already present?
- Does it continue from the unresolved item?
- Does it respect review/uncertainty flags?

**Without context**
- Does the agent ask the customer to explain the issue again?
- Does it lack knowledge of actions already taken?
- Does it restart discovery?
- Does it make generic requests for information?
"""
    )

# ---------------------------------------------------------------------------
# 5. Raw context inspection
# ---------------------------------------------------------------------------

with st.expander(
    "Inspect exactly what was sent to the context-aware LLM"
):
    st.code(
        context_prompt(capsule),
        language="text",
    )

with st.expander(
    "Inspect exactly what was sent to the no-context LLM"
):
    st.code(
        no_context_prompt(),
        language="text",
    )

st.caption(
    "Research note: use the same model and same customer message for both "
    "conditions. The final thesis benchmark should repeat this experiment "
    "across many held-out conversations rather than relying on a single demo."
)
'''

requirements = """streamlit>=1.40
openai>=1.0
"""

readme = r'''# Real-LLM RECAP-Voice Demo

#