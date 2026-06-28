# Context Reuse Demo

This demo showcases the end-to-end workflow of conversational context extraction and reuse.

Pipeline

Synthetic Transcript

↓

State Extraction Pipeline

↓

Context Memory

↓

Follow-up Conversation

↓

Context Retrieval

↓

LLM Response Generation

The objective is to demonstrate how conversational state extracted from previous customer interactions can be reused to improve future customer experience.

```aiignore
demo/
│
├── README.md
├── run_demo.py                 # (Phase 5)
│
├── generators/                 # (Phase 2)
│
├── extraction/                 # (Phase 3)
│
├── llm/                        # (Phase 4)
│
├── memory/
│   ├── context_memory.py
│   └── context_db.json         # Auto-created
│
├── utils/
│   ├── config.py
│   └── printer.py
│
├── generated/
│   ├── transcripts/
│   └── states/
│
└── logs/
```

Phase 1 test:

``bash
python -m demo.test_phase1
``

Phase 2:

- customer_profiles.py models different customer personas.
- conversation_templates.py contains reusable domain-specific dialogue templates.
- tool_simulator.py simulates enterprise backend API/tool interactions and failures.
- conversation_engine.py orchestrates the conversation flow and business logic.
- transcript_builder.py serializes conversation events into the transcript format consumed by the state extraction pipeline.
- transcript_generator.py acts as the entry point that coordinates the generation process.

Run
```bash
python -m demo.generators.transcript_generator
```