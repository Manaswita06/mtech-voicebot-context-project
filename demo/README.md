# Context Reuse Demo

This demo showcases the end-to-end workflow of conversational context extraction and reuse.


Phase 1:

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

Phase 2:
```aiignore
demo/
└── dataset/
    ├── dataset_generator.py      # Main entry point
    └── scenario_distribution.py  # Controls intent/failure distributions
```
generate transcripts

- demo/dataset/scenario_distribution.py: "Instead of generating conversations uniformly at random, I created a distribution controller. It explicitly models the probability of each intent, ambiguity, multi-intent occurrence, tool failures, follow-up conversations, and sentiment. This ensures that datasets generated at different times are reproducible and statistically controlled, making model evaluation fair and repeatable."
  - demo/test_distribution.py: tests the distribution of scenarios for synthetic transcript generation
  - Run 
      ```bash 
      python -m demo.test_distribution
      ```
- generate 500 transcripts
  - Run
    ```bash
    python -m demo.dataset.dataset_generator
    ```

Folder structure

```aiignore
demo/
│
├── run_demo.py                # Main entry point
│
├── context_memory.py          # Stores & retrieves previous conversation context
│
├── context_formatter.py       # Formats previous state for prompt/context
│
├── response_generator.py      # Generates context-aware bot response
│
├── scenario_selector.py       # Picks demo scenarios
│
├── printer.py                 # Pretty console output
│
│
└── README.md
```

Demo flow:

```aiignore
Conversation 1
        │
        ▼
State Extraction Pipeline
        │
        ▼
State JSON
        │
        ▼
ContextMemory.save()
        │
──────── Time Passes ────────
        │
Conversation 2
        │
        ▼
ContextMemory.load()
        │
        ▼
Previous Context
        │
        ▼
LLM Response Generator
```

"Unlike the state extraction module, which processes one conversation at a time, the Context Memory persists the extracted state across customer interactions. When the same customer contacts the system again, the previous conversational context can be retrieved and reused, enabling context-aware responses."

- context_memory.py: simply stores and retrieves the structured state produced by your existing pipeline.
- context_formatter.py: Its only responsibility is to convert the structured state stored in ContextMemory into a concise natural-language summary that can be supplied to the LLM (or displayed during the demo).
- response_generator.py: Its job is simply to generate a context-aware response using the current transcript and the formatted context from ContextFormatter.
- scenario_selector.py: This file simply selects transcripts from your existing dataset for the demo.
