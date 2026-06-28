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

Phase 3:
```aiignore
demo/
└── dataset/
    ├── dataset_generator.py      # Main entry point
    ├── scenario_distribution.py  # Controls intent/failure distributions
    ├── transcript_factory.py     # Creates transcripts using ConversationEngine
    ├── manifest.py               # Generates dataset statistics
    └── validator.py              # Validates generated transcripts
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
    
Phase 4: State extraction
```aiignore
demo/

    extraction/

        state_extractor.py

        intent_extractor.py

        ambiguity_detector.py

        sentiment_detector.py

        failure_detector.py

        secondary_intent_detector.py

        extractor_pipeline.py            # Validates generated transcripts
```

Flow:
```aiignore
generated/transcripts/

conversation_001.json

        │

        ▼

extractor_pipeline.py

        │

        ▼

generated/states/

conversation_001.state.json
```