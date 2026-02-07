# Voicebot Context Extraction and Reuse

This project focuses on extracting conversational context from voicebot transcripts and storing it in a structured form for reuse across future interactions. The goal is to enable conversational continuity across automated systems such as voicebots and IVRs, while also supporting analysis of customer behavior.

The project applies NLP techniques for intent detection, entity recognition, sentiment analysis, and conversation state tracking. Context is modeled at both session and customer levels and evaluated against stateless conversational approaches.

This work is part of an academic M.Tech project at IIT Madras and is implemented using publicly available or synthetic datasets on a personal development environment.

---

# Project Structure

```
mtech-voicebot-context-project/
│
├── data/
│   ├── raw/                    # Original public or synthetic transcripts
│   ├── processed/              # Cleaned and segmented conversations
│   └── README.md               # Dataset description
│
├── preprocessing/
│   ├── clean_text.py
│   ├── segment_conversations.py
│   └── normalize.py
│
├── context_extraction/
│   ├── intent_detection.py
│   ├── entity_recognition.py
│   ├── sentiment_analysis.py
│   ├── topic_modeling.py
│   └── conversation_state.py
│
├── context_model/
│   ├── context_schema.py        # Session & customer context definitions
│   ├── context_storage.py
│   └── context_retrieval.py
│
├── experiments/
│   ├── baseline_vs_context.py
│   ├── cold_vs_warm_start.py
│   └── evaluation_metrics.py
│
├── demo/
│   ├── run_pipeline.py
│   └── sample_outputs/
│
├── reports/
│   ├── figures/
│   └── experiment_results.md
│
├── README.md
├── requirements.txt
└── project_log.md             # Weekly progress log
```
