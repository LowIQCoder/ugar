# Uncertainty Guided Agent Reasoning

UGAR is a thesis project on uncertainty-guided reasoning for software
agents. The current experiments measure how well five uncertainty
estimators detect code defects in archived responses from MBPP, HumanEval,
HumanEval-Java, and Defects4J.

## Experiments

- [Results and interpretation](RESULTS.md)
- [Setup and notebook index](experiments/README)

Use Python 3.12–3.14 in the project's `.venv` and select the
**Python (UGAR)** kernel. The default configuration uses an NVIDIA GPU
and 4-bit Llama 3.1 8B Instruct through Hugging Face Transformers.

## Methodology

The extractor replays 1,198 archived Llama 3 responses and canonical
solutions through Llama 3.1. Each method fits a separate estimator for
each benchmark using task-disjoint splits. The notebooks report
held-out metrics at token, line, and response level.

## Project structure

```text
datasets/
├── benchmarks.py       Dataset loading and task splits
└── repair.py           Repair cases and archived assertions
experiments/
├── data_*.ipynb        Dataset examples and split counts
├── 01–05 notebooks     One executed notebook per estimator
├── 06_agent_loop.ipynb  Uncertainty-guided agent review
├── test.ipynb          Original model-loading notebook
└── README              Setup and experiment settings
src/
├── traces.py           Transformers feature extraction
├── methods.py          Five estimators with a shared interface
├── evaluation.py       Trace caching, benchmark execution, and grading
└── agent_loop.py       LangGraph review and revision
requirements.txt        Python dependencies
```

## Authors

* **Marsel Berheev** (`m.berheev@innopolis.university`)
  * Author of paper
* **Vladimir Ivanov** (`v.ivanov@innopolis.ru`)
  * Thesis supervisor
* **Georgy Andruschenko**
  * Thesis consultant

## Acknowledgments

The experiments build on Georgy Andruschenko's work on code hallucination
detection and the neighboring `llm-uncertainty-layer` repository.
[Efficient Hallucination Detection in Automatic Code Generation](https://aclanthology.org/2026.findings-acl.2143/)

## License

Distributed under [MIT License](./LICENSE)
