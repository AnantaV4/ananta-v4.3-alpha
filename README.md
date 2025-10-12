# Ananta V4.3 — The Conscience Layer (alpha)
This repository is a developer starter skeleton for **Ananta V4.3: The Conscience Layer** — a modular, auditable ethical layer that augments existing AI models with recursive ethical sensitivity (VITO cycles).

This alpha contains minimal, runnable stubs to demonstrate the CovenantEngine loop, Heart and Mind node wrappers, the Coherence Bridge, a basic Corruption Transmutation Layer (CTL), and utilities for metrics logging and an example demo run.

**Structure**
- core/ : main python modules (covenant engine, heart_node, mind_node, bridge, ctl)
- demo/ : demo runner and sample inputs
- docker/ : Dockerfile for containerization
- requirements.txt : minimal dependencies
- README.md : this file
- CONTRIBUTING.md : how to contribute

**Quickstart (local)**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo/run_demo.py
```

Note: this alpha uses lightweight pretrained models and random placeholders for some logic to keep it runnable in low-resource environments. Replace with your preferred models for production.