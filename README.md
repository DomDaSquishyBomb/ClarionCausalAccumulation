# Sequential Causal Reasoning (SCR) in CLARION

Final-project code for COG403 – Seminar on Cognitive Architectures

"Counterfactual Knowledge: Counterfactual Reasoning as Implicit Knowledge Representation in CLARION"
Dominic Le · University of Toronto

--------------------------------------------------------------------------------
Repository layout
--------------------------------------------------------------------------------

```
.
├── model.py                # SCR agent + helper run_trials()
├── causal_knowledge.py     # Structure-specific causal rules
├── causal_inhibition.py    # Causal inhibition process
├── working_memory.py       # Working-memory pool component
├── lukasiewicz_rules.py    # RuleStore implementing Łukasiewicz (Luk) logic
├── simulations.py          # Generates *_results.csv for each structure
├── stats.py                # χ² tests, descriptive stats, figures/tables
└── requirements.txt        # Python dependencies
```

pyClarion is vendored in `pyClarion/` (installed separately, not tracked in
this repo — see `.gitignore`). Generated result CSVs, plots, and the report
are likewise left untracked; only the source needed to run and test the
model is committed.

--------------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------------

1. Create and activate a virtual environment

   ```
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

3. Make the vendored pyClarion package importable (e.g. add `pyClarion/` to
   `PYTHONPATH`, or run scripts from the repo root where it already lives).

4. Run the simulations

   ```
   python simulations.py
   ```

   This produces CSV files named `{structure}_results.csv`.

5. Re-create stats, χ² tests, and figures/tables

   ```
   python stats.py
   ```

--------------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------------

- Python ≥ 3.12
- pyClarion (vendored, see above)
- pandas, numpy, scipy, matplotlib, seaborn

All pip-installable dependencies are installed via `pip install -r requirements.txt`.

--------------------------------------------------------------------------------
Re-using the model
--------------------------------------------------------------------------------

```python
from model import run_trials

obs = run_trials(n=500, structure="fork", evidence="A", target="C", sd=0.3)
```

`run_trials` returns a list of dicts (one per agent) with keys:

- `match` — `True` if the agent reached the correct target
- `reasoning` — ordered list of chunks chosen during inference
- `A`, `B`, … — counts of how often each chunk was selected

--------------------------------------------------------------------------------
License
--------------------------------------------------------------------------------

MIT License
