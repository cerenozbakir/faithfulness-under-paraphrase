# Faithfulness Under Paraphrase — Experiment

See the [project README](../README.md) for the research question, result summary, metric definitions, and limitations.

From this directory:

```bash
python -m pip install -r requirements.txt
python src/run_pilot.py
```

The default run requests 15 correctly classified examples and uses German and French translation pivots. It needs Hugging Face downloads and runs on CPU in the current implementation.

**Running the pipeline overwrites `results/pilot_results.csv`. Back up the committed results before a new run.**

| Module | Purpose |
| :--- | :--- |
| `src/data.py` | Load the classifier and SST-2; select correctly classified inputs |
| `src/paraphrase.py` | Generate backtranslations through German and French |
| `src/attribution.py` | Compute SHAP and Layer Integrated Gradients attributions |
| `src/metrics.py` | Compute stability, overlap, edit distance, and BLEU |
| `src/run_pilot.py` | Run the experiment and write the CSV |
| `src/smoke_test.py` | Exercise attribution calls and metrics with an offline toy model |

[Kaggle instructions](KAGGLE_TUTORIAL.md) · [Recorded results](results/pilot_results.csv)
