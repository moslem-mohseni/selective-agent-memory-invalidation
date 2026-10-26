"""Regenerate the project findings from current result files."""
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent.parent
t=pd.read_csv(root/'results/table1_main.csv').set_index('policy')
save=100*(1-t.loc['selective_tool','probe_units']/t.loc['revalidate_all','probe_units'])
cost=100*(t.loc['clear_all','cost']/t.loc['selective_tool','cost']-1)
text=f'''# Selective Invalidation of Procedural Agent Memory under Tool Contract Changes

**فارسی:** مقایسهٔ شبیه‌سازی سیاست‌های نگه‌داری حافظه پس از تغییر قرارداد ابزار؛ هیچ سیاستی در همهٔ شرایط برتر نیست.

## Current findings
Tool-level selection has mean success {t.loc['selective_tool','success']:.3f}, versus {t.loc['revalidate_all','success']:.3f} for exhaustive revalidation, with {save:.1f}% less probe effort in the default simulation. Clearing achieves {t.loc['clear_all','success']:.3f} at {cost:.1f}% higher call-equivalent cost than tool selection. Clause-level success is {t.loc['selective_clause','success']:.3f}. A small or zero difference is not an equivalence test. Results depend on the executable grammar and parametric assumptions.

## Reproduce from source
Use Python 3.12 and the pinned requirements. Native Windows execution is provided alongside Bash. Word is needed only for the Windows PDF export; the Docker pipeline creates the DOCX and all scientific outputs.

```bash
pip install -r requirements.txt
bash run_all.sh
```

```powershell
./run_all.ps1
# Without Microsoft Word:
./run_all.ps1 -SkipPdf
```

```bash
docker build -t p1_memory_invalidation .
docker run --name p1_reproduction p1_memory_invalidation
docker cp p1_reproduction:/work/results ./container_results
docker cp p1_reproduction:/work/paper/supplementary_p1.zip ./
```

The pipeline runs tests, all experiments, analysis, figures, an explicitly mocked Study 2 dry run, manuscript generation, number/citation/template audits and supplementary packaging. A clean Docker context excludes old results and figures. Fixed seed streams separate main, confirmation and newly added exploratory sensitivity conditions. Compare CSV contents with pandas rather than file line endings.

## Study 2
No real language model was run. No API-key variables or ready local model were available. The official Ollama runtime acquisition timed out; the partial installer was never executed. Exact public attempt evidence is in `src/llm_study/availability_attempt.json`. Mock output is labelled and excluded from scientific inference.

The provider-independent client supports Ollama, llama.cpp-compatible serving, OpenAI and Anthropic, with cached transcripts and a hard default five-dollar per-paper cap. Paid endpoints require explicit conservative pricing; unknown pricing aborts before a request. Never put keys in files. To run a separately reported local study once a model is available:

```bash
python src/llm_study/run.py --backend ollama --model EXACT_MODEL_TAG --episodes 100
```

The protocol uses 100 held-out worlds with five paired policies and exact executable scoring. Preserve its real-model transcripts and model digest before rebuilding. The default reproduction remains mock-only and does not claim a model validation.

## Files and manuscript metadata
- `src/env.py`, `src/sim.py`: exact contracts, structured trace extractor and parametric task stream.
- `src/run_experiments.py`: original conditions, fresh confirmation, larger library and trace ablation.
- `results/`: raw seed-level and analysed CSVs; `results/llm/` separates mock output from any real-model experiment.
- `figures/`: publication-resolution charts with alternative text in the manuscript.
- `literature/verified_sources.md`, `notes_fulltext.md`: primary-record verification and closest-work positioning.
- `paper/author.json`: the central three-author metadata source, including correspondence and repository details.
- `paper/numeric_provenance.json`, `number_audit.json`: recomputable expressions, input hashes and numeric coverage.
- `paper/manuscript_audit.json`: citation, abstract, image and template evidence.
- `paper/supplementary_p1.zip`: independent code, template, requirements, seeds, results and figures with a SHA256 manifest.

After editing only `paper/author.json`, run `python build_manuscript.py` from the `paper/` directory and then `./docx2pdf.ps1` with the DOCX/PDF paths. Confirmed author metadata and the research repository are recorded in paper/author.json.

## Scientific boundaries
The tiered policy is exploratory, with fresh-seed confirmation. The added scale condition changes library utilisation together with size. The trace extractor operates on a known grammar and its measured recall does not transfer to opaque APIs. All reported uncertainty concerns simulated worlds. AgentMemoryBench and the related external benchmarks were not run. The conclusions do not establish safety or frequencies in real deployments.
'''
(root/'docs/RESEARCH.md').write_text(text,encoding='utf-8')
print('README regenerated from CSV results')
