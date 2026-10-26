"""Build P1 from executable results and the retained journal template."""
from pathlib import Path
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from build_docx import build
from manuscript_support import Evidence, author_meta, declarations
from sim import Params

RES, FIG = HERE.parent / "results", HERE.parent / "figures"
OUT = str(HERE / "P1_Selective_Invalidation_of_Procedural_Agent_Memory.docx")
datasets = {p.stem: pd.read_csv(p) for p in RES.glob("*.csv") if p.stem not in ("table4_sensitivity", "table6_magnitude")}
t = datasets["table1_main"].set_index("policy")
paired = datasets["table2_paired"]
P = Params()
N = int(datasets["e1_main"].seed.nunique())

def avg(dataset, condition, policy, metric="success"):
    d = datasets[dataset]
    return float(d[(d.cond == condition) & (d.policy == policy)][metric].mean())

def delta(a, b, metric, field="diff"):
    return float(paired[(paired.a == a) & (paired.b == b) & (paired.metric == metric)].iloc[0][field])

def trace(recorder):
    d = datasets["e9_trace_recall"]
    return float(d[d.recorder == recorder].recall.mean())

E = Evidence(HERE, dict(t=t, datasets=datasets, avg=avg, delta=delta, trace=trace, N=N, abs=abs))
V = E.value
LABELS = {"keep_all": "Keep all", "clear_all": "Clear all", "ttl": "Time expiry", "revalidate_all": "Revalidate all",
          "selective_tool": "Selective tool", "selective_clause": "Selective clause",
          "selective_clause_audit0.1": "Clause with random audit", "selective_clause_tiered0.5": "Clause with tiered probe"}
blocks = []
def h1(text): blocks.append(("h1", text))
def h2(text): blocks.append(("h2", text))
def p(text): blocks.append(("p", text))
def figure(key, filename, caption, alt, width=15):
    blocks.append(("figure", dict(key=key, path=str(FIG / filename), caption=caption, alt=alt, width_cm=width)))

ABSTRACT = (
    "Procedural memories can become stale when tool interfaces change, yet clearing a complete skill library discards valid experience. "
    "This simulation study evaluates maintenance decisions immediately after a known contract change. An executable environment distinguishes visible argument or "
    "precondition errors from silent unit errors and harmless optional additions. Eight policies are compared on "
    f"{V('N','d')} paired random worlds per condition, with controlled dependency-recording recall, probe sensitivity and relearning accuracy. "
    "A parametric agent represents execution and repair; its probabilities are assumptions rather than measurements of a language model. "
    f"Tool-level selection achieved success {V('t.loc[\"selective_tool\", \"success\"]')} versus {V('t.loc[\"revalidate_all\", \"success\"]')} for exhaustive revalidation, "
    f"while reducing probe effort by {V('100*(1-t.loc[\"selective_tool\",\"probe_units\"]/t.loc[\"revalidate_all\",\"probe_units\"])','.1f')}%. "
    f"Clause-level selection changed success by {V('100*delta(\"selective_clause\",\"revalidate_all\",\"success\")','.2f')} percentage points relative to exhaustive revalidation. "
    "Fresh-seed confirmation, a larger library and a simple trace extractor examine robustness and the operational meaning of these trade-offs. "
    "No policy dominates across assumptions: selective probing reduces maintenance effort, incomplete dependency records leave silent failures, and clearing can "
    "improve success when probes are unreliable. The results characterise a controlled maintenance problem and require validation with real model-driven agents."
)

h1("1. Introduction")
p("An agent that retains an executable procedure must decide whether yesterday's procedure remains valid today. Skill reuse can reduce repeated planning, "
  "as illustrated by Voyager, while reflective memory can preserve lessons from failed attempts (G. Wang et al., 2023; Shinn et al., 2023). "
  "Neither representation makes an old tool assumption permanently valid. Renaming an argument produces an obvious error; changing its unit can instead "
  "produce a plausible response with the wrong effect. Interface evolution is a documented software-maintenance problem (Brito et al., 2018).")
p("At a known change event, keeping every skill preserves both valid and stale procedures. Clearing every skill removes staleness but forces relearning. "
  "Revalidating every skill spends work on unaffected procedures. Selective maintenance records dependencies between skills and contract clauses, then probes "
  "only records touched by the diff. Its value depends on whether those dependencies are complete and whether a probe detects the relevant error.")
p("Dependency-directed revision has a long history in truth maintenance (Doyle, 1979). Recent memory systems also retain verification metadata or related "
  "memory units (H. Wang et al., 2026; Yang et al., 2026). Selective skill retention itself has already been tested by R. Wang et al. (2026). "
  "The contribution here is a controlled measurement at an externally specified tool-contract change, rather than a new general principle of selective memory.")
p("The study asks: RQ1, how do selective policies compare with retention, expiry, clearing and exhaustive revalidation in success and cost? "
  "RQ2, how do dependency granularity and completeness affect that comparison? RQ3, which error, detection, relearning and scale assumptions change the practical choice?")

h1("2. Literature Review")
h2("2.1. Procedural reuse and memory governance")
p("Voyager stores executable skills for an embodied environment; Reflexion stores verbal feedback to guide later attempts (G. Wang et al., 2023; Shinn et al., 2023). "
  "These establish distinct reusable objects, not a universal maintenance policy. MemGuard persists verifier-derived descriptors for later retrieval and retirement "
  "(H. Wang et al., 2026). Its full text discusses stale interface hints and dependence on imperfect verification. Our probe-sensitivity parameter "
  "isolates that dependence without reproducing its memory induction or language-model verifier.")
p("MACE preserves conditions, actions and outputs in linked functional memory units (Yang et al., 2026). Its dependency-retention and noisy-retrieval experiments "
  "overlap with our concern for incomplete metadata, but its graph composition and task-feedback loop differ from a single known contract diff. "
  "Its imported auxiliary baselines differ from its locally controlled comparisons; their headline scores are not treated as comparable evidence here.")
p("Anti-Persistence reconstructs temporary guidance from trajectories, while its selective-retention appendix tests filtering and a frozen skill layer "
  "(R. Wang et al., 2026). That appendix precludes a novelty claim for selective persistence alone. Our experiment measures invalidation coverage and "
  "maintenance cost under executable external changes; it does not model iterative abstraction descendants or reproduce error reproduction.")
h2("2.2. Executable evaluation and the remaining question")
p("Function-calling evaluation distinguishes syntactic invocation from stateful reasoning (Patil et al., 2025). Tau-bench evaluates final database state rather "
  "than relying on the agent's success claim (Yao et al., 2024). Ambiguous tool outcomes likewise distinguish model behaviour from harness and contract "
  "effects (Li, 2026). We use executable postconditions for the same methodological reason: a successful-looking return is insufficient for task success. "
  "AgentMemoryBench is a possible future transfer environment, not a benchmark run in this study (solomoon313, n.d.).")
p("The unresolved experimental question is how much maintenance work can be avoided while preserving executable validity, given incomplete dependency metadata. "
  "Our setting makes the diff known and the task evaluator exact. It leaves contract discovery, semantic parsing of natural-language changes and "
  "open-ended planning outside the evidence. This narrow boundary permits attribution of policy differences to explicit assumptions.")

h1("3. Materials and Methods")
h2("3.1. Executable environment and paired design")
p(f"Study 1 generates {N} worlds per condition. Each contains {P.n_types} task types and {P.n_tools} tools; a skill uses three to six tools. "
  f"The warm-started library is evaluated for {P.horizon} task steps, with one change at step {P.change_at} and {P.horizon-P.change_at} post-change tasks. "
  "A Zipf task-frequency distribution gives repeated exposure to popular skills. All policies receive the same world, task stream and execution draws. "
  "Policy-specific maintenance consumes a separate deterministic random stream; pairing is by world rather than by individual probe outcome.")
p("A contract declares accepted argument names, unit factors and prerequisite tools. A stored skill records the names and factors used when learned. "
  "Execution checks argument acceptance and prerequisite order, then compares interpreted values with intended values. Missing requested outputs are failures. "
  "Renames and unmet prerequisites are visible errors; changed units can silently alter final state. Optional additions are harmless by construction. "
  f"The default diff touches {P.n_changed_tools} tools, with equally weighted change types. Requested prerequisite changes on the lowest-index tool "
  "fall back to a rename; the prerequisite regime therefore includes that boundary case rather than a strictly pure intervention.")
h2("3.2. Dependency recording and maintenance policies")
p(f"The main recorder independently retains true clauses with probability {P.recall}. Tool-level selection flags a skill if any recorded clause refers "
  "to a changed tool; clause-level selection requires a clause intersection. This is a controlled missingness model, not a measured natural-language extractor. "
  "A separate exploratory extractor reads structured call traces, recording argument, unit and prerequisite-contract reads. An arguments-only ablation omits "
  "prerequisite reads. Recall is measured against the executable grammar's dependency set; it makes no claim about arbitrary program dependencies.")
blocks.append(("ul", [
    "Keep all retains the library; clear all deletes it; time expiry removes skills older than the configured window.",
    "Revalidate all probes every skill; selective tool and selective clause probe only flagged records.",
    "Clause with random audit also probes a random tenth of unflagged records; clause with tiered probe samples half of the records sharing a changed tool but lacking a changed clause.",
]))
p(f"The expiry window is {P.ttl} steps. Probes always detect visible errors and detect silent errors with probability {P.q_det}; each costs {P.c_probe} "
  "call-equivalents. A failed probe deletes the skill for later relearning. The tiered strategy was introduced after exploratory inspection, so its generality "
  "is assessed on a separate confirmation stream. No policy receives hidden task-success labels during task execution.")
h2("3.3. Agent assumptions and outcome accounting")
p(f"A valid stored skill succeeds with probability {P.p_exec}. Missing memory requires exploration, with success probability {P.p_explore} and a cost of "
  f"twice the requested tool count plus two calls, excluding any added prerequisite tools. Visible failure can trigger repair, costing {P.c_repair:g} calls with success probability {P.p_repair}. "
  "Silent failure is not noticed in the main condition. These are assumptions, not calibrated language-model measurements. The simulator records "
  "exact outcome labels and separately counts visible and silent stale executions.")
p("The primary outcomes are post-change success and mean call-equivalent cost. Cost equals execution, exploration, repair and maintenance cost divided by "
  "post-change tasks. Secondary outcomes are probe effort, stale executions, stale skills remaining after maintenance and healthy retention. "
  "Because cheap silent failure can look efficient, harm-weighted cost separately charges each silent failure; this is a preference sensitivity, "
  "not an estimated monetary loss. Actual tokens and latency are not represented by call-equivalents.")
h2("3.4. Statistical analysis and exploratory separation")
p(f"Main seeds are zero through {N-1}; fresh confirmation seeds start at 10000. Added scale and extractor conditions use seeds starting at 20000. "
  "Percentile bootstrap intervals use 2000 resamples across worlds (Efron, 1979). Paired mean differences use two-sided sign-flip tests with 3000 permutations. "
  "Holm correction covers the complete reported family of paired success, cost and probe comparisons (Holm, 1979), conservatively including the secondary probe outcome. "
  "Intervals are nominal rather than simultaneous. The family was inherited from exploratory development and is not a preregistration; confirmation and new "
  "sensitivities are reported separately, with no tuning on their outcomes.")
h2("3.5. Study 2 protocol and ethics")
p("A separate model-driven harness replaces parametric execution with a language-model-generated plan on a held-out world stream, comparing retention, clearing, "
  "exhaustive revalidation and both selective granularities. Prompts expose the current contract and available stored plan; exact executable effects judge "
  "success, including omitted or invalid steps. Model calls and maintenance probes remain separate because their units differ. "
  "Responses are cached by prompt, model, seed and decoding configuration; paid calls require conservative pricing and a persistent per-paper budget cap. "
  "Mock responses test only software plumbing and are excluded from scientific results. Only synthetic tasks are used, without human participants or personal data.")

h1("4. Results")
h2("4.1. Success and maintenance cost")
rows=[]
for policy,label in LABELS.items():
    def cell(field,fmt=".3f"): return V(f't.loc["{policy}","{field}"]',fmt)
    rows.append([label,cell("success")+" ["+cell("s_lo")+", "+cell("s_hi")+"]",
                 cell("cost",".2f")+" ["+cell("c_lo",".2f")+", "+cell("c_hi",".2f")+"]",cell("probe_units",".1f"),cell("silent_stale",".1f")])
blocks.append(("table",dict(key="main",caption="Main simulation comparison",header=["Policy","Success and interval","Cost and interval","Probe cost","Silent failures"],
    rows=rows,widths=[2600,2300,2100,900,1000],font_half_points=17,
    note="Nominal 95% bootstrap intervals across paired worlds. Probe cost is call-equivalent effort, not a probe count. Silent failures are counts per post-change world.")))
p(f"For RQ1, {{T:main}} shows success {V('t.loc[\"clear_all\",\"success\"]')} for clearing and {V('t.loc[\"keep_all\",\"success\"]')} for retention. "
  f"Tool selection versus exhaustive revalidation had a paired success difference {V('delta(\"selective_tool\",\"revalidate_all\",\"success\")','.4f')} "
  f"[{V('delta(\"selective_tool\",\"revalidate_all\",\"success\",\"lo\")','.4f')}, {V('delta(\"selective_tool\",\"revalidate_all\",\"success\",\"hi\")','.4f')}], "
  f"and cost difference {V('delta(\"selective_tool\",\"revalidate_all\",\"cost_per_task\")','.4f')} "
  f"[{V('delta(\"selective_tool\",\"revalidate_all\",\"cost_per_task\",\"lo\")','.4f')}, {V('delta(\"selective_tool\",\"revalidate_all\",\"cost_per_task\",\"hi\")','.4f')}]. "
  "A small estimated success difference is not proof of equivalence; no equivalence margin was specified.")
figure("pareto","fig1_pareto.png","Success and call-equivalent cost in the main condition.","Eight policies plotted by mean success and cost with bootstrap intervals",13)
h2("4.2. Paired effects and recorded-dependency quality")
rows=[]
for a,b,metric in [("selective_tool","revalidate_all","success"),("selective_tool","revalidate_all","cost_per_task"),
                   ("selective_clause","revalidate_all","success"),("selective_tool","clear_all","success"),("selective_tool","clear_all","cost_per_task")]:
    expr=f'delta("{a}","{b}","{metric}"'
    rows.append([LABELS[a]+" vs "+LABELS[b],"Success" if metric=="success" else "Cost",V(expr+")",".4f")+" ["+V(expr+',"lo")',".4f")+", "+V(expr+',"hi")',".4f")+"]",V(expr+',"p_holm")',".4f")])
blocks.append(("table",dict(key="paired",caption="Selected paired mean differences and Holm-adjusted tests",header=["Comparison","Outcome","Difference and interval","Adjusted p"],
    rows=rows,widths=[3500,1000,2700,1100],font_half_points=17,note="First policy minus second; tests belong to the complete family supplied in the supplementary results, not only these displayed comparisons.")))
p(f"For RQ2, clause-level success was {V('avg(\"e2_recall\",\"recall=1.0\",\"selective_clause\")')} with complete records and "
  f"{V('avg(\"e2_recall\",\"recall=0.5\",\"selective_clause\")')} with half the dependencies recorded. Corresponding tool-level success was "
  f"{V('avg(\"e2_recall\",\"recall=1.0\",\"selective_tool\")')} and {V('avg(\"e2_recall\",\"recall=0.5\",\"selective_tool\")')}. "
  "Redundant clauses referring to a changed tool help flag records even when individual clauses are missing. This robustness depends on simulated redundancy "
  "and should not be extrapolated to sparse real traces.")
figure("recall","fig2_recall.png","Dependency completeness changes the success and cost trade-off.","Success and cost curves across dependency-recording recall")
h2("4.3. Error type and assumption sensitivity")
p(f"For RQ3, pure silent unit changes produced success {V('avg(\"e4_type\",\"unit\",\"keep_all\")')} for retention, "
  f"{V('avg(\"e4_type\",\"unit\",\"revalidate_all\")')} for exhaustive revalidation and {V('avg(\"e4_type\",\"unit\",\"clear_all\")')} for clearing. "
  f"For harmless changes, retention achieved {V('avg(\"e4_type\",\"benign\",\"keep_all\")')} and clearing {V('avg(\"e4_type\",\"benign\",\"clear_all\")')}. "
  "A benign diff alone does not justify deletion; maintenance value depends on silent errors and unnecessary relearning.")
figure("types","fig3_change_type.png","Success and cost by executable contract-change type.","Five policies compared for optional additions, renames, prerequisites and unit changes")
p(f"With perfect silent-error detection, tool-level success was {V('avg(\"e5_sensitivity\",\"q_det=1.0\",\"selective_tool\")')}; with weak detection it was "
  f"{V('avg(\"e5_sensitivity\",\"q_det=0.3\",\"selective_tool\")')}. At exploration accuracy 0.5, clearing achieved "
  f"{V('avg(\"e5_sensitivity\",\"p_explore=0.5\",\"clear_all\")')} versus {V('avg(\"e5_sensitivity\",\"p_explore=0.5\",\"selective_tool\")')} for selection. "
  "These reversals make a universal ranking inappropriate. Probe-cost and feedback sweeps are supplied so alternative preferences can be examined without "
  "selecting a favourable weight after inspection.")
h2("4.4. Confirmation and exploratory scale and trace tests")
p(f"On fresh confirmation worlds, success was {V('avg(\"e7_confirm\",\"main\",\"clear_all\")')} for clearing, "
  f"{V('avg(\"e7_confirm\",\"main\",\"revalidate_all\")')} for exhaustive probing, {V('avg(\"e7_confirm\",\"main\",\"selective_tool\")')} for tool selection and "
  f"{V('avg(\"e7_confirm\",\"main\",\"selective_clause_tiered0.5\")')} for the exploratory tiered strategy. This confirms only the specified conditions "
  "and does not convert exploratory policy design into preregistered evidence.")
p(f"The added library contains 100 skills and 30 tools. Selection achieved {V('avg(\"e8_scale_trace\",\"scale_100_30\",\"selective_tool\")')} success "
  f"at cost {V('avg(\"e8_scale_trace\",\"scale_100_30\",\"selective_tool\",\"cost_per_task\")','.2f')}; exhaustive revalidation achieved "
  f"{V('avg(\"e8_scale_trace\",\"scale_100_30\",\"revalidate_all\")')} at cost {V('avg(\"e8_scale_trace\",\"scale_100_30\",\"revalidate_all\",\"cost_per_task\")','.2f')}. "
  "The horizon is unchanged, so this jointly changes size and utilisation rather than isolating an asymptotic scaling effect.")
p(f"The full-trace extractor had measured clause recall {V('trace(\"trace_full\")')}; omitting prerequisite reads reduced it to {V('trace(\"trace_arguments\")')}. "
  f"Clause-level success was {V('avg(\"e8_scale_trace\",\"trace_full\",\"selective_clause\")')} and {V('avg(\"e8_scale_trace\",\"trace_arguments\",\"selective_clause\")')}, "
  "respectively. Omitting prerequisite reads need not reduce success here because visible repair and relearning have different assumed accuracies. "
  "Exact grammar knowledge favours the complete extractor; its recall is not an estimate for natural-language or opaque external tools.")
h2("4.5. Study 2 execution status")
status_path=RES/"llm"/"status.json"
status=json.loads(status_path.read_text()) if status_path.exists() else {"real_llm_run":False}
if status.get("real_llm_run"):
    llm=pd.read_csv(RES/"llm"/"summary.csv").set_index("policy")
    E.context["llm"]=llm
    p("Separate model-driven validation was completed with "+status["model"]+". Model metadata and raw transcripts are supplied; these synthetic tasks assess only this model and prompt protocol.")
    rows=[[LABELS[q],V(f'llm.loc["{q}","success"]'),V(f'llm.loc["{q}","lo"]')+" to "+V(f'llm.loc["{q}","hi"]')] for q in llm.index]
    blocks.append(("table",dict(key="llm",caption="Separate model-driven validation",header=["Policy","Success","Interval"],rows=rows,widths=[3500,2200,2200])))
else:
    p("No real language-model result is reported. Provider discovery found no API credentials or ready local model. The local-runtime acquisition attempt "
      "did not provide an executable model during this study. The held-out harness completed its mock dry run, which checks parsing, caching and exact scoring "
      "but provides no estimate of model success. No confidence interval or model comparison is inferred from mock output.")

h1("5. Discussion")
p("Selective maintenance buys avoided probes; it does not guarantee a better task outcome. A sound decision must consider both affected-skill coverage "
  "and the probe's ability to detect semantic errors. Tool-level redundancy can rescue missing clauses in this grammar, while clause precision saves work "
  "by refusing to examine harmless overlaps. Its benefit and risk arise from the same filter. A lower cost can conceal undetected incorrect executions "
  "rather than reflect a more effective agent.")
p("Clearing is a strong comparator because it removes silent staleness before any probe is trusted. It also discards valid skills and pays exploration costs. "
  "Whether this is worthwhile depends on library utilisation, recovery accuracy and the cost of a wrong effect. The larger-library sensitivity changes "
  "utilisation together with size; it cannot establish a general scaling law. An implementation should estimate its own workload and verifier errors "
  "before treating this simulation as a deployment decision.")
p("The results complement memory governance. MemGuard motivates retaining verification evidence; MACE preserves dependencies; Anti-Persistence shows that "
  "persistent abstractions and filters behave differently across contexts (H. Wang et al., 2026; Yang et al., 2026; R. Wang et al., 2026). Our imposed diff "
  "gives exact attribution of stale records but removes the difficult task of identifying whether an observed failure reflects a changed contract. "
  "No experiment here ranks those complete systems.")
p("Study 2 is a transfer test. Without real completions, the results support only the specified parametric regimes. A next test should freeze prompts and "
  "executable tasks, report invalid output and repair costs, and keep inference, probes and wall-clock latency separate. Model-driven evidence could "
  "reverse the policy ranking and would require revising the practical conclusions.")
h1("6. Conclusion")
p("RQ1 is answered by a conditional success-cost trade-off: selective probing reduces maintenance effort without uniformly improving success. RQ2 is answered "
  "by dependence on metadata completeness and redundant tool-level coverage. RQ3 is answered by detection, exploration, change-type and scale sensitivities: "
  "the preferred policy changes with those assumptions. No policy dominates across the studied regimes. The contribution is a reproducible comparison "
  "at an external contract-change event, with the limits of that comparison made explicit.")
h2("6.1. Recommendations")
p("Record tool use and explicit contract reads before attempting fine-grained invalidation. Measure recorder omissions and semantic-probe sensitivity on "
  "known changes. When a clause filter is incomplete, examine tool-sharing records or require independent postconditions. Compare maintenance savings "
  "with relearning and wrong-effect costs under the actual workload, retaining clearing and exhaustive revalidation as baselines.")
h2("6.2. Limitations")
p("The environment has one fully known change, fixed semantics and a warm-started library. Parametric execution, exploration and repair are uncalibrated; "
  "call-equivalents account for resources rather than measured inference cost. Independent missing clauses and exact structured traces omit correlated "
  "semantic misunderstandings. The tiered strategy followed exploratory inspection; fresh seeds limit reuse of that outcome stream but cannot remove "
  "researcher choices. Added scale and trace tests are exploratory, and the unchanged horizon confounds size with utilisation. No real model validation "
  "was completed, so planning, prompt sensitivity and language-mediated errors remain unmeasured. Repeated changes, shared subplans and production data "
  "may yield different trade-offs. Bootstrap uncertainty concerns simulation worlds and does not establish deployment frequencies.")

spec=dict(meta=author_meta(HERE,"Selective Invalidation of Procedural Agent Memory under Tool Contract Changes in a Controlled Simulation"),abstract=ABSTRACT,
    keywords=["procedural memory","tool contracts","selective invalidation","agent reliability","dependency recording","simulation"],
    blocks=blocks,declarations=declarations(HERE,"P1"),references=json.loads((HERE/"references.json").read_text(encoding="utf-8")))
if __name__=="__main__":
    E.save(spec)
    print(build(spec,OUT))
    print("Abstract words:",len(ABSTRACT.split()))
