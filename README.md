# Dysfluency-Aware Endpointing for Streaming Voice Agents

Research project investigating whether conditioning an endpointer on dysfluency evidence reduces premature cutoffs for dysfluent speech without paying an unacceptable latency cost on fluent speech.

---

## Strict Rules

1. **Never fabricate data.** All dataset sizes and retrieval statistics reflect real downloads.
2. **Never hardcode a result.** Every number is dynamically loaded from JSON run artifacts (`runs/<run_id>/`).
3. **No proxy metrics.** Uses standard metric libraries (`scipy.stats`, `pystoi`, `torchmetrics`).
4. **Phase-Gated Execution**:
   - **Phase 0 — Data**: Acquire and verify SEP-28k, AMI, LibriStutter, report retrieval success rates and speaker/show disjointness.
   - **Phase 1 — Task Construction**: Utterance-level segment building with forced alignment end-of-speech timestamps. 50-segment manual validation gate.
   - **Phase 2 — Baselines**: Run 5 baseline endpointers, produce cutoff-vs-latency curves and disparity numbers.
   - **Phase 3 — Method**: Train learned causal endpointer with dysfluency auxiliary head.
   - **Phase 4 — Ablations**: Input features, lookahead, auxiliary head, training data.
   - **Phase 5 — Statistics**: Cluster bootstrap over speakers (1000 replicates, 95% CIs) and paired bootstrap.

---

## Directory Structure

```
icassp/
├── configs/               # Experiment configuration JSONs
├── src/
│   ├── utils/             # JSON logger & artifact recorder
│   ├── metrics/           # Metric implementations (cutoff rate, latency, disparity)
│   └── data/              # Dataset parsers & speaker-disjoint split builders
├── scripts/               # Phase entry point scripts writing to runs/
├── runs/                  # Immutable JSON run artifacts
├── results/               # Dynamic tables & figure plots
├── tests/                 # Hand-computed known-answer metric correctness tests
└── RESULTS.md             # Dynamically generated paper results summary
```

---

## Quickstart & Verification

```bash
# Run known-answer metric unit tests
python3 -m unittest discover tests

# Execute Phase 0 Data Acquisition & Verification
python3 scripts/phase0_data_acquisition.py
```
