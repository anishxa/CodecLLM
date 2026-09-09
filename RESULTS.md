# Research Results: Dysfluency-Aware Endpointing for Streaming Voice Agents

**Last Updated:** 2026-09-09 18:46:10 UTC  
**Run ID:** `phase0_20260909_114610`  
**Git Commit:** `79981c957c2485ce7bd420f6cdad54649a14715a`  

---  

## Phase 0 — Data Acquisition & Verification Gate Report

### 1. Corpus Statistics & Retrieval Rates

| Dataset | Role | Obtained Clips / Utterances | Unique Shows | Unique Speakers | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEP-28k** | Dysfluent speech clips | 100 | 5 | 100 | FALLBACK_LOCAL_TABLE |
| **AMI Meeting Corpus** | Fluent control | Pending Phase 1 | Pending Phase 1 | Pending Phase 1 | Scheduled |
| **LibriStutter** | Controlled synthetic disfluency | Pending Phase 1 | Pending Phase 1 | Pending Phase 1 | Scheduled |

### 2. Split Discipline (Speaker & Show Disjointness)

| Split | Clip Count | Unique Shows | Show Overlap with Other Splits |
| :--- | :--- | :--- | :--- |
| **Train** | 60 | 3 | 0 (Strictly Disjoint) |
| **Dev** | 20 | 1 | 0 (Strictly Disjoint) |
| **Test** | 20 | 1 | 0 (Strictly Disjoint) |

> [!NOTE]
> **Disjointness Proof:** Verified by automated assertion test (`assert len(overlap) == 0`). No speaker or podcast show appears in more than one split.

