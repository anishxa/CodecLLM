# Phase 0 Real Data Acquisition Report: SEP-28k

**Date:** 2026-09-10 04:31:43 UTC  
**Source Repository:** `https://github.com/apple/ml-stuttering-events-dataset`  

---  

## 1. Episode Retrieval & Clip Statistics

| Metric | Value |
| :--- | :--- |
| **Total Podcast Episodes Attempted** | 385 |
| **Episodes Successfully Retrieved** | 86 |
| **Dead / Failed Episode URLs** | 299 |
| **Episode Retrieval Success Rate** | **22.34%** |
| **Total 3-Second .wav Clips Extracted** | **7,819** |
| **Total Audio Disk Footprint (`clips/`)** | **716.11 MB** |

---  

## 2. Official Show Breakdown Table

| Show Name | Total Nominal Clips | Extracted Clips | Status |
| :--- | :--- | :--- | :--- |
| **HVSA** | 736 | 736 | Active |
| **HeStutters** | 3,684 | 3,684 | Active |
| **IStutterSoWhat** | 870 | 0 | Dead URLs |
| **MyStutteringLife** | 2,339 | 2,259 | Active |
| **StrongVoices** | 2,308 | 0 | Dead URLs |
| **StutterTalk** | 5,064 | 1,060 | Active |
| **StutteringIsCool** | 4,013 | 0 | Dead URLs |
| **WomenWhoStutter** | 9,163 | 80 | Active |

---  

## 3. Split Discipline & Speaker Identity Verification

- **Clustering Unit**: Podcast `Show` (Leave-One-Show-Out across the 8 official shows).
- **Speaker Identity**: Episode ID (`Show_EpId`). Every episode represents a distinct speaker session.
- **Disjointness Assertion**: Proved 0 overlap across Train / Dev / Test sets.

> [!IMPORTANT]
> **Human Inspection Required**: Phase 0 deliverable consists of real `.wav` audio files inside `clips/`. Play a clip to verify audio quality before proceeding to Phase 1.
