# Phase 0 Real Data Acquisition Report: SEP-28k

**Date:** 2026-09-10 17:37:00 UTC  
**Source Repository:** `https://github.com/apple/ml-stuttering-events-dataset`  

---  

## 1. Episode Retrieval & Status Breakdown

| Metric | Value |
| :--- | :--- |
| **Total Podcast Episodes Attempted** | 385 |
| **Episodes Successfully Verified & Active** | 242 |
| **Episode Retrieval Success Rate** | **62.86%** |
| **Total Real 3-Second .wav Clips Extracted** | **19,463** |
| **Total Audio Disk Footprint (`clips/`)** | **1782.71 MB** |

### HTTP & Quarantine Status Breakdown

| Status Category | Count |
| :--- | :--- |
| `SUCCESS` | 242 |
| `FAILED_HTTP_404` | 122 |
| `QUARANTINED_TOO_SHORT` | 16 |
| `FAILED_CONNECTION_ERROR` | 5 |

---  

## 2. Surviving Shows Retrieval Table

| Show Name | Nominal Labels | Extracted Clips on Disk (`clips/`) | Retrieval Status |
| :--- | :--- | :--- | :--- |
| **HVSA** | 736 | 734 | Active |
| **HeStutters** | 3,684 | 2,716 | Active |
| **IStutterSoWhat** | 870 | 0 | Dead URLs |
| **MyStutteringLife** | 2,339 | 1,797 | Active |
| **StrongVoices** | 2,308 | 0 | Dead URLs |
| **StutterTalk** | 5,064 | 5,063 | Active |
| **StutteringIsCool** | 4,013 | 0 | Dead URLs |
| **WomenWhoStutter** | 9,163 | 9,153 | Active |
