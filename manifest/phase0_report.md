# Phase 0 Real Data Acquisition Report: SEP-28k

**Date:** 2026-09-10 16:42:39 UTC  
**Source Repository:** `https://github.com/apple/ml-stuttering-events-dataset`  

---  

## 1. Episode Retrieval & HTTP Breakdown

| Metric | Value |
| :--- | :--- |
| **Total Podcast Episodes Attempted** | 385 |
| **Episodes Successfully Retrieved & Verified** | 242 |
| **Episode Retrieval Success Rate** | **62.86%** |
| **Total Real 3-Second .wav Clips Extracted** | **20,605** |
| **Dropped Non-3.0s Clips** | 8 |
| **Total Audio Disk Footprint (`clips/`)** | **1887.24 MB** |

### HTTP Retrieval Status Breakdown

| Status Code / Category | Count |
| :--- | :--- |
| `SUCCESS` | 242 |
| `FAILED_HTTP_404` | 122 |
| `FAILED_CONNECTION_ERROR` | 5 |
| `QUARANTINED_TOO_SHORT (samples 37845504 < 37850240)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 52372701 < 52374880)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 15289619 < 15296640)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 53323879 < 53328160)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 27262937 < 76639360)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 39845815 < 56707520)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 29491161 < 40983360)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 36437956 < 63992000)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 16908249 < 61730080)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 36569006 < 58559840)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 21888985 < 22105760)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 3801049 < 6921280)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 80605936 < 80608640)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 37964531 < 37970400)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 45319760 < 45326400)` | 1 |
| `QUARANTINED_TOO_SHORT (samples 43244928 < 43251680)` | 1 |

---  

## 2. Surviving Shows Retrieval Table

| Show Name | Nominal Labels | Extracted Clips on Disk (`clips/`) | Retrieval Status |
| :--- | :--- | :--- | :--- |
| **HVSA** | 736 | 736 | Active |
| **HeStutters** | 3,684 | 3,684 | Active |
| **IStutterSoWhat** | 870 | 0 | Dead URLs |
| **MyStutteringLife** | 2,339 | 2,259 | Active |
| **StrongVoices** | 2,308 | 0 | Dead URLs |
| **StutterTalk** | 5,064 | 5,064 | Active |
| **StutteringIsCool** | 4,013 | 0 | Dead URLs |
| **WomenWhoStutter** | 9,163 | 8,862 | Active |
