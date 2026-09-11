# Cross-Sensor Lunar Location Matching API - Integration Guide

This document describes the FastAPI REST API layer built around the **Lunar Location Matching Engine**.
It acts as the clean integration boundary between external application backends (or web frontends) and the model-side matching pipeline (`HybridMatcher` + `RegistrationEngine`).

---

## 1. Architecture Overview

```
  External Backend / Web Client
               |
               | HTTP (POST /match, GET /match/{job_id})
               | SSE  (GET /match/{job_id}/events)
               v
     FastAPI Service Layer (`api/`)
               |
               | In-Process Background Thread (Single Lock)
               v
     Lunar Matching Engine (`src/`)
      ├── HybridMatcher (Crater Constellations + LoFTR Tiled Deep Matching)
      └── RegistrationEngine (Tie-point selection, refit, quality evaluation)
```

- **Non-blocking Execution**: `POST /match` decodes input images, generates a unique `job_id`, and immediately returns HTTP 202 Accepted.
- **Single-Concurrency Worker Lock**: Only **1 heavy matching job executes at a time** to guarantee CPU/memory safety during CPU-bound LoFTR inference on large images.
- **Live Progress Reporting**: Clients monitor real-time progress via Server-Sent Events (SSE) or polling.
- **Model Independence**: The underlying Python engine in `src/` remains completely independent of FastAPI and can still be used directly in Python scripts.

---

## 2. API Endpoints

### 2.1 Health Check
**Endpoint**: `GET /health`  
**Description**: Verifies service status and model initialization.

**Sample Response**:
```json
{
  "status": "ok",
  "service": "lunar-image-matching-api",
  "model_available": true
}
```

---

### 2.2 Submit Image Pair for Matching
**Endpoint**: `POST /match`  
**Content-Type**: `multipart/form-data`  
**Parameters**:
- `image_a`: UploadFile (Reference lunar image)
- `image_b`: UploadFile (Query lunar image to align)

**Sample Response (HTTP 202 Accepted)**:
```json
{
  "job_id": "c9b8a1e2-4d3f-4e5a-8b9c-1d2e3f4a5b6c",
  "status": "queued"
}
```

---

### 2.3 Poll Job Status and Results
**Endpoint**: `GET /match/{job_id}`  
**Description**: Retrieves current job status (`queued`, `running`, `completed`, `failed`), latest progress event, and registration results.

**Sample Completed Response**:
```json
{
  "job_id": "c9b8a1e2-4d3f-4e5a-8b9c-1d2e3f4a5b6c",
  "status": "completed",
  "progress": {
    "stage": "completed",
    "current": 1,
    "total": 1,
    "progress": 1.0,
    "message": "Matching and registration complete"
  },
  "result": {
    "matched": true,
    "correspondences": 142,
    "inliers": 88,
    "inlier_ratio": 0.6197,
    "confidence": 0.925,
    "rmse": 0.412,
    "mean_error": 0.354,
    "median_error": 0.312,
    "max_error": 0.985,
    "coverage": 0.845,
    "quality": "EXCELLENT",
    "transform": {
      "scale": 1.002,
      "rotation_deg": 12.45,
      "translation": {
        "x": 45.2,
        "y": -18.7
      }
    },
    "failure_reason": null
  },
  "error": null
}
```

> **Note on Failure Semantics**: If geometric matching completes but fails to find valid alignment (`matched: false`), the HTTP job status remains `"completed"`, returning `matched: false` and a descriptive `failure_reason`. It is NOT an HTTP 500 error.

---

### 2.4 Live Progress Streaming (SSE)
**Endpoint**: `GET /match/{job_id}/events`  
**Header**: `Accept: text/event-stream`  
**Description**: Server-Sent Events (SSE) stream delivering real-time progress events as matching progresses through stages.

**Wire Format**:
```http
event: progress
data: {"stage": "loading", "current": 0, "total": 1, "progress": 0.0, "message": "Loading input images"}

event: progress
data: {"stage": "learned_matching", "current": 1, "total": 4, "progress": 0.25, "message": "Processed LoFTR tile pair 1/4"}

event: progress
data: {"stage": "learned_matching", "current": 2, "total": 4, "progress": 0.50, "message": "Processed LoFTR tile pair 2/4"}

event: progress
data: {"stage": "geometric_verification", "current": 1, "total": 1, "progress": 1.0, "message": "Executing RANSAC geometric verification"}

event: completed
data: {"matched": true, "correspondences": 142, "inliers": 88, ...}
```

---

## 3. Progress Pipeline Stages

| Stage Name | Description |
| :--- | :--- |
| `loading` | Initial image decoding and memory allocation |
| `crater_detection` | Detecting impact crater keypoints in Image A & B |
| `learned_matching` | LoFTR deep feature tile matching (reports `current`/`total` completed tile pairs) |
| `correspondence_fusion` | Fusing crater and learned correspondences |
| `geometric_verification` | Running RANSAC spatial transformation estimation |
| `registration` | Tie-point selection, sub-pixel refit, and image warping |
| `quality_assessment` | Computing reprojection RMSE, grid coverage, and confidence |
| `completed` | Task finished successfully |
| `failed` | Task terminated due to error |

---

## 4. Integration Code Examples

### 4.1 Python Client Example (`requests` + `httpx` for SSE)

```python
import time
import requests
import httpx

API_BASE = "http://127.0.0.1:8000"

# 1. Submit job
files = {
    "image_a": ("reference.png", open("data/ref.png", "rb"), "image/png"),
    "image_b": ("query.png", open("data/query.png", "rb"), "image/png"),
}
res = requests.post(f"{API_BASE}/match", files=files)
res.raise_for_status()
job_id = res.json()["job_id"]
print(f"Submitted job: {job_id}")

# 2. Listen to SSE live progress stream
with httpx.stream("GET", f"{API_BASE}/match/{job_id}/events") as stream:
    for line in stream.iter_lines():
        if line.startswith("data: "):
            print("SSE Event:", line[6:])

# 3. Fetch final result
status_res = requests.get(f"{API_BASE}/match/{job_id}").json()
print("Final Registration Result:", status_res["result"])
```

### 4.2 JavaScript / Web Frontend Example (`EventSource`)

```javascript
// 1. Submit job via FormData
const formData = new FormData();
formData.append("image_a", imageAFile);
formData.append("image_b", imageBFile);

const response = await fetch("http://127.0.0.1:8000/match", {
  method: "POST",
  body: formData,
});
const { job_id } = await response.json();

// 2. Connect EventSource for live UI updates
const eventSource = new EventSource(`http://127.0.0.1:8000/match/${job_id}/events`);

eventSource.addEventListener("progress", (e) => {
  const data = JSON.parse(e.data);
  console.log(`[${data.stage}] ${data.message} (${(data.progress * 100).toFixed(1)}%)`);
  updateProgressBar(data.progress, data.message);
});

eventSource.addEventListener("completed", (e) => {
  const result = JSON.parse(e.data);
  console.log("Registration Complete:", result);
  eventSource.close();
  displayTransformResult(result);
});

eventSource.addEventListener("failed", (e) => {
  const error = JSON.parse(e.data);
  console.error("Job Failed:", error);
  eventSource.close();
});
```

### 4.3 cURL Commands

```bash
# Health Check
curl http://127.0.0.1:8000/health

# Submit Match Job
curl -X POST "http://127.0.0.1:8000/match" \
  -F "image_a=@data/processed/ref.png" \
  -F "image_b=@data/processed/query.png"

# Poll Job Status
curl http://127.0.0.1:8000/match/{job_id}

# Stream SSE Events
curl -N http://127.0.0.1:8000/match/{job_id}/events
```
