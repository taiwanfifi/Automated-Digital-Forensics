## Digital Forensics AI: CCTV Person Retrieval System

This project is an end-to-end digital forensics prototype that simulates a realistic scenario where **prosecutors or lawyers need to locate a suspect or witness from hours of CCTV footage**.

The system turns **hundreds of hours of manual video review** into a **few seconds of semantic search**, e.g.:

> "Find the person wearing yellow shorts and red sandals."

---

### System Overview

- **`main.py`: visual perception and evidence logging**
  - Uses **YOLO + ByteTrack + supervision** for person detection and multi-object tracking (`tracker_id`).
  - Defines **polygon zones** (e.g. checkout counter) and computes each person's **dwell time** inside each zone.
  - For individuals whose dwell time exceeds a threshold:
    - Dynamically selects the **best frame** (largest bbox with padding),
    - Calls a **local Gemma3 VLM (via Ollama REST API)** to generate a **short structured English description** (gender, age group, clothing colors, role guess),
    - Simultaneously writes a **cropped person-only video clip** for each `tracker_id` × `zone` (fixed size `400x400` AVI).
  - When a person exits a zone, the system appends a JSON record to `people_zone_log.jsonl` containing:
    - `tracker_id`, `zone_index`, `entered_at`, `exited_at`, `dwell_time_seconds`
    - `person_description` (from Gemma3)
    - `video_file` (path to the cropped clip)

- **`RAG_retrieve/RAG_retrieve.py`: RAG-based retrieval and legal-style answers**
  - Loads `people_zone_log.jsonl` into `Document` objects:
    - `page_content = person_description`
    - `metadata = tracker_id / zone / time / video_file / global_timestamp`.
  - Uses **Ollama `nomic-embed-text:v1.5`** for text embeddings and stores them in **Milvus Lite** (local vector database, single file `milvus_people_log.db`).
  - Builds a **RAG retrieval chain**:
    - Retrieves Top-K similar descriptions and formats them as an "evidence list".
    - Feeds this context into a local **`llama3.1:8b`** model, prompting it to act as a **digital forensics expert / legal investigator**, and to output:
      - Which tracked persons are most likely matches.
      - For each, the `tracker_id`, `video_file`, entry/exit times, and key appearance features.
      - An explanation of why they match the natural-language query.
  - Supports interactive queries such as:
    - "Find the person wearing yellow shorts and red sandals"
    - "Find a male in white shirt and black trousers who looks like an office worker"

---

### Technical Highlights (for reviewers and hiring managers)

- **1. Multi-modal pipeline: from raw video to semantic vectors**
  - **Perception layer**: YOLO + ByteTrack + supervision for multi-object tracking, ID consistency across frames, and zone entry/exit events.
  - **Vision-language layer**: local Gemma3 VLM over HTTP, converting cropped person images into structured natural language descriptions.
  - **Vector search layer**: `nomic-embed-text` embeddings stored in Milvus, enabling high-dimensional semantic nearest-neighbor search.
  - **Language reasoning layer**: `llama3.1` aggregates multiple retrieved records and generates a legal-style analysis report.

- **2. Streaming processing and near real-time design**
  - Video is processed frame-by-frame; the main loop must remain responsive while handling:
    - detection, tracking, zone checks, overlay rendering, and video writing.
  - Gemma3 calls are **delayed and sparsified**:
    - Only triggered once per person/zone when dwell time passes a threshold *and* a good "best frame" is available,
    - Avoids calling the VLM on every frame, which would be too slow and expensive.
  - I/O design:
    - Cropped clips (`cv2.VideoWriter`) and JSONL logging are both driven by **zone exit events**,
    - Prevents generating unnecessary files/records for transient passers-by.

- **3. Data model and chain-of-custody friendly design**
  - A central `person_memory` structure tracks, per `tracker_id` and per zone:
    - entry time, last seen time, dwell seconds, best frame, description, and video writer.
  - Each JSONL record preserves:
    - `global_timestamp`, entry/exit times, dwell time, zone index,
    - and the exact **video file path**: `tracker_clips/tracker_{id}_zone_{idx}.avi`.
  - During RAG, these fields are mapped into metadata so the LLM can reference **concrete evidence** (file names and timestamps), which is crucial for legal workflows.

- **4. Practical engineering and robustness**
  - Logging uses **append-only JSONL**, avoiding large monolithic writes during long recording sessions.
  - All VLM calls and file writes are wrapped in `try/except` blocks:
    - A VLM failure only means the person might lack a description; the main pipeline keeps running.
    - A single failed write only affects that record, not the whole system.
  - Milvus metadata types are normalized (e.g. `video_file` always cast to string), ensuring the vector index can be reliably recreated.

---

### How to Run

- **1. Generate `people_zone_log.jsonl` and cropped person clips**

```bash
python main.py \
  --zone_configuration_path path/to/zones.json \
  --source_video_path path/to/source_video.mp4 \
  --weights yolov8s.pt \
  --device cpu
```

- **2. Start the RAG retrieval service (local Ollama + Milvus)**

```bash
cd RAG_retrieve
python RAG_retrieve.py
```

After startup, the script will:
- Load `../people_zone_log.jsonl` and (re)build the Milvus vector index.
- Run a couple of demo queries, then enter an interactive loop where you can type natural-language descriptions of suspects/witnesses and get back matching clips and timestamps.

---

### Localization

- This `README.md` is the **primary English documentation** for international recruiters.
- A Traditional Chinese version is available in **`README-zh.md`** for local readers (Taiwan / Hong Kong).

---

### Author

- **Name**: William
- **Email**: taiwanfifi@gmail.com
