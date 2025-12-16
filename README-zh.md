## 數位鑑識 AI：監視器人物檢索系統

一個端到端的數位鑑識原型系統，模擬「檢察官 / 律師為了找嫌疑犯或證人，需要從大量 CCTV 畫面中找特定人物」的真實場景。

系統將原本需要人工肉眼查看上百小時監視器的工作，縮短為數秒內的語意搜尋：「幫我找穿黃褲子、紅色拖鞋的人」。

---

### 專案架構與流程

- **`main.py`：視覺感知與證據寫入**
  - 使用 **YOLO + ByteTrack + supervision** 做人形偵測與多目標追蹤（`tracker_id`）。
  - 以多邊形區域 (polygon zones) 定義「關鍵區域」（例如收銀台），計算每個人於各 zone 的 **停留時間 (dwell time)**。
  - 對停留時間超過門檻的個體：
    - 動態選擇「最佳畫面」（最大 bbox，含 padding），
    - 呼叫 **本地 Gemma3 VLM（透過 Ollama REST API）**，將單人畫面轉成 **英文結構化文字描述**（性別、年齡層、服裝顏色、角色猜測）。
    - 同時建立每個 `tracker_id` × `zone` 的 **裁切個人影片剪輯**（固定尺寸 `400x400` 的 AVI 檔）。
  - 於人物離開 zone 時，將以下資訊以 **JSONL 追加寫入 `people_zone_log.jsonl`**：
    - `tracker_id`, `zone_index`, `entered_at`, `exited_at`, `dwell_time_seconds`
    - `person_description`（由 Gemma3 產生）
    - `video_file`（對應的人物剪輯檔路徑）

- **`RAG_retrieve/RAG_retrieve.py`：RAG 檢索與法律語境回答**
  - 將 `people_zone_log.jsonl` 讀入為 `Document`：
    - `page_content = person_description`
    - `metadata = tracker_id / zone / time / video_file / global_timestamp`。
  - 使用 **Ollama 的 `nomic-embed-text:v1.5`** 產生向量，寫入 **Milvus Lite**（本地向量資料庫，單檔 `milvus_people_log.db`）。
  - 建立 **RAG 檢索鏈**：
    - 檢索 Top-K 最相似的描述，格式化為「證據清單」。
    - 交由 **本地 `llama3.1:8b`** 以「數位鑑識專家 / 法律調查助理」的語氣，輸出：
      - 最可疑的目標有哪些？
      - 各自的 `tracker_id`、`video_file`、進出時間與關鍵外觀特徵。
      - 為何判斷這些人符合「查詢描述」。
  - 支援互動式查詢，例如：
    - 「幫我找穿黃褲子、紅色拖鞋的人」
    - 「找穿白色上衣、黑色長褲，看起來像上班族的男性」。

---

### 技術重點與難點說明（給技術審查者 / 招募方看）

- **1. 多模態串接：從影像到語意向量的完整鏈路**
  - 影像層：YOLO + ByteTrack + supervision 管理多目標、跨幀 ID 一致性與 zone 進出事件。
  - 視覺語言模型層：本地 Gemma3 VLM 透過 HTTP API，將裁切畫面轉成具結構的自然語言描述。
  - 向量檢索層：使用 `nomic-embed-text` 將文字描述嵌入到 Milvus，支援高維語意近似搜尋。
  - 語言推理層：llama3.1 針對多筆檢索結果做「法律語境」的綜合判斷與報告撰寫。

- **2. 串流處理與接近即時的非同步設計思維**
  - 影片是逐幀處理，主迴圈需維持即時性：
    - 物件偵測、追蹤、zone 計算、畫面標註、影片寫檔全部在同一條流程上執行。
  - 對於 Gemma3 調用：
    - 採「**延後且挑一次最有代表性的畫面**」策略（只在 dwell time 達門檻時、且找到最佳 bbox 才呼叫 API），
    - 避免每幀都打 VLM API 造成延遲與資源浪費。
  - I/O 設計：
    - 影片寫入 (`cv2.VideoWriter`) 與 JSONL 追加寫檔都放在「事件觸發點」（離開 zone）處理，
    - 確保大量路過行人不會產生多餘檔案或記錄。

- **3. 資料模型設計與可追溯性（Chain of Custody）**
  - `person_memory` 統一管理一個人跨 zone 的狀態（進入時間、最後一次看到的時間、停留秒數、最佳畫面、描述、影片 writer）。
  - JSONL 記錄中明確保留：
    - 原始時間戳 (`global_timestamp`)、進出時間、停留秒數、zone index。
    - 對應影片檔路徑：`tracker_clips/tracker_{id}_zone_{idx}.avi`。
  - 在 RAG 階段，這些欄位被映射到 metadata，讓 LLM 回答時可以直接引用具體證據（影片檔名與時間），方便法律工作者進一步調閱原始影片。

- **4. 工程實務考量與穩定性**
  - JSONL 採用 **追加寫入** 模式，避免長時間錄製時一次性大檔寫入風險。
  - 對 VLM / 檔案寫入全部有 try/except 包覆：
    - VLM 失敗時不會中斷主流程，只是該行人可能缺少文字描述。
    - 檔案寫入失敗只影響單筆紀錄，不會讓整個系統崩潰。
  - Milvus metadata 型別嚴格處理（例如 `video_file` 一律轉字串），確保向量庫可以穩定重建。

---

### 如何執行

- **1. 產生 people_zone_log 與剪輯影片**

```bash
python main.py \
  --zone_configuration_path path/to/zones.json \
  --source_video_path path/to/source_video.mp4 \
  --weights yolov8s.pt \
  --device cpu
```

- **2. 啟動 RAG 檢索（本地 Ollama 與 Milvus）**

```bash
cd RAG_retrieve
python RAG_retrieve.py
```

執行後會：
- 讀取 `../people_zone_log.jsonl` 建立 / 重建 Milvus 向量庫。
- 自動跑幾個 Demo 查詢，並開啟互動模式，讓你直接輸入「嫌疑人 / 證人描述」文字來檢索對應影片與時間點。

---

### 作者

- **姓名**：William
- **Email**：taiwanfifi@gmail.com
