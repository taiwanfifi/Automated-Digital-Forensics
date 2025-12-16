import json
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus

# --- 基本路徑設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 指向 time_in_zone 目錄
JSONL_PATH = os.path.join(BASE_DIR, "people_zone_log.jsonl")
MILVUS_URI = os.path.join(os.path.dirname(__file__), "milvus_people_log.db")
COLLECTION_NAME = "people_zone_evidence_v1"


def load_people_logs(file_path: str) -> List[Document]:
    """從 people_zone_log.jsonl 讀取資料並轉成 Document。

    page_content: person_description
    metadata: 追蹤 ID、區域、時間、影片檔名等
    """
    docs: List[Document] = []

    if not os.path.exists(file_path):
        print(f"❌ 找不到 JSONL 檔案: {file_path}")
        return docs

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            description = record.get("person_description") or "Unknown person description."

            page_content = description

            video_file = record.get("video_file") or ""

            metadata = {
                "tracker_id": record.get("tracker_id"),
                "zone_index": record.get("zone_index"),
                "entered_at": record.get("entered_at"),
                "exited_at": record.get("exited_at"),
                "dwell_time_seconds": record.get("dwell_time_seconds"),
                # Milvus metadata 需要固定類型，這裡強制轉成字串
                "video_file": str(video_file),
                "global_timestamp": record.get("global_timestamp"),
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

    print(f"📂 已載入 {len(docs)} 筆 people_zone_log 證據。")
    return docs


def format_evidence(docs: List[Document]) -> str:
    """將檢索結果整理成 LLM 易讀的文字。"""
    if not docs:
        return "(未檢索到任何相關證據)"

    lines = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        lines.append(
            f"【證據 #{i}】\n"
            f"- 外觀描述: {doc.page_content}\n"
            f"- 追蹤 ID: {m.get('tracker_id')}\n"
            f"- 區域 (zone_index): {m.get('zone_index')}\n"
            f"- 進入時間 entered_at: {m.get('entered_at')}\n"
            f"- 離開時間 exited_at: {m.get('exited_at')}\n"
            f"- 停留秒數 dwell_time_seconds: {m.get('dwell_time_seconds')}\n"
            f"- 影片檔案 video_file: {m.get('video_file')}\n"
            f"- 原始時間戳 global_timestamp: {m.get('global_timestamp')}\n"
            "--------------------------------------------------"
        )
    return "\n".join(lines)


def build_vectorstore(docs: List[Document]) -> Milvus:
    """用 Ollama 的 embedding 模型，把描述寫入 Milvus。"""
    print("🧠 載入 Ollama Embedding 模型 (nomic-embed-text:v1.5)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

    print(f"🗄️ 建立 / 重建 Milvus collection: {COLLECTION_NAME}")
    vectorstore = Milvus.from_documents(
        docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,
    )
    print("✅ Milvus 建立完成。")
    return vectorstore


def build_rag_chain(vectorstore: Milvus):
    """建立一個簡單的 RAG chain：檢索 Top-k + LLM 分析。"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(model="llama3.1:8b", temperature=0)

    prompt_text = """
你是一位專業的數位鑑識專家與法律調查助理，
負責協助檢察官或律師，從監視器追蹤紀錄中找到符合描述的可疑對象。

以下是已經根據使用者查詢，從資料庫檢索出來的相關片段（可能是 Top-K 筆）：

{context}

使用者的查詢（他們希望找的人物或特徵）：
{question}

請你：
1. 評估這些證據中，哪些人物最可能符合描述（可以有多個）。
2. 對每一位可能目標，列出：
   - 追蹤 ID (tracker_id)
   - 影片檔案路徑 (video_file)
   - 進出時間 (entered_at / exited_at 或 global_timestamp)
   - 他的外觀特徵重點（用你自己的話總結）
   - 為什麼你覺得他符合查詢描述。
3. 如果證據不足或沒有找到明顯符合的對象，也請老實說明。

請用繁體中文、條列式輸出，語氣專業且中立。
"""

    prompt = ChatPromptTemplate.from_template(prompt_text)

    rag_chain = (
        {"context": retriever | format_evidence, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    print("--- 🕵️ AI People Zone RAG 檢索系統啟動 ---")
    print(f"JSONL 來源: {JSONL_PATH}")

    docs = load_people_logs(JSONL_PATH)
    if not docs:
        print("❌ 沒有任何 Document，被迫結束程式。")
        return

    vectorstore = build_vectorstore(docs)
    rag_chain = build_rag_chain(vectorstore)

    # Demo 幾個典型查詢
    demo_queries = [
        "幫我找穿黃褲子、紅色拖鞋的人",
        "找穿白色上衣、黑色長褲，看起來像上班族的男性",
    ]

    for q in demo_queries:
        print("\n==================================================")
        print(f"🔍 Demo 查詢: {q}")
        ans = rag_chain.invoke(q)
        print("\n📄 AI 報告:")
        print(ans)
        print("==================================================\n")

    # 互動模式
    while True:
        user_q = input("⚖️ 請輸入要搜尋的嫌疑人 / 證人特徵 (輸入 q 離開): ")
        if user_q.strip().lower() == "q":
            break
        if not user_q.strip():
            continue
        print(f"\n🔍 正在檢索: {user_q}")
        ans = rag_chain.invoke(user_q)
        print("\n📄 AI 報告:")
        print(ans)
        print("\n")


if __name__ == "__main__":
    main()
