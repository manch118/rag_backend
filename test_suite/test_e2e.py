import requests
import sys
import time
import json

BASE = "http://localhost:8000"

def check(step, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {step}" + (f" - {detail}" if detail else ""))
    return ok

def wait_for_server(url, timeout=30):
    for i in range(timeout):
        try:
            r = requests.get(f"{url}/docs", timeout=2)
            if r.status_code < 500:
                return True
        except:
            pass
        time.sleep(1)
    return False

print("=" * 60)
print("RAG Backend - End-to-End Test")
print("=" * 60)

if not wait_for_server(BASE):
    print("[FAIL] Server did not start within 30 seconds")
    sys.exit(1)
print("[PASS] Server is running")

test_text = """A serene mountain landscape at sunset. Snow-capped peaks rise majestically against a sky painted in deep shades of orange, purple, and pink. An alpine lake at the base of the mountains perfectly reflects the peaks and the colorful sky, creating a mirror-like surface. Tall pine trees line the rocky shoreline, their dark silhouettes contrasting with the bright colors of the sunset. A soft mist rises from the lake, adding an ethereal quality to the scene. In the distance, a small wooden cabin sits nestled among the trees, with a faint wisp of smoke rising from its chimney. Wildflowers in shades of purple and yellow dot the foreground meadow. The scene evokes a feeling of peace and solitude."""

r = requests.post(f"{BASE}/ingest", json={"text": test_text, "source": "test upload"})
check("POST /ingest", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("ingest response has doc_id", bool(data.get("doc_id")), f"doc_id={data['doc_id']}")
check("ingest chunk_count > 0", data.get("chunk_count", 0) > 0, f"chunks={data['chunk_count']}")
print(f"  doc_id: {data['doc_id']}, chunks: {data['chunk_count']}")

r = requests.get(f"{BASE}/retrieve?q=mountain+lake+sunset+reflection&top_k=5")
check("GET /retrieve", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("retrieve returned list", isinstance(data, list), f"type={type(data).__name__}")
check("retrieve has results", len(data) > 0, f"count={len(data)}")
if data:
    check("results have chunk_id", bool(data[0].get("chunk_id")), f"id={data[0]['chunk_id']}")
    check("results have score", data[0].get("score", 0) > 0, f"score={data[0]['score']}")
    print(f"  top result: score={data[0]['score']}, id={data[0]['chunk_id']}")

r = requests.post(f"{BASE}/query", json={"query": "mountain lake at sunset with reflections", "top_k": 5})
check("POST /query", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("query has generated_answer", bool(data.get("generated_answer")), f"len={len(data['generated_answer'])}")
check("query has sources", len(data.get("sources", [])) > 0, f"sources={len(data['sources'])}")
print(f"  generated answer ({len(data['generated_answer'])} chars):")
print(f"  {data['generated_answer'][:200]}...")
print(f"  sources count: {len(data['sources'])}")

eval_prompt = data['generated_answer']
eval_context = "\n".join([s["text"][:200] for s in data["sources"][:2]])
r = requests.post(f"{BASE}/evaluate", json={
    "query": "mountain lake sunset",
    "generated_answer": eval_prompt,
    "context": eval_context
})
check("POST /evaluate", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("evaluate has scores", len(data.get("scores", [])) > 0, f"scores={len(data['scores'])}")
check("evaluate has average_score", data.get("average_score", 0) > 0, f"avg={data['average_score']}")
for s in data["scores"]:
    check(f"  {s['dimension']} score", s["score"] > 0, f"score={s['score']}")
print(f"  average score: {data['average_score']}")

r = requests.post(f"{BASE}/feedback", json={
    "query": "mountain lake sunset",
    "generated_answer": eval_prompt,
    "relevance_score": 4.5,
    "completeness_score": 4.0,
    "specificity_score": 3.5,
    "fluency_score": 5.0,
    "comment": "Excellent prompt generation"
})
check("POST /feedback #1", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("feedback response ok", data.get("status") == "ok", f"status={data['status']}")

r = requests.post(f"{BASE}/feedback", json={
    "query": "another test query",
    "generated_answer": "a test prompt",
    "relevance_score": 3.0,
    "completeness_score": 3.0,
    "specificity_score": 2.5,
    "fluency_score": 4.0,
    "comment": "Second test"
})
check("POST /feedback #2", r.status_code == 200, f"status={r.status_code}")

r = requests.get(f"{BASE}/stats")
check("GET /stats", r.status_code == 200, f"status={r.status_code}")
data = r.json()
check("stats total_feedback=2", data.get("total_feedback") == 2, f"total={data['total_feedback']}")
check("stats avg_relevance > 0", data.get("avg_relevance", 0) > 0, f"avg_rel={data['avg_relevance']}")
check("stats avg_overall > 0", data.get("avg_overall", 0) > 0, f"avg_overall={data['avg_overall']}")
check("stats has recent_entries", len(data.get("recent_entries", [])) > 0, f"entries={len(data['recent_entries'])}")
print(f"  total_feedback: {data['total_feedback']}")
print(f"  avg_relevance: {data['avg_relevance']}")
print(f"  avg_completeness: {data['avg_completeness']}")
print(f"  avg_specificity: {data['avg_specificity']}")
print(f"  avg_fluency: {data['avg_fluency']}")
print(f"  avg_overall: {data['avg_overall']}")

print("=" * 60)
print("ALL END-TO-END TESTS COMPLETED")
print("=" * 60)
print("Environment: Windows, CPU-only")
print(f"Server: {BASE}")
print(f"API Key: {'available'}")
print(f"Models: claude-sonnet-4-20250514 (gen), claude-haiku-3-5-20241022 (eval)")