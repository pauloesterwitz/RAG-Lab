import sys, numpy as np
from pathlib import Path
from rag_lab.ingest import _chunk_doc, extract_pages, list_pdfs
from rag_lab.ollama_client import embed_many, embed_one
from rag_lab.store import BaseIndex
from rag_lab.approaches.plain import PlainRAG

pdfs = list_pdfs()
small = [p for p in pdfs if "Software Engineering" in p.name][0]
print("PDF:", small.name)
pages = extract_pages(small)
print("pages:", len(pages))
chunks = _chunk_doc(small.name, pages)[:30]
print("chunks:", len(chunks), "| sample chars:", chunks[0].char_count, "| cite:", chunks[0].citation())
vecs = embed_many([c.text for c in chunks], role="document")
emb = np.asarray(vecs, dtype=np.float32)
print("emb shape:", emb.shape)
idx = BaseIndex(chunks, emb)
app = PlainRAG(idx)
trace=[]
res = app.retrieve("What are the challenges of software engineering for machine learning?", trace)
print("--- top hits ---")
for rc in res:
    print(round(rc.score,3), rc.chunk.citation(), "::", rc.chunk.text[:90].replace("\n"," "))
print("trace:", [t.label for t in trace])
print("SMOKE OK")
