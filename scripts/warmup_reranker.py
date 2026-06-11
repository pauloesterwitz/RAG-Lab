from rag_lab.reranker import rerank, warmup
name = warmup()
print("BACKEND:", name)
scores = rerank("What is corrective RAG?",
                ["Corrective RAG grades retrieved documents and corrects weak retrieval.",
                 "The mitochondria is the powerhouse of the cell.",
                 "CRAG uses a retrieval evaluator to decide actions."])
print("SCORES:", [round(s,3) for s in scores])
