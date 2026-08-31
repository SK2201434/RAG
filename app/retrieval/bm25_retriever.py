from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        tokenized_documents = [self._tokenize(document.page_content) for document in documents]
        self.bm25 = BM25Okapi(tokenized_documents)

    @staticmethod
    def _tokenize(text:str) -> List[str]:
        return text.lower().split()

    def search(self,query:str,k:int=5)-> List[Tuple[Document,float]]:
        query_tokens  = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
        return [(self.documents[index], float(scores[index]),) for index in ranked_indices]