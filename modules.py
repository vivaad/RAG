
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

class ScoreFilteredRetriever(VectorStoreRetriever):
    def __init__(self, vectorstore, score_threshold=0.3, k=5):
        super().__init__(vectorstore=vectorstore, search_kwargs={"k": k})
        self.score_threshold = score_threshold
    
    def get_relevant_documents(self, query):
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.search_kwargs["k"])
        return [doc for doc, score in docs_and_scores if score >= self.score_threshold]
