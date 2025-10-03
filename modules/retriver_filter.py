
from typing import Optional, List, Any
from langchain.schema import BaseRetriever, Document
from pydantic import Field

class ScoreFilteredRetriever(BaseRetriever):
    vectorstore: Any = Field(..., description="Chroma vector store instance")
    k: int = 5
    fetch_k: Optional[int] = None
    min_score: Optional[float] = None
    score_is_distance: bool = True

    class Config:
        arbitrary_types_allowed = True  # allows vectorstore to be any object

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_name: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> List[Document]:
    
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k or self.k)

        # Filter by score
        if self.min_score is not None:
            if self.score_is_distance:
                docs_and_scores = [(d, s) for d, s in docs_and_scores if s <= self.min_score]
            else:
                docs_and_scores = [(d, s) for d, s in docs_and_scores if s >= self.min_score]

        # Sort top-k
        if self.score_is_distance:
            docs_and_scores.sort(key=lambda x: x[1])
        else:
            docs_and_scores.sort(key=lambda x: x[1], reverse=True)

        return [d for d, _ in docs_and_scores[: self.k]]

    async def _aget_relevant_documents(        
        self,
        query: str,
        *,
        run_name: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> List[Document]:
        return self._get_relevant_documents(
                query, 
                run_name=run_name, 
                callbacks=callbacks, 
                tags=tags, 
                metadata=metadata, 
                **kwargs
        )
