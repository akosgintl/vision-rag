"""Embedding index service using Qdrant for ColPali multi-vector storage."""

import uuid

import structlog
from qdrant_client import AsyncQdrantClient, models

from proxy.config import settings

logger = structlog.get_logger()


class EmbeddingIndex:
    """Manages ColPali embeddings in Qdrant using the async client."""

    EMBEDDING_DIM = 320  # TomoroAI/tomoro-ai-colqwen3-embed-8b-awq embedding dimension

    def __init__(self, url: str | None = None, collection: str | None = None):
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.client = AsyncQdrantClient(url=self.url)

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "colpali": models.VectorParams(
                        size=self.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                    ),
                },
            )
            # Create payload index for fast filtering
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="collection",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_collection_created", collection=self.collection)

    async def index_page(
        self,
        embeddings: list[list[float]],
        document_id: str,
        page_number: int,
        collection_name: str = "default",
        metadata: dict | None = None,
    ) -> str:
        """
        Index a single page's multi-vector embeddings.

        Args:
            embeddings: List of patch embeddings from ColPali [[float]*128, ...]
            document_id: Source document ID
            page_number: Page number within document
            collection_name: Logical collection for filtering
            metadata: Extra metadata to store

        Returns:
            point_id: UUID of the indexed point
        """
        point_id = str(uuid.uuid4())
        payload = {
            "document_id": document_id,
            "page_number": page_number,
            "collection": collection_name,
        }
        if metadata:
            payload["metadata"] = metadata

        await self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"colpali": embeddings},
                    payload=payload,
                )
            ],
        )
        logger.info(
            "page_indexed",
            point_id=point_id,
            document_id=document_id,
            page=page_number,
            num_vectors=len(embeddings),
        )
        return point_id

    async def search(
        self,
        query_embeddings: list[list[float]],
        collection_name: str = "default",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for similar pages using query multi-vector embeddings.

        Args:
            query_embeddings: Query patch embeddings from ColPali
            collection_name: Filter to this collection
            top_k: Number of results

        Returns:
            List of dicts with document_id, page_number, score
        """
        results = await self.client.query_points(
            collection_name=self.collection,
            query=query_embeddings,
            using="colpali",
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="collection",
                        match=models.MatchValue(value=collection_name),
                    )
                ]
            ),
        )

        return [
            {
                "point_id": str(r.id),
                "document_id": r.payload["document_id"],
                "page_number": r.payload["page_number"],
                "score": r.score,
                "collection": r.payload["collection"],
            }
            for r in results.points
        ]

    async def delete_document(self, document_id: str) -> None:
        """Delete all pages belonging to a document."""
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info("document_deleted", document_id=document_id)

    async def count(self, collection_name: str | None = None) -> int:
        """Count indexed pages, optionally filtered by collection."""
        if collection_name:
            result = await self.client.count(
                collection_name=self.collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="collection",
                            match=models.MatchValue(value=collection_name),
                        )
                    ]
                ),
            )
        else:
            result = await self.client.count(collection_name=self.collection)
        return result.count

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self.client.close()
