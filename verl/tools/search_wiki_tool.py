import logging
import uuid
from typing import Any, Optional, List

import aiohttp
from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class RetrievalTool(BaseTool):
    """
    A retrieval tool that searches documents for given queries.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        assert "search_url" in config, "search_url must be provided in config"
        assert "topk" in config, "topk must be provided in config"
        self.search_url = config["search_url"]
        self.topk = config["topk"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "retrieve_documents",
                "description": "Retrieve relevant documents for given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",          
                            "description": "A single search query string."
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        if kwargs:
            logger.debug(f"[RetrievalTool.create] Ignored kwargs: {kwargs}")
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        if not isinstance(query, str):
            query = str(query)

        logger.debug(f"Executing retrieval for query: {query}")

        search_results = await self._batch_search([query])
        formatted = [self._passages2string(res) for res in search_results]
        response_str = "\n\n".join(formatted)

        return ToolResponse(text=response_str[:2000]), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # 可选：记录释放日志
        # logger.debug(f"Released retrieval tool instance {instance_id}")
        pass

    async def _batch_search(self, queries: List[str]) -> List[List[dict]]:
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.search_url, json=payload) as resp:
                    data = await resp.json()
            return data.get("result", [])
        except Exception as e:
            logger.warning(f"Search failed for queries {queries}: {e}")
            return []

    def _passages2string(self, retrieval_result: List[dict]) -> str:
        parts = []
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title, *body = content.split("\n", 1)
            body = body[0] if body else ""
            parts.append(f"Doc {idx + 1} (Title: {title})\n{body}")
        return "\n".join(parts)

# import json
# import uuid
# from typing import Any, Optional, List

# import aiohttp
# from pydantic import BaseModel

# from verl.utils.rollout_trace import rollout_trace_op
# from .base_tool import BaseTool
# from .schemas import OpenAIFunctionToolSchema


# class RetrievalTool(BaseTool):
#     """
#     A retrieval tool that searches documents for given queries.
#     """

#     def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
#         super().__init__(config, tool_schema)
#         assert "search_url" in config, "search_url must be provided in config"
#         assert "topk" in config, "topk must be provided in config"
#         self.search_url = config["search_url"]
#         self.topk = config["topk"]

#     def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
#         return OpenAIFunctionToolSchema.model_validate({
#             "type": "function",
#             "function": {
#                 "name": "retrieve_documents",
#                 "description": "Retrieve relevant documents for given query.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "query": {
#                             "type": "string",          
#                             "description": "A single search query string."
#                         }
#                     },
#                     "required": ["query"]
#                 }
#             }
#         })

#     async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
#         if instance_id is None:
#             instance_id = str(uuid.uuid4())
#         return instance_id

#     @rollout_trace_op
#     async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
#         query = parameters.get("query", "")
#         if not isinstance(query, str):
#             query = str(query)

#         search_results = await self._batch_search([query])
#         formatted = [self._passages2string(res) for res in search_results]
#         response_str = "\n\n".join(formatted)

#         return response_str[:2000], 0.0, {}

#     async def calc_reward(self, instance_id: str, **kwargs) -> float:
#         return 0.0

#     async def release(self, instance_id: str, **kwargs) -> None:
#         pass

#     async def _batch_search(self, queries: List[str]) -> List[List[dict]]:
#         payload = {"queries": queries, "topk": self.topk, "return_scores": True}
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(self.search_url, json=payload) as resp:
#                     data = await resp.json()

#             return data.get("result", [])
#         except:
#             return []

#     def _passages2string(self, retrieval_result: List[dict]) -> str:
#         parts = []
#         for idx, doc_item in enumerate(retrieval_result):
#             content = doc_item["document"]["contents"]
#             title, *body = content.split("\n", 1)
#             body = body[0] if body else ""
#             parts.append(f"Doc {idx + 1} (Title: {title})\n{body}")
#         return "\n".join(parts)