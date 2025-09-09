import re
import json
from typing import List
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen25_detector import Qwen25Detector


class Qwen25SingleCallNoTrailingDetector(Qwen25Detector):
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        bot_idx = text.find(self.bot_token)
        if bot_idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        leading_text = text[:bot_idx].strip()

        search_start = bot_idx
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match = re.search(pattern, text[search_start:], re.DOTALL)

        if not match:
            return StreamingParseResult(normal_text=leading_text, calls=[])

        full_match_start = search_start + match.start()
        full_match_end = search_start + match.end()

        json_str = match.group(1).strip()
        calls = []
        try:
            parsed_call = json.loads(json_str)
            calls.extend(self.parse_base_json(parsed_call, tools))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse first tool call JSON: {json_str}, error: {e}")

        return StreamingParseResult(normal_text=leading_text, calls=calls)