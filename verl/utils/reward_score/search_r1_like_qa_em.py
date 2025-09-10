import re
import json
from typing import List, Union, Dict
import random
import string
from verl.workers.rollout.sglang_rollout.chat_client import run_chat_completions_sync

# =================================================================
# Constants
# =================================================================

FORMAT_REWARD = 0.5
CONTENT_REWARD = 0.5
JUDGE_REWARD = 0.5 

# =================================================================
# Answer Normalization & EM Check
# =================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: Union[str, List[str]]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


# =================================================================
# Solution String Parsing
# =================================================================

def get_tool_call_contents(s: str) -> List[str]:
    return re.findall(r"<tool_call>(.*?)<tool_call>", s, flags=re.DOTALL)


def extract_solution(solution_str: str) -> Union[str, None]:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


# =================================================================
# Solution Validation (Format Legality Check)
# =================================================================

def is_legal(solution_str: str) -> bool:
    # 1. Check balanced tool_call tags
    if solution_str.count("<tool_call>") != solution_str.count("<tool_call>"):
        return False

    tool_contents = get_tool_call_contents(solution_str)

    # 2. Check for duplicate tool calls
    normalized_contents = [c.replace("\n", "").strip() for c in tool_contents]
    if len(normalized_contents) != len(set(normalized_contents)):
        return False

    # 3. Validate JSON structure in each tool call
    for content in tool_contents:
        try:
            data = json.loads(content.strip())
            if not (
                isinstance(data, dict) and
                "name" in data and isinstance(data["name"], str) and
                "arguments" in data and isinstance(data["arguments"], dict) and
                set(data["arguments"].keys()) == {"query"} and
                isinstance(data["arguments"]["query"], str)
            ):
                return False
        except json.JSONDecodeError:
            return False

    # 4. Ensure exactly one valid <answer> tag with extractable content
    if solution_str.count("<answer>") != 1 or solution_str.count("</answer>") != 1:
        return False

    if extract_solution(solution_str) is None:
        return False

    return True


# =================================================================
# LLM-as-Judge (Minimal Integration)
# =================================================================

def llm_judge(predicted: str, goldens: List[str]) -> bool:
    if run_chat_completions_sync is None:
        return False

    golden_text = "\\n".join(goldens)
    prompt = f"""Determine if the predicted answer is semantically equivalent to any golden answer.
Golden Answers:
{golden_text}
Predicted Answer:
{predicted}
Respond ONLY with "yes" or "no"."""

    try:
        result = run_chat_completions_sync(
            messages_list=[[{"role": "user", "content": prompt}]],
            model="deepseek-v3",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            concurrent_limit=1,
            max_retries=3,
            save_to_file=False
        )
        response = result["results"][0]["response"]
        if response.get("status") != "success":
            return False
        judge_output = response["data"]["choices"][0]["message"]["content"].strip().lower()
        return "yes" in judge_output
    except Exception:
        return False


# =================================================================
# Main Scoring Function (with minimal LLM-as-Judge addition)
# =================================================================

def compute_score(solution_str: str, ground_truth: Dict) -> Dict[str, float]:
    predicted_answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if predicted_answer is not None:
            print(f"Extracted answer is not None: {predicted_answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    # Initialize rewards
    format_reward = FORMAT_REWARD if is_legal(solution_str) else 0.0
    content_reward = 0.0
    judge_reward = 0.0

    # Only check content if answer is extractable
    if predicted_answer is not None:
        golden_answers = ground_truth.get("target", [])
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        # First, try EM
        if em_check(predicted_answer, golden_answers):
            content_reward = CONTENT_REWARD
        # else:
        #     # EM failed → try LLM-as-Judge
        #     if llm_judge(predicted_answer, golden_answers):
        #         judge_reward = JUDGE_REWARD
        #         if do_print:
        #             print(f"[LLM Judge] Accepted: {predicted_answer}")

    total_score = format_reward + content_reward + judge_reward
    em_acc = 1.0 if (content_reward > 0 or judge_reward > 0) else 0.0

    return {"score": total_score, "acc": em_acc}