from verl.workers.rollout.sglang_rollout.chat_client import run_chat_completions_sync

def llm_judge(predicted, goldens):
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
            max_retries=1,
            save_to_file=False
        )
        response = result["results"][0]["response"]
        if response.get("status") != "success":
            return False
        judge_output = response["data"]["choices"][0]["message"]["content"].strip().lower()
        return "yes" in judge_output
    except Exception:
        return False

print(llm_judge("142", ["one hundred forty-two"]))  # True