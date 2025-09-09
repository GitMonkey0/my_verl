import asyncio
import aiohttp
import json
import random
import threading
from tqdm.asyncio import tqdm

async def chat_completion_with_retry(
    session,
    messages,
    index,
    api_token,
    model,
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    max_retries
):
    retry_count = 0
    while True:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}",
            "User-Agent": "curl/7.68.0"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        if stop is not None:
            payload["stop"] = stop

        try:
            async with session.post(
                "http://publicshare.a.pinggy.link/v1/chat/completions",
                data=json.dumps(payload),
                headers=headers,
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return index, {"status": "success", "data": result, "retries": retry_count}
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            retry_count += 1
            if max_retries > 0 and retry_count > max_retries:
                return index, {"status": "failed_permanently", "error": str(e), "retries": retry_count}

            delay = min(60, (2 ** retry_count) + random.uniform(0, 1))
            await asyncio.sleep(delay)

async def run_chat_completions(
    messages_list,
    api_token="token-abc123",
    model="deepseek-v3",
    max_tokens=100,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=None,
    concurrent_limit=20,
    max_retries=-1,
    save_to_file=False,
    output_file="results.json"
):
    semaphore = asyncio.Semaphore(concurrent_limit)

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            chat_completion_with_retry(
                session,
                msg,
                idx,
                api_token,
                model,
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                presence_penalty,
                stop,
                max_retries
            )
            for idx, msg in enumerate(messages_list)
        ]
        results = await tqdm.gather(*tasks, desc="Processing requests")

        results.sort(key=lambda x: x[0])

        output = {
            "total_requests": len(messages_list),
            "results": [
                {
                    "index": idx,
                    "input_messages": messages_list[idx],
                    "response": result
                }
                for idx, result in results
            ]
        }

        if save_to_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        success = sum(1 for _, r in results if r.get("status") == "success")
        print(f"\n✅ 完成: {success}/{len(messages_list)} 成功" + (f", 结果已保存至 {output_file}" if save_to_file else ""))

        return output

def run_chat_completions_sync(
    messages_list,
    api_token="token-abc123",
    model="deepseek-v3",
    max_tokens=100,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=None,
    concurrent_limit=20,
    max_retries=-1,
    save_to_file=False,
    output_file="results.json"
):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    kwargs = {
        "api_token": api_token,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "concurrent_limit": concurrent_limit,
        "max_retries": max_retries,
        "save_to_file": save_to_file,
        "output_file": output_file,
    }

    if loop and loop.is_running():
        result_container = []
        exception_container = []

        def run_in_thread():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(
                    run_chat_completions(messages_list, **kwargs)
                )
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if exception_container:
            raise exception_container[0]
        return result_container[0]
    else:
        return asyncio.run(run_chat_completions(messages_list, **kwargs))

