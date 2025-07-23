"""
utils/llm_call.py

This module calls the OpenAI Chat API with retry mechanism.
"""

import requests
import time


def call_openai_chat(prompt: str, api_key: str, model: str, base_url: str, max_retries: int = 3) -> str:
    """Call OpenAI Chat API with retry mechanism

    Args:
        prompt: Prompt text
        api_key: API key
        model: Model name
        base_url: API base URL
        max_retries: Max retry times

    Returns:
        str: API response content
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    session = requests.Session()
    retry_strategy = requests.adapters.Retry(
        total=max_retries,
        backoff_factor=2,  # Increase backoff time
        status_forcelist=[500, 502, 503, 504, 408, 429],  # Add timeout and rate limit status codes
        allowed_methods=["POST"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=60)  # 增加超时时间到60秒
        response.raise_for_status()  # 检查响应状态
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"🔴 API request error: {str(e)}")
        if isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout)):
            print("Detected connection error, retrying...")
            # Special handling for connection errors
            for i in range(max_retries):
                try:
                    print(f"Retry #{i+1}...")
                    time.sleep(2 ** i)  # 指数退避
                    response = session.post(url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except requests.exceptions.RequestException as retry_e:
                    print(f"Retry #{i+1} failed: {str(retry_e)}")
                    if i == max_retries - 1:  # 如果是最后一次重试
                        print("All retries failed")
                        return ""
    except Exception as e:
        print(f"🔴 Other error: {str(e)}")
        return ""
    finally:
        session.close()
