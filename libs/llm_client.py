# libs/llm_client.py
import os, time, json
from typing import Any, Dict, Optional, Union
import httpx
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APITimeoutError
from libs.log_utils import setup_logger, safe_write_json, now_slug, redact


def _sanitize_proxies(p: Optional[Union[str, Dict[str, str]]]) -> Optional[Union[str, Dict[str, str]]]:
    """
    Accepts None, a proxy URL string, or a dict of scheme->url.
    - If dict, drop empty/whitespace values.
    - If result is empty, return None.
    """
    if not p:
        return None
    if isinstance(p, str):
        return p.strip() or None
    if isinstance(p, dict):
        cleaned = {k: v for k, v in p.items() if isinstance(v, str) and v.strip()}
        return cleaned or None
    return None


class LLMClient:
    """
    Usage:
        llm = LLMClient(
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
            request_timeout_sec=cfg["llm"].get("request_timeout_sec", 60),
            connect_timeout_sec=cfg["llm"].get("connect_timeout_sec", 30),
            read_timeout_sec=cfg["llm"].get("read_timeout_sec", 60),
            max_retries=cfg["llm"].get("max_retries", 4),
            json_strict=cfg["llm"].get("json_strict", True),
            verify_ssl=cfg.get("network", {}).get("verify_ssl", True),
            ca_bundle_path=cfg.get("network", {}).get("ca_bundle_path"),
            proxies=cfg.get("network", {}).get("proxies"),
            base_url=cfg.get("network", {}).get("openai_base_url"),
            log_dir="outputs/<run_id>/artifacts/logs",
            log_level=cfg.get("log_level","INFO"),
        )
    """
    def __init__(self,
                 model: str,
                 temperature: float = 0.1,
                 request_timeout_sec: int = 60,
                 connect_timeout_sec: int = 30,
                 read_timeout_sec: int = 60,
                 max_retries: int = 4,
                 json_strict: bool = True,
                 api_key: Optional[str] = None,
                 verify_ssl: bool = True,
                 ca_bundle_path: Optional[str] = None,
                 proxies: Optional[Union[str, Dict[str, str]]] = None,
                 base_url: Optional[str] = None,
                 log_dir: str = "artifacts/logs",
                 log_level: str = "INFO",
                 log_prompts: bool = True,
                 redact_values: Optional[list] = None,
                 max_input_bytes: int = 180000):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI API key not provided (in config or env).")

        self.logger = setup_logger("llm", log_dir, log_level)

        # httpx verify parameter:
        # - True to use system CA
        # - str path to a CA bundle (e.g., corp root)
        # - False to disable verify (not recommended)
        verify_param: Any = False if not verify_ssl else (ca_bundle_path or True)

        # sanitize proxies (support str or dict)
        proxies = _sanitize_proxies(proxies)

        timeout = httpx.Timeout(
            timeout=request_timeout_sec,
            connect=connect_timeout_sec,
            read=read_timeout_sec,
            write=read_timeout_sec
        )

        # Build http client, auto-fallback when proxy value is invalid
        try:
            http_client = httpx.Client(timeout=timeout, verify=verify_param, proxies=proxies or None)
        except Exception as e:
            self.logger.warning(f"Invalid proxy configuration detected ({e}); falling back to no proxy.")
            http_client = httpx.Client(timeout=timeout, verify=verify_param)

        self.client = OpenAI(api_key=key, http_client=http_client, base_url=base_url or None)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.json_strict = json_strict
        self.log_dir = log_dir
        self.log_prompts = log_prompts
        self.redact_values = (redact_values or []) + [key]
        self.max_input_bytes = max_input_bytes


    def _maybe_truncate(self, text: str) -> str:
        """Hard truncate by bytes to avoid over-long prompts."""
        if isinstance(text, str) and len(text.encode("utf-8")) > self.max_input_bytes:
            b = text.encode("utf-8")[: self.max_input_bytes]
            return b.decode("utf-8", errors="ignore")
        return text


    def _chat(self, messages, response_format=None) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if response_format:
            payload["response_format"] = response_format

        # log input size info
        total_bytes = sum(len((m.get("content") or "").encode("utf-8")) for m in messages)
        self.logger.info(f"LLM call -> model={self.model}, msgs={len(messages)}, bytes={total_bytes}")

        # optionally save prompt
        stamp = now_slug()
        prompt_path = os.path.join(self.log_dir, f"{stamp}_prompt.json")
        if self.log_prompts:
            safe_write_json(redact(payload, self.redact_values), prompt_path)

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**payload)
                dur = round(time.time() - t0, 2)
                self.logger.info(f"LLM ok (attempt {attempt}) in {dur}s")
                out = resp.to_dict()
                if self.log_prompts:
                    resp_path = os.path.join(self.log_dir, f"{stamp}_response.json")
                    safe_write_json(out, resp_path)
                return out
            except (RateLimitError, APIConnectionError, APITimeoutError, httpx.ReadTimeout) as e:
                dur = round(time.time() - t0, 2)
                self.logger.warning(f"LLM retryable error (attempt {attempt}) after {dur}s: {type(e).__name__}: {e}")
                if attempt == self.max_retries:
                    if self.log_prompts:
                        err_path = os.path.join(self.log_dir, f"{stamp}_error.txt")
                        with open(err_path, "w", encoding="utf-8") as f:
                            f.write(f"{type(e).__name__}: {e}\n")
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("LLM call failed after retries.")


    def json_completion(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Strict JSON answer:
          - Appends a line to the system prompt containing the word 'json' to satisfy OpenAI's
            response_format=json_object gate.
          - Attempts a fallback JSON extraction if the model returns extra text.
        """
        if self.json_strict:
            if system_prompt is None:
                system_prompt = ""
            system_prompt = (system_prompt + "\n\n"
                             "CRITICAL OUTPUT RULE: respond ONLY with valid json (a single JSON object). "
                             "Do not include prose before or after the json.\n")

        user_prompt = self._maybe_truncate(user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response_format = {"type": "json_object"} if self.json_strict else None
        resp = self._chat(messages, response_format=response_format)
        content = resp["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # attempt JSON extraction
            start = content.find("{"); end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end+1])
            self.logger.error("JSON decode failed; saving raw content.")
            if self.log_prompts:
                with open(os.path.join(self.log_dir, f"{now_slug()}_badjson.txt"), "w", encoding="utf-8") as f:
                    f.write(content)
            raise


    def text_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Plain text chat completion (no JSON response_format). Returns message.content (str).
        Uses the same retry/log/timeouts plumbing as json_completion.
        """
        user_prompt = self._maybe_truncate(user_prompt)
        messages = [
            {"role": "system", "content": system_prompt or "" },
            {"role": "user", "content": user_prompt},
        ]
        resp = self._chat(messages, response_format=None)  # <- no JSON forcing
        content = (resp["choices"][0]["message"]["content"] or "").strip()
        return content
