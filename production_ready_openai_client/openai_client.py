import os
import openai
import logging
import time
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session, ConnectionError, Timeout, HTTPError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, retries=3, backoff_factor=0.5, timeout=5, cache_enabled=False) -> None:
        self.api_key = self._load_api_key()
        openai.api_key = self.api_key
        self.cache_enabled = cache_enabled
        self.cache = {} if cache_enabled else None
        self.session = self._create_session(retries, backoff_factor, timeout)

    def _load_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key

    def _create_session(self, retries, backoff_factor, timeout) -> Session:
        session = Session()

        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.request = self._apply_timeout(session.request, timeout)
        return session

    def _apply_timeout(self, request_method, timeout) -> callable:
        def wrapped(*args, **kwargs):
            kwargs.setdefault('timeout', timeout)
            return request_method(*args, **kwargs)
        return wrapped

    def chat_completion(self, model="gpt-3.5-turbo", messages=None, temperature=0.7) -> openai.ChatCompletion:
        if messages is None:
            messages = []

        cache_key = self._generate_cache_key(model, messages, temperature)
        if self.cache_enabled and cache_key in self.cache:
            logger.info("Using cached result for chat completion")
            return self.cache[cache_key]

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            logger.info("Chat completion successful")

            if self.cache_enabled:
                self.cache[cache_key] = response

            return response
        except (openai.OpenAIError, ConnectionError, HTTPError, Timeout) as e:
            self._handle_api_errors(e)
            return None

    def _generate_cache_key(self, model, messages, temperature) -> str:
        return f"{model}_{hash(str(messages))}_{temperature}"

    def _handle_api_errors(self, error) -> None:
        if isinstance(error, openai.Timeout):
            logger.error("OpenAI request timed out: %s", error)
        elif isinstance(error, openai.APIError):
            logger.error("OpenAI API error: %s", error)
        elif isinstance(error, openai.RateLimitError):
            logger.error("Rate limit exceeded: %s", error)
            time.sleep(2)
        else:
            logger.error("Unexpected error: %s", error)

    def clear_cache(self) -> None:
        if self.cache_enabled:
            self.cache.clear()


if __name__ == "__main__":
    client = OpenAIClient(retries=5, backoff_factor=1, timeout=10, cache_enabled=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]

    response = client.chat_completion(messages=messages)
    if response:
        print(response.choices[0].message.content)
    else:
        print("Failed to get a response from OpenAI.")
