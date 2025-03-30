from dataclasses import dataclass


@dataclass
class AppConfig:
    groq_api_key: str
    firecrawl_api_key: str
    groq_model: str = "llama3-70b-8192"
    default_search_limit: int = 5
    request_timeout: int = 30