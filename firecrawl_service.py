import httpx
from typing import Any, Dict, List, Optional, Tuple


class FirecrawlService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.com/v1"
        
    async def scrape_website(self, url: str) -> Dict[str, Any]:
        """Scrape website content using Firecrawl API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/scrape",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "url": url,
                    "render": True,
                    "extract_text": True,
                    "extract_links": True,
                    "extract_metadata": True
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Firecrawl API error: {response.status_code} - {response.text}")
                
            return response.json()
    
    async def search_web(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "limit": limit,
                    "include_snippets": True
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Firecrawl API error: {response.status_code} - {response.text}")
                
            return response.json()["results"]