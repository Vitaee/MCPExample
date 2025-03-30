import httpx


class GroqService:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/v1"
        
    async def generate_text(self, prompt: str) -> str:
        """Generate text from Groq LLM API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def summarize_content(self, content: str, max_length: int = 500) -> str:
        """Summarize content using Groq LLM"""
        prompt = f"Please summarize the following content in a concise way, maximum {max_length} words:\n\n{content}"
        return await self.generate_text(prompt)