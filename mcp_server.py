import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base

from firecrawl_service import FirecrawlService
from groq_service import GroqService
from configs import AppConfig


@dataclass
class AppContext:
    config: AppConfig
    groq_service: GroqService
    firecrawl_service: FirecrawlService


# Lifespan manager for resource initialization and cleanup
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")
    groq_model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    if not firecrawl_api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable is required")
    
   
    config = AppConfig(
        groq_api_key=groq_api_key,
        firecrawl_api_key=firecrawl_api_key,
        groq_model=groq_model
    )
    
    
    groq_service = GroqService(config.groq_api_key, config.groq_model)
    firecrawl_service = FirecrawlService(config.firecrawl_api_key)
    
    print(f"Initialized MCP server with Groq model: {config.groq_model}")
    
    try:
        yield AppContext(
            config=config,
            groq_service=groq_service,
            firecrawl_service=firecrawl_service
        )
    finally:
        print("Shutting down MCP server")


# Create the MCP server
mcp = FastMCP(
    "Groq-Firecrawl MCP Server",
    lifespan=app_lifespan,
    dependencies=["httpx>=0.24.0", "pydantic>=2.0.0"]
)


# Define resources
@mcp.resource("website://{url}")
async def get_website_content(url: str, ctx: Context) -> str:
    """Get content from a website URL"""
    firecrawl = ctx.request_context.lifespan_context.firecrawl_service
    
    ctx.info(f"Scraping website: {url}")
    try:
        result = await firecrawl.scrape_website(url)
        return result.get("text", "No content found")
    except Exception as e:
        return f"Error scraping website: {str(e)}"


@mcp.resource("search://{query}/{limit}")
async def search_web(query: str, limit: str, ctx: Context) -> str:
    """Search the web for information"""
    firecrawl = ctx.request_context.lifespan_context.firecrawl_service
    config = ctx.request_context.lifespan_context.config
    
    try:
        limit_int = int(limit) if limit else config.default_search_limit
    except ValueError:
        limit_int = config.default_search_limit
    
    ctx.info(f"Searching web for: {query} with limit {limit_int}")
    
    try:
        results = await firecrawl.search_web(query, limit_int)
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"Result {i}:")
            formatted_results.append(f"Title: {result.get('title', 'No title')}")
            formatted_results.append(f"URL: {result.get('url', 'No URL')}")
            formatted_results.append(f"Snippet: {result.get('snippet', 'No snippet')}")
            formatted_results.append("")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching web: {str(e)}"


@mcp.resource("metadata://{url}")
async def get_website_metadata(url: str, ctx: Context) -> str:
    """Get metadata from a website URL"""
    firecrawl = ctx.request_context.lifespan_context.firecrawl_service
    
    ctx.info(f"Getting metadata for website: {url}")
    try:
        result = await firecrawl.scrape_website(url)
        metadata = result.get("metadata", {})
        
        formatted_metadata = []
        formatted_metadata.append(f"Title: {metadata.get('title', 'No title')}")
        formatted_metadata.append(f"Description: {metadata.get('description', 'No description')}")
        formatted_metadata.append(f"Author: {metadata.get('author', 'No author')}")
        formatted_metadata.append(f"Published: {metadata.get('published_date', 'No date')}")
        
        return "\n".join(formatted_metadata)
    except Exception as e:
        return f"Error getting metadata: {str(e)}"


# Define tools
@mcp.tool()
async def generate_content(prompt: str, ctx: Context) -> str:
    """Generate content using Groq LLM"""
    groq = ctx.request_context.lifespan_context.groq_service
    
    ctx.info(f"Generating content with prompt: {prompt[:50]}...")
    try:
        return await groq.generate_text(prompt)
    except Exception as e:
        return f"Error generating content: {str(e)}"


@mcp.tool()
async def summarize_text(ctx: Context, content: str, max_length: int = 500) -> str:
    """Summarize text content using Groq LLM"""
    groq = ctx.request_context.lifespan_context.groq_service
    
    ctx.info(f"Summarizing content of length {len(content)} with max length {max_length}")
    try:
        return await groq.summarize_content(content, max_length)
    except Exception as e:
        return f"Error summarizing content: {str(e)}"


@mcp.tool()
async def analyze_website(ctx: Context, url: str, include_summary: bool = True) -> str:
    """Analyze a website and optionally summarize its content"""
    firecrawl = ctx.request_context.lifespan_context.firecrawl_service
    groq = ctx.request_context.lifespan_context.groq_service
    
    ctx.info(f"Analyzing website: {url}")
    try:
        # Step 1: Scrape the website
        await ctx.report_progress(0, 3)
        scrape_result = await firecrawl.scrape_website(url)
        content = scrape_result.get("text", "")
        metadata = scrape_result.get("metadata", {})
        
        # Step 2: Prepare analysis
        await ctx.report_progress(1, 3)
        analysis = [
            f"Website Analysis: {url}",
            f"Title: {metadata.get('title', 'No title')}",
            f"Description: {metadata.get('description', 'No description')}",
            f"Content Length: {len(content)} characters",
            f"Links Found: {len(scrape_result.get('links', []))}",
        ]
        
        # Step 3: Generate summary if requested
        if include_summary and content:
            await ctx.report_progress(2, 3)
            summary = await groq.summarize_content(content)
            analysis.append("\nSummary:")
            analysis.append(summary)
        
        await ctx.report_progress(3, 3)
        return "\n".join(analysis)
    except Exception as e:
        return f"Error analyzing website: {str(e)}"


@mcp.tool()
async def research_topic(ctx: Context, topic: str, depth: int = 2) -> str:
    """Research a topic by searching the web and analyzing top results"""
    firecrawl = ctx.request_context.lifespan_context.firecrawl_service
    groq = ctx.request_context.lifespan_context.groq_service
    
    ctx.info(f"Researching topic: {topic} with depth {depth}")
    try:
        # Step 1: Search for the topic
        ctx.info("Searching web...")
        await ctx.report_progress(0, depth + 1)
        search_results = await firecrawl.search_web(topic, limit=depth)
        
        if not search_results:
            return f"No search results found for topic: {topic}"
        
        # Step 2: Analyze each result
        research_data = []
        for i, result in enumerate(search_results):
            url = result.get("url")
            if not url:
                continue
                
            ctx.info(f"Analyzing result {i+1}: {url}")
            await ctx.report_progress(i + 1, depth + 1)
            
            try:
                scrape_result = await firecrawl.scrape_website(url)
                content = scrape_result.get("text", "")
                
                if content:
                    summary = await groq.summarize_content(content, max_length=300)
                    research_data.append(f"Source {i+1}: {result.get('title', 'No title')}")
                    research_data.append(f"URL: {url}")
                    research_data.append(f"Summary: {summary}")
                    research_data.append("")
            except Exception as inner_e:
                research_data.append(f"Error analyzing {url}: {str(inner_e)}")
        
        # Step 3: Generate a comprehensive analysis
        if research_data:
            research_text = "\n".join(research_data)
            prompt = f"Based on the following research on '{topic}', provide a comprehensive analysis:\n\n{research_text}"
            final_analysis = await groq.generate_text(prompt)
            
            return f"# Research on: {topic}\n\n{final_analysis}\n\n## Sources\n\n{research_text}"
        else:
            return f"Could not gather meaningful research data on topic: {topic}"
    except Exception as e:
        return f"Error researching topic: {str(e)}"


# Define prompts
@mcp.prompt()
def research_prompt(topic: str) -> str:
    return f"""I need to research about {topic}. Please:
1. Search for relevant information
2. Analyze the key findings
3. Provide a comprehensive summary
4. Include citations to sources
"""


@mcp.prompt()
def website_analysis_prompt(url: str) -> list[base.Message]:
    return [
        base.UserMessage(f"Please analyze this website: {url}"),
        base.UserMessage("I'd like to understand its content, structure, and main points."),
        base.AssistantMessage("I'll analyze this website for you. Would you like me to include a summary of the content?"),
    ]


# Main execution block
if __name__ == "__main__":
    mcp.run()