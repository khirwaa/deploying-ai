import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch, TavilyExtract


load_dotenv('../../.secrets')


if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

search = TavilySearch(
    max_results=3,
    topic='finance',
    description='tavily search engine',
    include_raw_content=True,
    include_images=False,
    include_image_descriptions=False,
    search_depth="advanced",
    # include_domains=["www.canada.ca", "thelocal.to", "211ontario.ca","investingintroduction.ca","ca.finance.yahoo.com"],
)

extract = TavilyExtract(
            extract_depth="basic")
    
@tool    
def web_search_tool(query: str) -> str:
    """
    This tool performs a web search for the given query and returns a summary of the search results.
    The search should be limited to a specific set of domains relevant to financial planning and retirement, such as www.canada.ca and thelocal.to.
    The tool should return a concise summary of the most relevant information found in the search results, along with the source URLs.
    """
    response = search.invoke(query)
    
    for result in response['results']:
        print(f"URL: {result['url']}, Score: {result['score']}")

    return response