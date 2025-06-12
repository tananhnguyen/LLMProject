import sys
import os
import json
import arxiv
from loguru import logger
from typing import List
from datetime import datetime
# from Bio import Entrez, Medline

# Entrez.email = "202459045@jbnu.ac.kr"

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from mcp.server.fastmcp import FastMCP
# from pubmed_tool import PubMedSearchTool
from translate_tool import translate_en_to_ko, translate_ko_to_en

# pubmed_tool = PubMedSearchTool(max_results=5)

PAPER_DIR = os.getenv("PAPER_DIR", "papers")
RESEARCH_PORT = int(os.getenv("RESEARCH_PORT", "8001"))

# Port 8000 
mcp = FastMCP(
    name="mcp-server",
    port=8000,
)

# @mcp.tool()
# def search_pubmed(query: str, max_results: int = 5) -> dict:
#     """Search PubMed research articles by query"""
#     return str(pubmed_tool(query=query))

@mcp.tool()
def translate_en_kr(query: str) -> dict:
    """Translate English text to Korean"""
    return translate_en_to_ko(query=query)

@mcp.tool()
def translate_kr_en(query: str) -> dict:
    """Translate Korean text to English"""
    return translate_ko_to_en(query=query)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _topic_path(topic: str) -> str:
    """Return filesystem path for a research *topic*."""
    return os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))


def _papers_file(topic: str) -> str:
    return os.path.join(_topic_path(topic), "papers_info.json")


def _load_topic(topic: str) -> dict[str, dict[str, str]]:
    """Load papers metadata for *topic* (may be empty)."""
    path = _papers_file(topic)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Corrupted JSON in %s – starting fresh", path)
        return {}


def _save_topic(topic: str, data: dict[str, dict[str, str]]) -> None:
    os.makedirs(_topic_path(topic), exist_ok=True)
    with open(_papers_file(topic), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """Search arXiv for *topic* and persist metadata. Returns stored short IDs."""
    print(f"Searching arXiv for topic: {topic} with max results: {max_results}")
    logger.info("🔎 Searching arXiv for '%s' (%d results)", topic, max_results)

    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    stored_ids: List[str] = []
    async for paper in client.results_async(search):
        pid = paper.get_short_id()
        stored_ids.append(pid)
        meta = _load_topic(topic)
        meta[pid] = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": paper.published.date().isoformat(),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        _save_topic(topic, meta)

    logger.success("✓ Stored %d papers for topic '%s'", len(stored_ids), topic)
    return "hi"


# @mcp.tool()
# async def search_papers_pubmed(topic: str, max_results: int = 5) -> List[str]:
#     """Search PubMed for *topic* and persist metadata. Returns stored PMIDs."""
#     logger.info("🔎 Searching PubMed for '%s' (%d results)", topic, max_results)

#     # Step 1: Search for PMIDs
#     loop = asyncio.get_running_loop()
#     def esearch():
#         handle = Entrez.esearch(
#             db="pubmed",
#             term=topic,
#             retmax=max_results,
#             sort="relevance",
#             retmode="xml",
#         )
#         results = Entrez.read(handle)
#         handle.close()
#         return results.get("IdList", [])

#     pmids = await loop.run_in_executor(None, esearch)
#     if not pmids:
#         logger.warning("No papers found for topic '%s'", topic)
#         return []

#     # Step 2: Fetch summaries
#     def efetch(id_list):
#         handle = Entrez.efetch(
#             db="pubmed",
#             id=",".join(id_list),
#             rettype="medline",
#             retmode="text",
#         )
#         records = Medline.parse(handle)
#         return list(records)

#     records = await loop.run_in_executor(None, efetch, pmids)

#     stored_ids: List[str] = []
#     meta = _load_topic(topic)
#     for rec in records:
#         pmid = rec.get("PMID")
#         if not pmid:
#             continue
#         stored_ids.append(pmid)
#         meta[pmid] = {
#             "title": rec.get("TI", ""),
#             "authors": rec.get("AU", []),
#             "abstract": rec.get("AB", ""),
#             "journal": rec.get("JT", ""),
#             "published": rec.get("DP", ""),
#             "saved_at": datetime.utcnow().isoformat() + "Z",
#         }

#     _save_topic(topic, meta)
#     logger.success("✓ Stored %d papers for topic '%s'", len(stored_ids), topic)
#     return stored_ids


@mcp.tool()
async def extract_info(paper_id: str) -> str:
    """Retrieve stored metadata for *paper_id* across all topics."""
    logger.info("ℹ Extracting info for paper '%s'", paper_id)
    if not os.path.exists(PAPER_DIR):
        return f"No papers database found in '{PAPER_DIR}'."

    for topic_dir in os.listdir(PAPER_DIR):
        info_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")
        if not os.path.exists(info_file):
            continue
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if paper_id in data:
            return json.dumps(data[paper_id], indent=2)

    return f"No stored information for paper '{paper_id}'."

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("papers://folders")
async def list_folders() -> str:
    """Return markdown list of all topic folders."""
    logger.debug("Listing topic folders")
    lines = ["# Available Topics", ""]

    if not os.path.exists(PAPER_DIR):
        lines.append("No topics downloaded yet.")
        return "\n".join(lines)

    folders = [
        d for d in os.listdir(PAPER_DIR)
        if os.path.isdir(os.path.join(PAPER_DIR, d))
    ]
    if not folders:
        lines.append("No topics downloaded yet.")
    else:
        for f in folders:
            lines.append(f"- {f}")
        lines.append("")
        lines.append("Use `@<topic>` to view papers inside a topic.")

    return "\n".join(lines)


@mcp.resource("papers://{topic}")
async def topic_papers(topic: str) -> str:
    """Return markdown with all papers stored under *topic*."""
    logger.debug("Rendering resource for topic '%s'", topic)
    data = _load_topic(topic)
    if not data:
        return f"# No papers found for topic '{topic}'\n\nUse the `search_papers` tool first."

    lines = [
        f"# Papers on {topic.replace('_', ' ').title()}",
        "",
        f"Total papers: {len(data)}",
        "",
    ]
    for pid, meta in data.items():
        lines.extend([
            f"## {meta['title']}",
            f"- **Paper ID**: {pid}",
            f"- **Authors**: {', '.join(meta['authors'])}",
            f"- **Published**: {meta['published']}",
            f"- **PDF URL**: [{meta['pdf_url']}]({meta['pdf_url']})",
            "",
            "### Summary",
            meta['summary'][:700] + ("..." if len(meta['summary']) > 700 else ""),
            "",
            "---",
            "",
        ])
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """Prompt template instructing to process user prompt."""
    return (
        f"Translate prompt to English if it is in Korean. using"
        "`translate_kr_en(query: str)`.\n\n"
        f"Search for {num_papers} academic papers about '{topic}' using "
        f"`search_papers(topic=\"{topic}\", max_results={num_papers})`.\n\n"
        "After retrieving IDs, call `extract_info` on each to get metadata.\n"
        "Then synthesise a comprehensive overview including:\n"
        "• Key findings and methodologies\n"
        "• Trends and research gaps\n"
        "• Recommendations for future work\n\n"
        "Present results in structured markdown."
        f"Translate the final output to Korean if the input was in Korean using"
        "`translate_en_kr(query: str)`.\n\n"
    )

if __name__ == "__main__":
    mcp.run(transport='sse') 
