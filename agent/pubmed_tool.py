import requests

class PubMedSearchTool:
    """
    MCP Tool for searching PubMed articles by query.
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def __call__(self, query: str) -> dict:
        """
        Args:
            query (str): Search term for PubMed
            max_results (int, optional): Number of results to return
        
        Returns:
            dict: {
                "titles": List of article titles,
                "ids": List of article IDs,
                "query": Original search query,
            }
        """
        max_results = self.max_results

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        ids = resp.json()["esearchresult"]["idlist"]

        if not ids:
            return {"titles": [], "ids": [], "query": query, "message": "No PubMed results found."}

        # Fetch summaries for the found IDs
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json"
        }
        summary_resp = requests.get(summary_url, params=summary_params)
        summary_resp.raise_for_status()
        summaries = summary_resp.json()["result"]

        # Collect titles
        titles = []
        for pid in ids:
            title = summaries.get(pid, {}).get("title", "No title")
            titles.append(title)

        return {"titles": titles, "ids": ids, "query": query}
    
