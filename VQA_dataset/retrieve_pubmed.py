import requests
from xml.etree import ElementTree as ET
import tqdm
import pandas as pd

# Step 1: Define the search terms
search_terms = [
    "chest CT",
    "thoracic CT",
    "lung CT",
    "chest computed tomography",
    "thoracic computed tomography",
    "lung computed tomography",
    "chest scan",
    "thoracic scan",
    "lung scan"
]

# Step 2: Create a combined search query
search_query = " OR ".join([f'"{term}"' for term in search_terms])

# Step 3: Use E-utilities to search PubMed
search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={search_query}&retmax=50000&retmode=xml"
search_response = requests.get(search_url)
search_tree = ET.fromstring(search_response.content)

# Step 4: Extract article IDs
id_list = [id_elem.text for id_elem in search_tree.findall(".//Id")]

# Step 5: Fetch article details
articles = []
for article_id in tqdm.tqdm(id_list):
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={article_id}&retmode=xml"
    fetch_response = requests.get(fetch_url)
    articles.append(fetch_response.content)

# Step 6: Process and save articles to Excel files
data = []
for i, article in enumerate(articles):
    root = ET.fromstring(article)
    title_elem = root.find(".//article-title")
    title = title_elem.text if title_elem is not None else "No Title"

    # Combine section titles and text under <body> tags
    body_elems = root.findall(".//body//*")
    body = []
    for elem in body_elems:
        if elem.tag in ["sec", "title", "p"] and elem.text:
            body.append(elem.text.strip())

    body_text = "\n".join(body)

    data.append({"Title": title, "Full Text": body_text})

    # Save every 10000 articles to a new Excel file
    if (i + 1) % 10000 == 0 or i == len(articles) - 1:
        df = pd.DataFrame(data)
        file_index = (i // 10000) + 1
        df.to_excel(f"articles_{file_index}.xlsx", index=False)
        data = []

print(f"Downloaded {len(articles)} articles related to chest CT imaging")