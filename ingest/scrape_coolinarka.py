"""
Scraper za coolinarika.com koji koristi stranicu /jela/
- Preuzima kategorije jela
- Iz svake kategorije uzima recepte
- Dodaje klasifikaciju (kategorija) u metadata
- Sprema u format kompatibilan sa RAG pipeline-om
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_storage import save_documents
from chunking import chunk

BASE_URL = "https://www.coolinarika.com"
START_URL = "https://www.coolinarika.com/jela/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CookBot/1.0)"
}

MAX_RECIPES_PER_CATEGORY = 5
MAX_CATEGORY = 20


def get_categories():
    """Prikuplja sve kategorije jela koristeći Coolinarika API endpoint."""
    api_url = "https://api.coolinarika.com/api/v1/feed/jela/1st-level?query=%7B%7D&page=1"

    print(f"Loading categories from API: {api_url}")
    response = requests.get(api_url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    data = response.json()
    categories = []

    # API vraća listu objekata
    for item in data:
        title = item.get("title")
        slug = item.get("slug")
        item_type = item.get("type")

        if item_type != "food" or not title or not slug:
            continue

        categories.append({
            "name": title,
            "url": f"{BASE_URL}/jela/{slug}/"
        })
        if len(categories)>=MAX_CATEGORY:
            return categories

    print(f"Detected {len(categories)} categories")
    return categories


def get_recipe_links(category_url: str):
    print(f"Scraping category: {category_url}")

    category_url = category_url.replace("www.coolinarika.com", "web.coolinarika.com")

    response = requests.get(category_url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/recept/" in href:
            full_url = urljoin("https://www.coolinarika.com", href)
            links.append(full_url)

    unique_links = list(dict.fromkeys(links))[:MAX_RECIPES_PER_CATEGORY]

    print(f"Found {len(unique_links)} recipes")
    return unique_links



def parse_recipe(url: str, category_name: str):
    recipe_url = url.replace("www.coolinarika.com", "web.coolinarika.com")
    response = requests.get(recipe_url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else "Nepoznat recept"

    # =====================
    # SASTOJCI
    # =====================

    ingredients = []

    sastojci_header = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Sastojci" in tag.get_text())

    if sastojci_header:
        container = sastojci_header.find_next("div", class_="groupItems")
        if container:
            for line in container.get_text("\n", strip=True).split("\n"):
                line = line.strip()
                if line:
                    ingredients.append(line)
                    print(line)

    # ingredients = []

    # for item in soup.select("div.groupItems"):
    #     text = item.get_text(separator=" ", strip=True)
    #     if text:
    #         if text.strip().startswith((" 1 . ", "2 . ", " 3 . ", " 4 . ", " 5 . ")):
    #             continue
    #         print(text)
    #         ingredients.append(text)
            

    # =====================
    # PRIPREMA
    # =====================
    steps = []

    for step in soup.select("div.stepContent_description p"):
        text = step.get_text(strip=True)
        if text:
            steps.append(text)
            print(text)
    # =====================
    # SADRŽAJ
    # =====================
    content = f"Recept: {title}\nKategorija: {category_name}\n\n"

    if ingredients:
        content += "Sastojci:\n" + "\n".join(ingredients) + "\n\n"
    else:
        content += "Sastojci: Nisu pronađeni\n\n"

    if steps:
        content += "Priprema: " + " ".join(dict.fromkeys(steps))
    else:
        content += "Priprema: Nije pronađena"

    return {
        "id": f"coolinarka_{hash(url)}",
        "title": title,
        "content": content,
        "source": url,
        "type": "recipe",
        "category": category_name,
        "scraped_at": datetime.now().isoformat()
    }

def scrape_coolinarka_jela():
    """Glavna funkcija za scrape svih kategorija jela."""
    all_docs = []
    categories = get_categories()

    print(f"Found {len(categories)} categories")

    for cat in categories:
        name = cat["name"]
        url = cat["url"]

        print(f"CATEGORY: {name}")

        try:
            recipe_links = get_recipe_links(url)
        except Exception as e:
            print(f"Failed to load category {name}: {e}")
            continue

        for link in recipe_links:
            try:
                docs = parse_recipe(link, name)
                all_docs.append(docs)
                time.sleep(1)
            except Exception as e:
                print(f"Failed to parse {link}: {e}")

    print(f"Saving {len(all_docs)} documents...")
    save_documents(all_docs)
    print("Done. Cook-bot baza je spremna.")


if __name__ == "__main__":
    scrape_coolinarka_jela()