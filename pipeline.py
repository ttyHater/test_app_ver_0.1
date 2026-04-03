import os
import time
import sqlite3
import requests
import pandas as pd
import json
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log", encoding="utf-8")]
)
log = logging.getLogger(__name__)

# =============================
# CONFIG & ENV
# =============================
SEARCH_QUERY    = "cafe"
SEARCH_LOCATION = "Makati Philippines"
MAX_PAGES       = 3
MAX_CAFES       = 20
DATABASE_FILE   = "cafes.db"

# Each feature is a boolean. Weight = how much it contributes to the 0–10 tambay_score.
# e.g. a cafe with wifi + outlets + quiet = 0.25+0.25+0.20 = 0.70 → 7.0/10
TAMBAY_WEIGHTS = {
    "has_wifi":         0.25,
    "has_outlets":      0.25,
    "is_quiet":         0.20,
    "is_comfy":         0.10,
    "is_spacious":      0.05,
    "opens_until_late": 0.15,
}
assert abs(sum(TAMBAY_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL     = "https://maps.googleapis.com/maps/api/place/details/json"

# =============================
# DATA CLASSES
# =============================
@dataclass
class Review:
    text: str
    rating: Optional[float] = None

@dataclass
class Cafe:
    name: str
    place_id: str
    rating: float
    review_count: int
    address: str
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    reviews: List[Review] = field(default_factory=list)
    tambayable: bool = False
    tambay_score: float = 0.0
    reason: str = ""
    features: Dict = field(default_factory=lambda: {
        "has_wifi": False, "is_quiet": False, "has_outlets": False,
        "is_spacious": False, "is_comfy": False, "opens_until_late": False,
    })

# =============================
# UTILITIES
# =============================
def with_retries(fn, retries=3, delay=2, fallback=None, no_retry=()):
    """
    Call fn(); retry on Exception up to `retries` times.
    `no_retry` is a tuple of exception types that should NOT be retried.
    """
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except no_retry as e:
            log.warning(f"Non-retryable error: {type(e).__name__}: {e}")
            return fallback
        except Exception as e:
            log.warning(f"Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    log.error(f"All {retries} attempts failed. Returning fallback.")
    return fallback


def compute_tambay_score(features: Dict) -> float:
    """
    Weighted boolean score: each True feature contributes its full weight.
    Sum of all weights = 1.0, multiplied by 10 gives a 0–10 scale.
    """
    score = sum(TAMBAY_WEIGHTS.get(k, 0) for k, v in features.items() if v)
    return round(score * 10, 2)

# =============================
# SELENIUM BOOKY SCRAPER
# =============================
class BookyScraper:
    def __init__(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")

        # webdriver-manager auto-downloads the correct chromedriver version
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 15)
        self.driver.get("https://booky.ph/")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.driver.quit()

    def _parse_price(self, text: str) -> Optional[float]:
        try:
            return float(re.sub(r"[^\d.]", "", text))
        except (ValueError, TypeError):
            return None

    RESULT_SELECTORS = [
        ".search-result-tile-container",
        ".search-results .item",
        "[data-cy='search-result']",
    ]
    MENU_BTN_SELECTORS = [
        ".listing__menu-cta",
        "a[href*='menu']",
        "button.menu-btn",
    ]
    PRICE_SELECTORS = [
        ".rates-price",
        ".menu-item__price",
        ".price",
    ]
    NO_RESULTS_SELECTORS = [
        ".empty-state",
        ".no-results",
        "[class*='empty']",
        "[class*='no-result']",
    ]

    def _wait_for_any(self, selectors: list, timeout: int = 15):
        """
        Poll until one of `selectors` appears OR a no-results indicator is found.
        Raises ValueError on no-results so with_retries does NOT retry.
        """
        end = time.time() + timeout
        while time.time() < end:
            for empty_sel in self.NO_RESULTS_SELECTORS:
                if self.driver.find_elements(By.CSS_SELECTOR, empty_sel):
                    raise ValueError(f"Booky returned no results ('{empty_sel}' found)")
            for sel in selectors:
                if self.driver.find_elements(By.CSS_SELECTOR, sel):
                    return self.driver.find_elements(By.CSS_SELECTOR, sel)[0]
            time.sleep(0.5)

        raise TimeoutException(
            f"None of {selectors} found within {timeout}s. "
            f"Page title: '{self.driver.title}' | URL: {self.driver.current_url}"
        )

    @staticmethod
    def _clean_query(query: str) -> str:
        """Strip location suffixes appended by Google (e.g. 'Odd Cafe | Makati' -> 'Odd Cafe')."""
        for sep in ["|", ",", "-", "–"]:
            if sep in query:
                query = query.split(sep)[0]
        return query.strip()

    def scrape_price_range(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        clean = self._clean_query(query)
        log.info(f"  Booky search: '{clean}' (original: '{query}')")

        def _scrape():
            self.driver.get("https://booky.ph/")
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            search_box = self._wait_for_any([
                "input[name='search']",
                "input[placeholder*='Search']",
                "input[type='search']",
                ".search-input input",
                "#search",
            ])
            search_box.clear()
            search_box.send_keys(clean)
            search_box.send_keys(Keys.RETURN)

            first_result = self._wait_for_any(self.RESULT_SELECTORS, timeout=20)
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", first_result)
            first_result.click()

            menu_btn = self._wait_for_any(self.MENU_BTN_SELECTORS)
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", menu_btn)
            self.driver.execute_script("arguments[0].click();", menu_btn)

            WebDriverWait(self.driver, 15).until(
                lambda d: any(
                    d.find_elements(By.CSS_SELECTOR, sel) for sel in self.PRICE_SELECTORS
                )
            )
            self.driver.execute_script("""
                const menu = document.querySelector('.listing__menu');
                if (menu) menu.scrollTop = menu.scrollHeight;
            """)

            values = []
            for sel in self.PRICE_SELECTORS:
                els = self.driver.find_elements(By.CSS_SELECTOR, sel)
                values += [self._parse_price(e.text.strip()) for e in els if e.text.strip()]
            values = [v for v in values if v is not None]

            if not values:
                raise ValueError("No price elements found across all selectors")

            return int(min(values)), int(max(values))

        result = with_retries(_scrape, retries=2, fallback=(None, None), no_retry=(ValueError,))
        if result == (None, None):
            log.warning(f"Price scrape failed for: '{clean}'")
        return result

# =============================
# GOOGLE PLACES
# =============================
def search_google_cafes(max_cafes: int = MAX_CAFES) -> List[Dict]:
    cafes, next_page = [], None

    for page in range(MAX_PAGES):
        params = {"query": f"{SEARCH_QUERY} in {SEARCH_LOCATION}", "key": GOOGLE_API_KEY}
        if next_page:
            params["pagetoken"] = next_page
            time.sleep(2)

        def _fetch():
            return requests.get(TEXT_SEARCH_URL, params=params, timeout=10).json()

        r = with_retries(_fetch, retries=3, fallback={})
        cafes.extend(r.get("results", []))
        log.info(f"Page {page + 1}: fetched {len(r.get('results', []))} cafes (total {len(cafes)})")

        if len(cafes) >= max_cafes:
            return cafes[:max_cafes]

        next_page = r.get("next_page_token")
        if not next_page:
            break

    return cafes[:max_cafes]


def fetch_google_details(place_id: str) -> Optional[Cafe]:
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,formatted_address,reviews",
        "key": GOOGLE_API_KEY,
    }

    def _fetch():
        return requests.get(DETAILS_URL, params=params, timeout=10).json().get("result", {})

    r = with_retries(_fetch, retries=3, fallback=None)
    if not r:
        log.error(f"Could not fetch details for place_id={place_id}")
        return None

    reviews = [
        Review(text=rev.get("text", ""), rating=rev.get("rating"))
        for rev in r.get("reviews", [])
    ]
    return Cafe(
        name=r.get("name", "Unknown"),
        place_id=place_id,
        rating=r.get("rating", 0.0),
        review_count=r.get("user_ratings_total", 0),
        address=r.get("formatted_address", ""),
        reviews=reviews,
    )

# =============================
# DATABASE
# =============================
def init_db(conn: sqlite3.Connection):
    conn.execute("DROP TABLE IF EXISTS cafes")
    conn.execute("""
        CREATE TABLE cafes (
            place_id     TEXT PRIMARY KEY,
            name         TEXT,
            rating       REAL,
            review_count INTEGER,
            address      TEXT,
            min_price    INTEGER,
            max_price    INTEGER,
            tambayable   INTEGER,
            tambay_score REAL,
            reason       TEXT,
            has_wifi         INTEGER,
            has_outlets      INTEGER,
            is_quiet         INTEGER,
            is_comfy         INTEGER,
            is_spacious      INTEGER,
            opens_until_late INTEGER
        )
    """)
    conn.commit()


def upsert_cafe(conn: sqlite3.Connection, cafe: Cafe):
    f = cafe.features
    conn.execute("""
        INSERT OR REPLACE INTO cafes VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        cafe.place_id, cafe.name, cafe.rating, cafe.review_count, cafe.address,
        cafe.min_price, cafe.max_price,
        int(cafe.tambayable), cafe.tambay_score, cafe.reason,
        int(f.get("has_wifi", False)),         int(f.get("has_outlets", False)),
        int(f.get("is_quiet", False)),         int(f.get("is_comfy", False)),
        int(f.get("is_spacious", False)),      int(f.get("opens_until_late", False)),
    ))
    conn.commit()

# =============================
# LLM CLASSIFICATION
# =============================
SYSTEM_PROMPT = (
    "You are a cafe analyst specialising in work/study suitability. "
    "Return ONLY a valid JSON object — no markdown, no extra keys."
)

USER_PROMPT_TEMPLATE = """
Evaluate "{name}" for 'tambayable' suitability (good for long work/study sessions).

Reviews (truncated):
{reviews}

Return this exact JSON structure (all feature values must be boolean true/false):
{{
  "tambayable": <bool>,
  "reason": "<2–3 sentence summary referencing specific review evidence>",
  "features": {{
    "has_wifi":         <true|false>,
    "has_outlets":      <true|false>,
    "is_quiet":         <true|false>,
    "is_comfy":         <true|false>,
    "is_spacious":      <true|false>,
    "opens_until_late": <true|false>
  }}
}}
"""


def classify_cafe_with_llm(cafe: Cafe):
    if not client:
        log.warning("No OpenAI client — skipping LLM classification.")
        return

    review_text = "\n".join(
        f"- [{r.rating}★] {r.text[:300]}" for r in cafe.reviews[:5]
    ) or "No reviews available."

    prompt = USER_PROMPT_TEMPLATE.format(name=cafe.name, reviews=review_text)

    def _call():
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
        )
        return json.loads(res.choices[0].message.content)

    data = with_retries(_call, retries=2, fallback=None)
    if not data:
        cafe.reason = "LLM classification failed after retries."
        return

    cafe.tambayable   = bool(data.get("tambayable", False))
    cafe.reason       = data.get("reason", "")
    cafe.features     = data.get("features", cafe.features)
    cafe.tambay_score = compute_tambay_score(cafe.features)

    log.info(f"  -> tambayable={cafe.tambayable}, tambay_score={cafe.tambay_score}")

# =============================
# ANALYSIS HELPERS
# =============================
def build_dataframe(cafes: List[Cafe]) -> pd.DataFrame:
    rows = []
    for c in cafes:
        rows.append({
            "name":          c.name,
            "rating":        c.rating,
            "review_count":  c.review_count,
            "min_price":     c.min_price,
            "max_price":     c.max_price,
            "mid_price":     (c.min_price + c.max_price) / 2 if c.min_price and c.max_price else None,
            "tambayable":    c.tambayable,
            "tambay_score":  c.tambay_score,
            "reason":        c.reason,
            **c.features,
        })
    df = pd.DataFrame(rows)
    df["tambay_pct"] = (df["tambay_score"] / 10 * 100).round(1)
    return df


def print_summary(df: pd.DataFrame):
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE — TOP TAMBAYABLE CAFES")
    log.info("=" * 60)

    top = (
        df[df["tambayable"]]
        .sort_values("tambay_score", ascending=False)
        .head(10)[["name", "tambay_score", "rating", "mid_price", "has_wifi", "has_outlets", "opens_until_late"]]
    )
    log.info(f"\n{top.to_string(index=False)}")

    log.info("\n--- Feature Correlations with Tambay Score ---")
    feature_cols = list(TAMBAY_WEIGHTS.keys())
    corr = df[feature_cols + ["tambay_score"]].corr()["tambay_score"].drop("tambay_score")
    log.info(f"\n{corr.sort_values(ascending=False).to_string()}")

    log.info(f"\n--- Price vs Tambayable ---")
    log.info(df.groupby("tambayable")["mid_price"].describe().to_string())


def export_results(df: pd.DataFrame, path: str = "cafes_results.csv"):
    df.drop(columns=["reason"]).to_csv(path, index=False)
    log.info(f"\nResults exported → {path}")

# =============================
# PIPELINE
# =============================
def run_pipeline() -> pd.DataFrame:
    conn = sqlite3.connect(DATABASE_FILE)
    init_db(conn)

    google_results = search_google_cafes()
    log.info(f"Found {len(google_results)} cafes from Google.")

    cafes: List[Cafe] = []

    with BookyScraper() as scraper:
        for i, entry in enumerate(google_results, 1):
            name = entry.get("name", "Unknown")
            log.info(f"[{i}/{len(google_results)}] Processing: {name}")

            cafe = fetch_google_details(entry["place_id"])
            if cafe is None:
                continue

            cafe.min_price, cafe.max_price = scraper.scrape_price_range(name)
            classify_cafe_with_llm(cafe)
            upsert_cafe(conn, cafe)
            cafes.append(cafe)

    conn.close()
    df = build_dataframe(cafes)
    print_summary(df)
    export_results(df)
    return df


if __name__ == "__main__":
    df = run_pipeline()