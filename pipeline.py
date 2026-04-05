import os
import time
import sqlite3
import requests
import pandas as pd
import json
import re
import logging
from typing import List, Dict, Optional, Tuple, NamedTuple
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

# =============================
# CONFIG
# =============================
@dataclass
class PipelineConfig:
    search_query:    str = "cafe"
    search_location: str = "Makati Philippines"
    max_pages:       int = 3
    max_cafes:       int = 10
    database_file:   str = "cafes.db"
    tags_file:       str = "tags.json"
    driver_path:     str = ""          # empty = use webdriver-manager auto-detect nalang
    export_path:     str = "cafes_results.csv"
    google_api_key:  str = ""
    openai_api_key:  str = ""
    tambay_weights: Dict[str, float] = field(default_factory=lambda: {
        "has_wifi":         0.25,
        "has_outlets":      0.25,
        "is_quiet":         0.20,
        "is_comfy":         0.10,
        "is_spacious":      0.05,
        "opens_until_late": 0.15,
    })

    def __post_init__(self) -> None:
        total = sum(self.tambay_weights.values())
        if abs(total - 1.0) >= 1e-9:
            raise ValueError(f"tambay_weights must sum to 1.0, got {total:.4f}")


TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL     = "https://maps.googleapis.com/maps/api/place/details/json"

DEFAULT_TAGS: Dict[str, List[str]] = {
    "has_wifi": [
        r"\bwifi\b", r"\bwi-fi\b", r"\bwireless\b", r"\bfree wifi\b",
        r"\binternet\b", r"\bconnection\b",
    ],
    "has_outlets": [
        r"\boutlet\b", r"\boutlets\b", r"\bpower\b", r"\bcharging\b",
        r"\bplug\b", r"\bsocket\b",
    ],
    "is_quiet": [
        r"\bquiet\b", r"\bpeaceful\b", r"\bchill\b", r"\brelaxed\b",
        r"\bserene\b", r"\bnot (too )?noisy\b", r"\blow.?noise\b",
    ],
    "is_comfy": [
        r"\bcomfy\b", r"\bcomfortable\b", r"\bcozy\b", r"\bcosy\b",
        r"\bnice seats\b", r"\bgood seating\b",
    ],
    "is_spacious": [
        r"\bspacious\b", r"\bwide\b", r"\broomy\b", r"\blots of (seats|space|tables)\b",
        r"\bbig (place|cafe|space)\b",
    ],
    "opens_until_late": [
        r"\bopen late\b", r"\blate night\b", r"\buntil midnight\b",
        r"\b(open|closes).*\b(midnight|1[0-9]:|2[0-3]:)\b",
        r"\b24.?hour\b", r"\bopen 24\b",
    ],
}


@dataclass
class Review:
    text:   str
    rating: Optional[float] = None


@dataclass
class Cafe:
    name:         str
    place_id:     str
    rating:       float
    review_count: int
    address:      str
    reviews:      List[Review] = field(default_factory=list)
    min_price:    Optional[int] = None
    max_price:    Optional[int] = None
    tambayable:   bool  = False
    tambay_score: float = 0.0
    summary:      str   = ""
    features: Dict[str, bool] = field(default_factory=lambda: {
        "has_wifi":         False,
        "has_outlets":      False,
        "is_quiet":         False,
        "is_comfy":         False,
        "is_spacious":      False,
        "opens_until_late": False,
    })


class CachedCafe(NamedTuple):
    cafe:      Cafe
    tagged_by: str


# =============================
# UTILITIES
# =============================
def with_retries(fn, retries: int = 3, delay: int = 2, fallback=None, no_retry: tuple = ()):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except no_retry as e:
            return fallback
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * attempt)
    return fallback


def compute_tambay_score(features: Dict[str, bool], weights: Dict[str, float]) -> float:
    score = sum(weights.get(k, 0.0) for k, v in features.items() if v)
    return round(score * 10, 2)


def safe_format(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# =============================
# TAG FILE I/O
# =============================
def load_tags(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        save_tags(DEFAULT_TAGS, path)
        return DEFAULT_TAGS
    with open(path, "r", encoding="utf-8") as f:
        tags = json.load(f)
    changed = False
    for feature, defaults in DEFAULT_TAGS.items():
        if feature not in tags:
            tags[feature] = defaults
            changed = True
    if changed:
        save_tags(tags, path)
    return tags


def save_tags(tags: Dict[str, List[str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)


# =============================
# REGEX TAGGER
# =============================
def regex_tag_cafe(cafe: Cafe, tags: Dict[str, List[str]]) -> Tuple[Dict[str, bool], bool]:
    corpus = " ".join(r.text.lower() for r in cafe.reviews)
    features: Dict[str, bool] = {}
    for feature, patterns in tags.items():
        features[feature] = any(
            re.search(pattern, corpus, flags=re.IGNORECASE) for pattern in patterns
        )
    return features, any(features.values())


# =============================
# SELENIUM CHROME SCRAPER
# =============================
class BookyScraper:
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

    def __init__(self, driver_path: str = "") -> None:
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
        options.add_argument("--disable-setuid-sandbox")

        if driver_path:
            # User provided explicit path
            service = Service(driver_path)
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            # Auto-detect and download correct ChromeDriver version
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from webdriver_manager.utils import ChromeType
                
                # This auto-downloads ChromeDriver matching your installed Chrome version
                manager = ChromeDriverManager()
                chrome_driver_path = manager.install()
                service = Service(chrome_driver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
                logging.info(f"✅ ChromeDriver loaded from: {chrome_driver_path}")
                
            except ImportError:
                logging.warning("webdriver-manager not installed, falling back to system PATH")
                self.driver = webdriver.Chrome(options=options)
            except Exception as e:
                logging.error(f"webdriver-manager failed: {e}. Trying system Chrome...")
                self.driver = webdriver.Chrome(options=options)

        self.wait = WebDriverWait(self.driver, 15)
        self.driver.get("https://booky.ph/")

    def __enter__(self) -> "BookyScraper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.driver.quit()
        except:
            pass

    def _parse_price(self, text: str) -> Optional[float]:
        try:
            return float(re.sub(r"[^\d.]", "", text))
        except (ValueError, TypeError):
            return None

    def _wait_for_any(self, selectors: List[str], timeout: int = 15):
        end = time.time() + timeout
        while time.time() < end:
            for empty_sel in self.NO_RESULTS_SELECTORS:
                if self.driver.find_elements(By.CSS_SELECTOR, empty_sel):
                    raise ValueError(f"Booky returned no results ('{empty_sel}' found)")
            for sel in selectors:
                els = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if els:
                    return els[0]
            time.sleep(0.5)
        raise TimeoutException(f"None of {selectors} found within {timeout}s.")

    @staticmethod
    def _clean_query(query: str) -> str:
        for sep in ["|", ",", "-", "–"]:
            if sep in query:
                query = query.split(sep)[0]
        return query.strip()

    def scrape_price_range(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        clean = self._clean_query(query)

        def _scrape() -> Tuple[int, int]:
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
                lambda d: any(d.find_elements(By.CSS_SELECTOR, sel) for sel in self.PRICE_SELECTORS)
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
                raise ValueError("No price elements found")

            return int(min(values)), int(max(values))

        result = with_retries(_scrape, retries=2, fallback=(None, None), no_retry=(ValueError,))
        return result


# =============================
# GOOGLE PLACES
# =============================
def search_google_cafes(cfg: PipelineConfig) -> List[Dict]:
    cafes: List[Dict] = []
    next_page: Optional[str] = None

    for page in range(cfg.max_pages):
        params: Dict = {
            "query": f"{cfg.search_query} in {cfg.search_location}",
            "key":   cfg.google_api_key,
        }
        if next_page:
            params["pagetoken"] = next_page
            time.sleep(2)

        def _fetch():
            return requests.get(TEXT_SEARCH_URL, params=params, timeout=10).json()

        r = with_retries(_fetch, retries=3, fallback={})
        cafes.extend(r.get("results", []))

        if len(cafes) >= cfg.max_cafes:
            return cafes[:cfg.max_cafes]

        next_page = r.get("next_page_token")
        if not next_page:
            break

    return cafes[:cfg.max_cafes]


def fetch_google_details(place_id: str, api_key: str) -> Optional[Cafe]:
    params = {
        "place_id": place_id,
        "fields":   "name,rating,user_ratings_total,formatted_address,reviews",
        "key":      api_key,
    }

    def _fetch():
        return requests.get(DETAILS_URL, params=params, timeout=10).json().get("result", {})

    r = with_retries(_fetch, retries=3, fallback=None)
    if not r:
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
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cafes (
            place_id         TEXT PRIMARY KEY,
            name             TEXT,
            rating           REAL,
            review_count     INTEGER,
            address          TEXT,
            min_price        INTEGER,
            max_price        INTEGER,
            tambayable       INTEGER,
            tambay_score     REAL,
            summary          TEXT,
            tagged_by        TEXT,
            has_wifi         INTEGER,
            has_outlets      INTEGER,
            is_quiet         INTEGER,
            is_comfy         INTEGER,
            is_spacious      INTEGER,
            opens_until_late INTEGER
        )
    """)
    conn.commit()


def upsert_cafe(conn: sqlite3.Connection, cafe: Cafe, tagged_by: str = "unknown") -> None:
    f = cafe.features
    conn.execute(
        "INSERT OR REPLACE INTO cafes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            cafe.place_id, cafe.name, cafe.rating, cafe.review_count, cafe.address,
            cafe.min_price, cafe.max_price,
            int(cafe.tambayable), cafe.tambay_score, cafe.summary,
            tagged_by,
            int(f.get("has_wifi", False)),
            int(f.get("has_outlets", False)),
            int(f.get("is_quiet", False)),
            int(f.get("is_comfy", False)),
            int(f.get("is_spacious", False)),
            int(f.get("opens_until_late", False)),
        ),
    )
    conn.commit()


def load_cafe_from_db(conn: sqlite3.Connection, place_id: str) -> Optional[CachedCafe]:
    cursor = conn.execute("SELECT * FROM cafes WHERE place_id = ?", (place_id,))
    row = cursor.fetchone()
    if row is None:
        return None

    col_names = [d[0] for d in cursor.description]
    data = dict(zip(col_names, row))

    if data["min_price"] is None or data["max_price"] is None:
        return None

    cafe = Cafe(
        name=data["name"],
        place_id=data["place_id"],
        rating=data["rating"],
        review_count=data["review_count"],
        address=data["address"],
        min_price=data["min_price"],
        max_price=data["max_price"],
        tambayable=bool(data["tambayable"]),
        tambay_score=data["tambay_score"],
        summary=data["summary"],
        features={
            "has_wifi":         bool(data["has_wifi"]),
            "has_outlets":      bool(data["has_outlets"]),
            "is_quiet":         bool(data["is_quiet"]),
            "is_comfy":         bool(data["is_comfy"]),
            "is_spacious":      bool(data["is_spacious"]),
            "opens_until_late": bool(data["opens_until_late"]),
        },
    )
    return CachedCafe(cafe=cafe, tagged_by=data["tagged_by"])


# =============================
# LLM
# =============================
SYSTEM_PROMPT = (
    "You are a cafe analyst specialising in work/study suitability. "
    "Return ONLY a valid JSON object — no markdown, no extra keys."
)

FULL_CLASSIFICATION_PROMPT = """\
Evaluate "{name}" for 'tambayable' suitability (good for long work/study sessions).

Reviews:
{reviews}

Return this exact JSON (all feature values must be boolean):
{{
  "tambayable": <bool>,
  "summary": "<2-3 sentence summary referencing specific review evidence>",
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

VALIDATION_PROMPT = """\
Regex pattern matching flagged these features for "{name}":
{regex_features}

Reviews:
{reviews}

Your job: verify each flagged feature against the reviews.
Only correct a feature if the reviews clearly contradict the regex result.

Return this exact JSON:
{{
  "tambayable": <bool>,
  "summary": "<2-3 sentence summary referencing specific review evidence>",
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


def _llm_call(prompt: str, openai_api_key: str) -> Optional[Dict]:
    if not openai_api_key:
        return None
    client = OpenAI(api_key=openai_api_key)

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

    return with_retries(_call, retries=2, fallback=None)


def _apply_llm_result(cafe: Cafe, data: Dict, weights: Dict[str, float]) -> None:
    cafe.features     = data.get("features", cafe.features)
    cafe.summary      = data.get("summary", "")
    cafe.tambay_score = compute_tambay_score(cafe.features, weights)
    llm_verdict       = bool(data.get("tambayable", False))
    cafe.tambayable   = llm_verdict and cafe.tambay_score > 0


def classify_cafe(
    cafe: Cafe,
    tags: Dict[str, List[str]],
    weights: Dict[str, float],
    openai_api_key: str = "",
) -> str:
    review_text = "\n".join(
        f"- [{r.rating}★] {r.text[:300]}" for r in cafe.reviews[:5]
    ) or "No reviews available."

    regex_features, any_regex_match = regex_tag_cafe(cafe, tags)

    if any_regex_match:
        prompt = safe_format(
            VALIDATION_PROMPT,
            name=cafe.name,
            reviews=review_text,
            regex_features=json.dumps(regex_features, indent=2),
        )
        data = _llm_call(prompt, openai_api_key)

        if data:
            _apply_llm_result(cafe, data, weights)
            return "regex+llm_validated"

        cafe.features     = regex_features
        cafe.tambay_score = compute_tambay_score(regex_features, weights)
        cafe.tambayable   = cafe.tambay_score > 0
        return "regex"

    prompt = safe_format(
        FULL_CLASSIFICATION_PROMPT,
        name=cafe.name,
        reviews=review_text,
    )
    data = _llm_call(prompt, openai_api_key)

    if not data:
        cafe.summary = "Classification failed after retries."
        return "untagged"

    _apply_llm_result(cafe, data, weights)
    return "llm"


# =============================
# BUILD DATAFRAME
# =============================
def build_dataframe(cafes: List[Cafe], tagged_by_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cafes:
        rows.append({
            "name":         c.name,
            "rating":       c.rating,
            "review_count": c.review_count,
            "address":      c.address,
            "min_price":    c.min_price,
            "max_price":    c.max_price,
            "mid_price":    (
                (c.min_price + c.max_price) / 2
                if c.min_price is not None and c.max_price is not None else None
            ),
            "tambayable":   c.tambayable,
            "tambay_score": c.tambay_score,
            "tagged_by":    tagged_by_map.get(c.place_id, "unknown"),
            "summary":      c.summary,
            **c.features,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["tambay_pct"] = (df["tambay_score"] / 10 * 100).round(1)
    return df


# =============================
# PIPELINE (yields log lines)
# =============================
def run_pipeline(cfg: PipelineConfig):
    """
    Generator that yields (event_type, payload) tuples for live UI updates.
    event_type: "log" | "progress" | "cafe_done" | "result" | "error"
    """
    try:
        conn = sqlite3.connect(cfg.database_file)
        init_db(conn)
        tags = load_tags(cfg.tags_file)

        yield "log", "🔍 Searching Google Places..."
        google_results = search_google_cafes(cfg)
        yield "log", f"✅ Found {len(google_results)} cafes from Google."

        cafes: List[Cafe] = []
        tagged_by_map: Dict[str, str] = {}
        total = len(google_results)

        yield "log", "🚗 Starting Chrome headless scraper..."
        scraper = None
        try:
            scraper = BookyScraper(cfg.driver_path)
            
            for i, entry in enumerate(google_results, 1):
                name     = entry.get("name", "Unknown")
                place_id = entry["place_id"]

                yield "log", f"[{i}/{total}] Processing: **{name}**"
                yield "progress", (i, total)

                cached = load_cafe_from_db(conn, place_id)
                if cached is not None:
                    yield "log", f"  💾 Cache hit — loaded from DB (tagged_by={cached.tagged_by})"
                    tagged_by_map[place_id] = cached.tagged_by
                    cafes.append(cached.cafe)
                    yield "cafe_done", cached.cafe
                    continue

                cafe = fetch_google_details(place_id, cfg.google_api_key)
                if cafe is None:
                    yield "log", f"  ⚠️ Could not fetch details for {name}"
                    continue

                yield "log", f"  💰 Scraping price range from Booky..."
                cafe.min_price, cafe.max_price = scraper.scrape_price_range(name)
                price_str = (
                    f"₱{cafe.min_price}–₱{cafe.max_price}"
                    if cafe.min_price is not None else "N/A"
                )
                yield "log", f"  💰 Price: {price_str}"

                yield "log", f"  🤖 Classifying features..."
                method = classify_cafe(cafe, tags, cfg.tambay_weights, cfg.openai_api_key)
                tagged_by_map[cafe.place_id] = method

                score_emoji = "✅" if cafe.tambayable else "❌"
                yield "log", (
                    f"  {score_emoji} Tambayable={cafe.tambayable} | "
                    f"Score={cafe.tambay_score} | Method={method}"
                )

                upsert_cafe(conn, cafe, tagged_by=method)
                cafes.append(cafe)
                yield "cafe_done", cafe
        
        except Exception as scraper_error:
            yield "log", f"⚠️ ChromeDriver error: {scraper_error}"
            yield "log", f"⚠️ Proceeding without price scraping (Booky disabled)..."
            yield "log", f"⚠️ Prices will show as N/A, but analysis will continue"
            
            # Continue processing without scraper
            for i, entry in enumerate(google_results, 1):
                name     = entry.get("name", "Unknown")
                place_id = entry["place_id"]

                yield "log", f"[{i}/{total}] Processing: **{name}** (no scraper)"
                yield "progress", (i, total)

                cached = load_cafe_from_db(conn, place_id)
                if cached is not None:
                    yield "log", f"  💾 Cache hit"
                    tagged_by_map[place_id] = cached.tagged_by
                    cafes.append(cached.cafe)
                    yield "cafe_done", cached.cafe
                    continue

                cafe = fetch_google_details(place_id, cfg.google_api_key)
                if cafe is None:
                    yield "log", f"  ⚠️ Could not fetch details"
                    continue

                yield "log", f"  🤖 Classifying features..."
                method = classify_cafe(cafe, tags, cfg.tambay_weights, cfg.openai_api_key)
                tagged_by_map[cafe.place_id] = method

                score_emoji = "✅" if cafe.tambayable else "❌"
                yield "log", f"  {score_emoji} Score={cafe.tambay_score} | Method={method}"

                upsert_cafe(conn, cafe, tagged_by=method)
                cafes.append(cafe)
                yield "cafe_done", cafe
        
        finally:
            if scraper is not None:
                scraper.close()

        conn.close()
        df = build_dataframe(cafes, tagged_by_map)
        df.to_csv(cfg.export_path, index=False)
        yield "log", f"✅ Pipeline complete! Results saved to `{cfg.export_path}`"
        yield "result", df

    except Exception as e:
        yield "error", str(e)