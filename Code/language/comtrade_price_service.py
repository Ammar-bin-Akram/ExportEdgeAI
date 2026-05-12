"""UN Comtrade pricing helper for post-recommendation enrichment.

This module fetches trade rows from Comtrade and derives an estimated USD/kg
price for recommended destination markets.

Environment variables (.env):
- UN_API_KEY:            Required Comtrade subscription key
- EXPORT_ORIGIN_M49:     Optional exporter M49 code (default: 586 = Pakistan)
- COMTRADE_PERIOD:       Optional year (default: previous year)
- COMTRADE_CMD_CODE:     Optional commodity code (default: 080450)
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_env_path)
except Exception:
    # Safe fallback: use system environment only.
    pass


logger = logging.getLogger(__name__)


class ComtradePriceService:
    """Fetch estimated destination-market price from UN Comtrade rows."""

    BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

    # Exporter is always Pakistan as requested.
    PAKISTAN_M49 = "586"

    # Broad country-name mapping (canonical names + common aliases) for
    # importer partner lookup from LLM output.
    COUNTRY_TO_M49 = {
        "afghanistan": "004",
        "albania": "008",
        "algeria": "012",
        "andorra": "020",
        "angola": "024",
        "antigua and barbuda": "028",
        "argentina": "032",
        "armenia": "051",
        "australia": "036",
        "austria": "040",
        "azerbaijan": "031",
        "bahamas": "044",
        "bahrain": "048",
        "bangladesh": "050",
        "barbados": "052",
        "belarus": "112",
        "belgium": "056",
        "belize": "084",
        "benin": "204",
        "bhutan": "064",
        "bolivia": "068",
        "bolivia (plurinational state of)": "068",
        "bosnia and herzegovina": "070",
        "botswana": "072",
        "brazil": "076",
        "brunei": "096",
        "brunei darussalam": "096",
        "bulgaria": "100",
        "burkina faso": "854",
        "burundi": "108",
        "cabo verde": "132",
        "cape verde": "132",
        "cambodia": "116",
        "cameroon": "120",
        "canada": "124",
        "central african republic": "140",
        "chad": "148",
        "chile": "152",
        "china": "156",
        "colombia": "170",
        "comoros": "174",
        "congo": "178",
        "costa rica": "188",
        "cote d'ivoire": "384",
        "ivory coast": "384",
        "croatia": "191",
        "cuba": "192",
        "cyprus": "196",
        "czech republic": "203",
        "czechia": "203",
        "democratic people's republic of korea": "408",
        "north korea": "408",
        "democratic republic of the congo": "180",
        "dr congo": "180",
        "denmark": "208",
        "djibouti": "262",
        "dominica": "212",
        "dominican republic": "214",
        "ecuador": "218",
        "egypt": "818",
        "el salvador": "222",
        "equatorial guinea": "226",
        "eritrea": "232",
        "estonia": "233",
        "eswatini": "748",
        "swaziland": "748",
        "ethiopia": "231",
        "fiji": "242",
        "finland": "246",
        "france": "250",
        "gabon": "266",
        "gambia": "270",
        "georgia": "268",
        "germany": "276",
        "ghana": "288",
        "greece": "300",
        "grenada": "308",
        "guatemala": "320",
        "guinea": "324",
        "guinea-bissau": "624",
        "guyana": "328",
        "haiti": "332",
        "holy see": "336",
        "vatican": "336",
        "honduras": "340",
        "hungary": "348",
        "iceland": "352",
        "india": "356",
        "indonesia": "360",
        "iran": "364",
        "iran (islamic republic of)": "364",
        "iraq": "368",
        "ireland": "372",
        "italy": "380",
        "jamaica": "388",
        "japan": "392",
        "jordan": "400",
        "kazakhstan": "398",
        "kenya": "404",
        "kiribati": "296",
        "kuwait": "414",
        "kyrgyzstan": "417",
        "lao people's democratic republic": "418",
        "laos": "418",
        "latvia": "428",
        "lebanon": "422",
        "lesotho": "426",
        "liberia": "430",
        "libya": "434",
        "liechtenstein": "438",
        "lithuania": "440",
        "luxembourg": "442",
        "madagascar": "450",
        "malawi": "454",
        "malaysia": "458",
        "maldives": "462",
        "mali": "466",
        "malta": "470",
        "marshall islands": "584",
        "mauritania": "478",
        "mauritius": "480",
        "mexico": "484",
        "micronesia": "583",
        "micronesia (federated states of)": "583",
        "moldova": "498",
        "republic of moldova": "498",
        "monaco": "492",
        "mongolia": "496",
        "montenegro": "499",
        "morocco": "504",
        "mozambique": "508",
        "myanmar": "104",
        "namibia": "516",
        "nauru": "520",
        "nepal": "524",
        "netherlands": "528",
        "new zealand": "554",
        "nicaragua": "558",
        "niger": "562",
        "nigeria": "566",
        "north macedonia": "807",
        "norway": "578",
        "oman": "512",
        "pakistan": "586",
        "palau": "585",
        "panama": "591",
        "papua new guinea": "598",
        "paraguay": "600",
        "peru": "604",
        "philippines": "608",
        "poland": "616",
        "portugal": "620",
        "qatar": "634",
        "republic of korea": "410",
        "south korea": "410",
        "korea": "410",
        "romania": "642",
        "russian federation": "643",
        "russia": "643",
        "rwanda": "646",
        "saint kitts and nevis": "659",
        "saint lucia": "662",
        "saint vincent and the grenadines": "670",
        "samoa": "882",
        "san marino": "674",
        "sao tome and principe": "678",
        "saudi arabia": "682",
        "senegal": "686",
        "serbia": "688",
        "seychelles": "690",
        "sierra leone": "694",
        "singapore": "702",
        "slovakia": "703",
        "slovenia": "705",
        "solomon islands": "090",
        "somalia": "706",
        "south africa": "710",
        "south sudan": "728",
        "spain": "724",
        "sri lanka": "144",
        "sudan": "729",
        "suriname": "740",
        "sweden": "752",
        "switzerland": "756",
        "syrian arab republic": "760",
        "syria": "760",
        "tajikistan": "762",
        "thailand": "764",
        "timor-leste": "626",
        "east timor": "626",
        "togo": "768",
        "tonga": "776",
        "trinidad and tobago": "780",
        "tunisia": "788",
        "turkiye": "792",
        "turkey": "792",
        "turkmenistan": "795",
        "tuvalu": "798",
        "uganda": "800",
        "ukraine": "804",
        "united arab emirates": "784",
        "uae": "784",
        "united kingdom": "826",
        "uk": "826",
        "united republic of tanzania": "834",
        "tanzania": "834",
        "united states": "842",
        "usa": "842",
        "us": "842",
        "uruguay": "858",
        "uzbekistan": "860",
        "vanuatu": "548",
        "venezuela": "862",
        "venezuela (bolivarian republic of)": "862",
        "viet nam": "704",
        "vietnam": "704",
        "yemen": "887",
        "zambia": "894",
        "zimbabwe": "716",
    }

    # Region labels cannot be queried as a single partner in Comtrade.
    REGION_LABELS = {
        "eu", "europe", "middle east", "gulf", "south asia", "asia"
    }

    # High-frequency variants seen in LLM outputs.
    COUNTRY_ALIASES = {
        "u s a": "united states",
        "united states of america": "united states",
        "u s": "united states",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "scotland": "united kingdom",
        "emirates": "united arab emirates",
        "u a e": "united arab emirates",
        "korea republic of": "republic of korea",
        "korea south": "republic of korea",
        "korea north": "democratic people's republic of korea",
        "tanzania united republic of": "united republic of tanzania",
        "iran islamic republic of": "iran (islamic republic of)",
        "venezuela bolivarian republic of": "venezuela (bolivarian republic of)",
        "lao pdr": "lao people's democratic republic",
        "viet nam": "vietnam",
        "turkiye": "turkey",
    }

    FUZZY_MATCH_CUTOFF = 0.90

    def __init__(self) -> None:
        self.api_key = os.getenv("UN_API_KEY", "").strip()
        self.origin_m49 = self.PAKISTAN_M49
        self.period = os.getenv("COMTRADE_PERIOD", str(datetime.now().year - 1)).strip()
        self.cmd_code = os.getenv("COMTRADE_CMD_CODE", "080450").strip()
        self._unmapped_once = set()
        self._country_index = self._build_country_index()

    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _strip_country_label(text: str) -> str:
        # Remove bullets/numbering and text in parentheses.
        t = re.sub(r"^[-*\d\s.]+", "", (text or "")).strip()
        t = re.sub(r"\([^)]*\)", "", t).strip()
        return t

    @staticmethod
    def _normalize_country_key(text: str) -> str:
        t = (text or "").strip().lower()
        # Normalize accents and punctuation variants.
        t = unicodedata.normalize("NFKD", t)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        t = t.replace("&", " and ")
        t = t.replace("-", " ")
        t = t.replace(".", " ")
        # Keep alnum, apostrophe and spaces only.
        t = re.sub(r"[^a-z0-9' ]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t.startswith("the "):
            t = t[4:].strip()
        return t

    def _build_country_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}

        for name, code in self.COUNTRY_TO_M49.items():
            norm = self._normalize_country_key(name)
            if norm:
                index[norm] = code

            # Add parenthetical-free variants e.g. "iran (islamic republic of)" -> "iran".
            base = re.sub(r"\([^)]*\)", "", name).strip()
            base_norm = self._normalize_country_key(base)
            if base_norm:
                index.setdefault(base_norm, code)

        for alias, canonical in self.COUNTRY_ALIASES.items():
            alias_norm = self._normalize_country_key(alias)
            canonical_norm = self._normalize_country_key(canonical)
            code = index.get(canonical_norm) or self.COUNTRY_TO_M49.get(canonical)
            if alias_norm and code:
                index[alias_norm] = code

        return index

    def _resolve_country_code(self, label: str) -> Optional[str]:
        norm = self._normalize_country_key(label)
        if not norm:
            return None

        # Direct normalized lookup.
        direct = self._country_index.get(norm)
        if direct:
            return direct

        # Alias lookup.
        canonical = self.COUNTRY_ALIASES.get(norm)
        if canonical:
            canonical_norm = self._normalize_country_key(canonical)
            code = self._country_index.get(canonical_norm)
            if code:
                return code

        # Conservative fuzzy match fallback for near-exact typos.
        candidates = difflib.get_close_matches(
            norm, self._country_index.keys(), n=1, cutoff=self.FUZZY_MATCH_CUTOFF
        )
        if candidates:
            return self._country_index.get(candidates[0])

        return None

    def _country_to_m49(self, label: str) -> Optional[str]:
        norm = self._normalize_country_key(self._strip_country_label(label))
        # Trim trailing explanations after separators.
        for sep in [" - ", " — ", ":"]:
            if sep in norm:
                norm = norm.split(sep, 1)[0].strip()
        if norm in self.REGION_LABELS:
            return None

        resolved = self._resolve_country_code(norm)
        if resolved:
            return resolved

        if norm and norm not in self._unmapped_once:
            self._unmapped_once.add(norm)
            logger.warning("Comtrade country mapping missing for label: %s", label)
        return None

    def _request_rows(self, reporter_code: str, partner_code: str, flow_code: str) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        params = {
            "period": self.period,
            "reporterCode": reporter_code,
            "partnerCode": partner_code,
            "cmdCode": self.cmd_code,
            "flowCode": flow_code,
            "maxRecords": "500",
            "format_output": "JSON",
            "includeDesc": "true",
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"

        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "Ocp-Apim-Subscription-Key": self.api_key,
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw)
                data = payload.get("data", []) if isinstance(payload, dict) else []
                return [r for r in data if isinstance(r, dict)]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []

    def _price_from_rows(self, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        total_value = 0.0
        total_kg = 0.0
        value_field_used = ""

        for row in rows:
            net_wgt = self._to_float(row.get("netWgt"))
            if net_wgt <= 0:
                continue

            # Prefer landed import value, then primary, then FOB.
            value = self._to_float(row.get("cifvalue"))
            field = "cifvalue"
            if value <= 0:
                value = self._to_float(row.get("primaryValue"))
                field = "primaryValue"
            if value <= 0:
                value = self._to_float(row.get("fobvalue"))
                field = "fobvalue"
            if value <= 0:
                continue

            total_value += value
            total_kg += net_wgt
            value_field_used = field

        if total_kg <= 0:
            return None

        return {
            "price_usd_per_kg": total_value / total_kg,
            "total_value_usd": total_value,
            "total_net_weight_kg": total_kg,
            "value_field": value_field_used,
        }

    def get_price_for_country(self, country_label: str) -> Dict[str, Any]:
        """Fetch estimated USD/kg for one country label from LLM output."""
        clean_country = self._strip_country_label(country_label)
        partner_m49 = self._country_to_m49(country_label)

        if not self.api_key:
            return {
                "country": clean_country,
                "status": "error",
                "message": "UN_API_KEY missing in .env",
            }
        if not partner_m49:
            return {
                "country": clean_country,
                "status": "unavailable",
                "message": "Country/region not directly mappable to Comtrade partner code",
            }

        # 1) Importer-side query: destination imports from origin.
        rows = self._request_rows(
            reporter_code=partner_m49,
            partner_code=self.origin_m49,
            flow_code="M",
        )
        price = self._price_from_rows(rows)
        if price:
            return {
                "country": clean_country,
                "status": "success",
                "query_mode": "importer",
                "reporter_code": partner_m49,
                "partner_code": self.origin_m49,
                "period": self.period,
                "cmd_code": self.cmd_code,
                **price,
            }

        # 2) Fallback exporter-side query: origin exports to destination.
        rows = self._request_rows(
            reporter_code=self.origin_m49,
            partner_code=partner_m49,
            flow_code="X",
        )
        price = self._price_from_rows(rows)
        if price:
            return {
                "country": clean_country,
                "status": "success",
                "query_mode": "exporter",
                "reporter_code": self.origin_m49,
                "partner_code": partner_m49,
                "period": self.period,
                "cmd_code": self.cmd_code,
                **price,
            }

        return {
            "country": clean_country,
            "status": "unavailable",
            "message": "No trade rows found for selected period/commodity",
        }

    def get_prices_for_countries(self, countries: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen = set()
        for c in countries or []:
            clean = self._strip_country_label(c)
            key = clean.lower()
            if not clean or key in seen:
                continue
            seen.add(key)
            results.append(self.get_price_for_country(clean))
        return results


__all__ = ["ComtradePriceService"]
