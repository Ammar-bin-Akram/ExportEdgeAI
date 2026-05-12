"""
Quick tester for UN Comtrade Public API (free tier).

Why this script:
- Lets you test endpoint connectivity quickly.
- Shows status, response keys, and sample rows.
- Helps inspect what fields are available before integration.

Examples:
  python test_comtrade_api.py
    python test_comtrade_api.py --type-code C --freq-code A --cl-code HS --param reporterCode=586 --param partnerCode=842 --param period=2023 --param cmdCode=080450 --param flowCode=M --subscription-key YOUR_KEY
    python test_comtrade_api.py --client library --library-call previewTariffline --type-code C --freq-code M --cl-code HS --param period=202205 --param reporterCode=36 --param cmdCode=91,90 --param flowCode=M --param partnerCode=36 --param includeDesc=true --subscription-key YOUR_KEY
    python test_comtrade_api.py --endpoint releases
"""

from __future__ import annotations

import argparse
import os
import gzip
import json
import socket
import sys
import time
import zlib
from typing import Any, Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from dotenv import load_dotenv


load_dotenv()

BASE_URL = "https://comtradeapi.un.org/data/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_SUBSCRIPTION_KEY = os.getenv("UN_API_KEY")

# ----------------------------------------------------------------------------
# Default test case configuration
# Edit these values directly when you want to test a different country, period,
# commodity, or key. The script will run with no command-line parameters.
# ----------------------------------------------------------------------------
DEFAULT_CLIENT = "http"          # "http" or "library"
DEFAULT_LIBRARY_CALL = "previewTariffline"  # used only when client=library
DEFAULT_TYPE_CODE = "C"
DEFAULT_FREQ_CODE = "A"
DEFAULT_CL_CODE = "HS"
DEFAULT_PERIOD = "2024"
# Mango example: importer perspective (USA imports from Pakistan)
DEFAULT_REPORTER_CODE = "124"
DEFAULT_PARTNER_CODE = "586"
DEFAULT_PARTNER2_CODE = None
DEFAULT_CUSTOMS_CODE = None
DEFAULT_MOT_CODE = None
DEFAULT_CMD_CODE = "080450"
DEFAULT_FLOW_CODE = "M"
DEFAULT_MAX_RECORDS = 500
DEFAULT_FORMAT_OUTPUT = "JSON"
DEFAULT_COUNT_ONLY = None
DEFAULT_INCLUDE_DESC = True

KNOWN_PATHS = {
    "dataGet": "get/{typeCode}/{freqCode}/{clCode}",
    "releases": "getComtradeReleases",
    "metadata": "getMetadata",
    "preview": "preview",
    "previewTariffline": "previewTariffline",
    "da": "getDA",
    "daTariffline": "getDATariffline",
    "mbs": "getMBS",
    "worldShare": "getWorldShare",
}


def _decode_response_bytes(raw: bytes, content_encoding: str) -> str:
    """Decode HTTP body bytes, handling common compression encodings."""
    enc = (content_encoding or "").lower()
    try:
        if "gzip" in enc or raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
        elif "deflate" in enc:
            raw = zlib.decompress(raw)
    except Exception:
        # Fall back to best-effort text decoding even if decompression fails.
        pass
    return raw.decode("utf-8", errors="replace")


def _parse_key_value(items: list[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}'. Use key=value format.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --param '{item}'. Empty key is not allowed.")
        params[key] = value.strip()
    return params


def _http_get_json(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    subscription_key: str = "",
) -> Tuple[int, Any, float]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    if subscription_key and subscription_key != "REPLACE_WITH_YOUR_SUBSCRIPTION_KEY":
        headers["Ocp-Apim-Subscription-Key"] = subscription_key

    req = Request(
        url,
        headers=headers,
        method="GET",
    )
    start = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
            raw = _decode_response_bytes(
                raw_bytes,
                resp.headers.get("Content-Encoding", ""),
            )
            elapsed = time.time() - start
            try:
                return resp.status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return resp.status, {"raw": raw}, elapsed
    except HTTPError as e:
        raw_bytes = e.read()
        raw = _decode_response_bytes(
            raw_bytes,
            e.headers.get("Content-Encoding", "") if e.headers else "",
        )
        elapsed = time.time() - start
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return e.code, parsed, elapsed
    except URLError as e:
        elapsed = time.time() - start
        return 0, {"error": f"Network error: {e}"}, elapsed
    except TimeoutError:
        elapsed = time.time() - start
        return 0, {"error": "Request timed out"}, elapsed
    except socket.timeout:
        elapsed = time.time() - start
        return 0, {"error": "Request timed out"}, elapsed


def _print_response_summary(status: int, payload: Any, elapsed: float, sample_rows: int) -> None:
    print("\n=== Response Summary ===")
    print(f"Status:   {status}")
    print(f"Elapsed:  {elapsed:.2f}s")
    print(f"Type:     {type(payload).__name__}")

    if isinstance(payload, dict):
        keys = list(payload.keys())
        print(f"Top keys: {keys}")

        # Common shape used by Comtrade endpoints
        data = payload.get("data")
        if isinstance(data, list):
            print(f"Rows:     {len(data)}")
            if data:
                print("\n=== Sample Rows ===")
                for i, row in enumerate(data[:sample_rows], 1):
                    print(f"[{i}] {json.dumps(row, ensure_ascii=False)}")
        else:
            print("\n=== Payload (truncated) ===")
            print(json.dumps(payload, ensure_ascii=False)[:2000])
    elif isinstance(payload, list):
        print(f"Rows:     {len(payload)}")
        if payload:
            print("\n=== Sample Rows ===")
            for i, row in enumerate(payload[:sample_rows], 1):
                print(f"[{i}] {json.dumps(row, ensure_ascii=False)}")
    else:
        print(payload)


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _print_price_per_kg(payload: Any) -> None:
    """Print weighted average unit price (USD/kg) from returned rows."""
    rows: list[Dict[str, Any]] = []

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = [r for r in payload["data"] if isinstance(r, dict)]
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]

    if not rows:
        print("\n=== Price Calculation ===")
        print("No rows found to calculate price per kg.")
        return

    total_value = 0.0
    total_kg = 0.0

    for row in rows:
        net_wgt = _to_float(row.get("netWgt"))
        if net_wgt <= 0:
            continue

        # Prefer landed import value, then primary, then FOB as fallback.
        value = _to_float(row.get("cifvalue"))
        if value <= 0:
            value = _to_float(row.get("primaryValue"))
        if value <= 0:
            value = _to_float(row.get("fobvalue"))

        if value <= 0:
            continue

        total_value += value
        total_kg += net_wgt

    print("\n=== Price Calculation ===")
    if total_kg <= 0:
        print("Could not calculate price: no rows with positive netWgt and value.")
        return

    usd_per_kg = total_value / total_kg
    print(f"Formula: price_per_kg = total_value_usd / total_net_weight_kg")
    print(f"total_value_usd:      {total_value:,.3f}")
    print(f"total_net_weight_kg:  {total_kg:,.3f}")
    print(f"Price:                {usd_per_kg:.4f} USD/kg")


def _call_comtrade_library(
    *,
    call_name: str,
    type_code: str,
    freq_code: str,
    cl_code: str,
    params: Dict[str, str],
    subscription_key: str,
    sample_rows: int,
) -> int:
    try:
        import comtradeapicall
    except Exception:
        print("Error: comtradeapicall is not installed.")
        print("Install it with: pip install comtradeapicall")
        return 2

    func_name = "previewTarifflineData" if call_name == "previewTariffline" else "previewData"
    if not hasattr(comtradeapicall, func_name):
        print(f"Error: comtradeapicall has no function '{func_name}'.")
        return 2

    func = getattr(comtradeapicall, func_name)

    # Match the documented previewTarifflineData signature closely.
    # The free/public library call should not receive subscription_key as a kwarg.
    kwargs: Dict[str, Any] = {
        "typeCode": type_code,
        "freqCode": freq_code,
        "clCode": cl_code,
        "period": params.get("period"),
        "reporterCode": params.get("reporterCode"),
        "cmdCode": params.get("cmdCode"),
        "flowCode": params.get("flowCode"),
        "partnerCode": params.get("partnerCode"),
        "partner2Code": params.get("partner2Code", None),
        "customsCode": params.get("customsCode", None),
        "motCode": params.get("motCode", None),
        "maxRecords": int(params.get("maxRecords", "500")),
        "format_output": params.get("format_output", "JSON"),
        "countOnly": None,
        "includeDesc": str(params.get("includeDesc", "true")).lower() == "true",
    }

    # Preserve explicit None values for the parameters the library expects.
    # Only omit truly unknown/unused parameters.
    kwargs = {k: v for k, v in kwargs.items() if v is not None or k in {"partner2Code", "customsCode", "motCode", "countOnly"}}

    start = time.time()
    try:
        result = func(**kwargs)
    except Exception as e:
        elapsed = time.time() - start
        _print_response_summary(0, {"error": str(e)}, elapsed, sample_rows)
        return 1

    elapsed = time.time() - start

    # Normalize common outputs: DataFrame, dict, list.
    if hasattr(result, "to_dict"):
        try:
            rows = result.to_dict(orient="records")
            payload: Any = {"data": rows, "source": "comtradeapicall"}
        except Exception:
            payload = {"raw": str(result), "source": "comtradeapicall"}
    else:
        payload = result

    _print_response_summary(200, payload, elapsed, sample_rows)
    _print_price_per_kg(payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test UN Comtrade Public API endpoints")
    parser.add_argument(
        "--client",
        choices=["http", "library"],
        default=DEFAULT_CLIENT,
        help="Call API directly over HTTP or via comtradeapicall library",
    )
    parser.add_argument(
        "--library-call",
        choices=["preview", "previewTariffline"],
        default=DEFAULT_LIBRARY_CALL,
        help="comtradeapicall function selector when --client library",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="Base URL for Comtrade public API",
    )
    parser.add_argument(
        "--endpoint",
        choices=list(KNOWN_PATHS.keys()),
        default="dataGet",
        help="Known endpoint alias",
    )
    parser.add_argument(
        "--path",
        default="",
        help="Raw path override (e.g. get/C/A/HS, getComtradeReleases)",
    )
    parser.add_argument(
        "--type-code",
        default=DEFAULT_TYPE_CODE,
        help="Path variable for typeCode (C=commodities, S=services)",
    )
    parser.add_argument(
        "--freq-code",
        default=DEFAULT_FREQ_CODE,
        help="Path variable for freqCode (A=annual, M=monthly)",
    )
    parser.add_argument(
        "--cl-code",
        default=DEFAULT_CL_CODE,
        help="Path variable for clCode (HS, SITC, BEC, EBOPS)",
    )
    parser.add_argument(
        "--subscription-key",
        default=os.getenv("COMTRADE_SUBSCRIPTION_KEY", DEFAULT_SUBSCRIPTION_KEY),
        help="Comtrade subscription key (or set COMTRADE_SUBSCRIPTION_KEY env var)",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Query parameter in key=value format (can be repeated)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="How many rows to print from data[]",
    )

    args = parser.parse_args()

    try:
        params = _parse_key_value(args.param)
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    # Fill defaults when no CLI params are supplied.
    params.setdefault("period", DEFAULT_PERIOD)
    params.setdefault("reporterCode", DEFAULT_REPORTER_CODE)
    params.setdefault("partnerCode", DEFAULT_PARTNER_CODE if DEFAULT_PARTNER_CODE is not None else "")
    params.setdefault("partner2Code", DEFAULT_PARTNER2_CODE)
    params.setdefault("customsCode", DEFAULT_CUSTOMS_CODE)
    params.setdefault("motCode", DEFAULT_MOT_CODE)
    params.setdefault("cmdCode", DEFAULT_CMD_CODE)
    params.setdefault("flowCode", DEFAULT_FLOW_CODE)
    params.setdefault("maxRecords", str(DEFAULT_MAX_RECORDS))
    params.setdefault("format_output", DEFAULT_FORMAT_OUTPUT)
    params.setdefault("countOnly", DEFAULT_COUNT_ONLY)
    params.setdefault("includeDesc", str(DEFAULT_INCLUDE_DESC).lower())

    # Remove blank defaults from the HTTP query string, but keep None values for
    # the library call where the signature expects them.
    params = {k: v for k, v in params.items() if v not in ("", None)}

    if args.client == "library":
        print("=== Request (comtradeapicall) ===")
        print(
            f"Call: {args.library_call} | "
            f"typeCode={args.type_code}, freqCode={args.freq_code}, clCode={args.cl_code}"
        )
        return _call_comtrade_library(
            call_name=args.library_call,
            type_code=args.type_code,
            freq_code=args.freq_code,
            cl_code=args.cl_code,
            params=params,
            subscription_key=args.subscription_key,
            sample_rows=args.sample_rows,
        )

    raw_path = args.path.strip() or KNOWN_PATHS[args.endpoint]
    path = raw_path.format(
        typeCode=args.type_code,
        freqCode=args.freq_code,
        clCode=args.cl_code,
    )
    base = args.base_url.rstrip("/")
    query = urlencode(params)
    url = f"{base}/{path}"
    if query:
        url = f"{url}?{query}"

    print("=== Request ===")
    print(f"URL: {url}")

    status, payload, elapsed = _http_get_json(
        url,
        timeout=args.timeout,
        subscription_key=args.subscription_key,
    )
    _print_response_summary(status, payload, elapsed, sample_rows=args.sample_rows)
    _print_price_per_kg(payload)

    # Non-2xx should still print details but return non-zero for automation.
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
