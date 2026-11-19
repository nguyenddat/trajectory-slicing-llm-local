"""Small opt-in Google Sheets sink for per-sample runner results."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from urllib.parse import quote


HEADERS = [
    "timestamp_utc", "model", "config_id", "sample_key", "source", "index",
    "status", "step_accuracy", "agent_accuracy", "prediction", "slice_ranges", "error",
]


class GoogleSheetsSink:
    """Append one result row per completed sample using Google Sheets API v4."""

    def __init__(self, spreadsheet_id: str, worksheet: str):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        self.session = AuthorizedSession(credentials)
        self.spreadsheet_id = spreadsheet_id
        self.worksheet = worksheet
        self.base_url = (
            "https://sheets.googleapis.com/v4/spreadsheets/"
            + quote(spreadsheet_id, safe="")
        )
        self._ensure_header()
        self._existing_keys = self._read_existing_keys()

    def _ensure_header(self) -> None:
        range_name = quote(f"'{self.worksheet}'!A1:L1", safe="!:$'")
        response = self.session.get(f"{self.base_url}/values/{range_name}", timeout=30)
        response.raise_for_status()
        values = response.json().get("values", [])
        if not values:
            self._request(
                "post",
                f"{self.base_url}/values/{range_name}:append",
                params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
                json={"values": [HEADERS]},
            )

    def _read_existing_keys(self) -> set[tuple[str, str, str]]:
        # Columns B-D uniquely identify a result across model/config/sample.
        range_name = quote(f"'{self.worksheet}'!B2:D", safe="!:$'")
        response = self.session.get(f"{self.base_url}/values/{range_name}", timeout=30)
        response.raise_for_status()
        keys = set()
        for row in response.json().get("values", []):
            padded = (row + [""] * 3)[:3]
            if all(str(value).strip() for value in padded):
                keys.add(tuple(str(value) for value in padded))
        return keys

    def append(self, *, model: str, config_id: str, sample_key: str, sample: dict) -> None:
        unique_key = (model, config_id, sample_key)
        if unique_key in self._existing_keys:
            return
        metrics = sample.get("metrics") or {}
        row = [
            datetime.now(timezone.utc).isoformat(), model, config_id, sample_key,
            sample.get("source", ""), sample.get("index", ""),
            sample.get("status", ""), metrics.get("step_accuracy", ""),
            metrics.get("agent_accuracy", ""), _json_cell(sample.get("prediction")),
            _json_cell(sample.get("slice_ranges")), _json_cell(sample.get("error")),
        ]
        range_name = quote(f"'{self.worksheet}'!A:L", safe="!:$'")
        self._request(
            "post",
            f"{self.base_url}/values/{range_name}:append",
            params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
            json={"values": [row]},
        )
        self._existing_keys.add(unique_key)

    def _request(self, method: str, url: str, **kwargs):
        response = getattr(self.session, method)(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response


def create_sink() -> GoogleSheetsSink | None:
    """Create the sink when configured; Sheet failures never stop inference."""
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    if not spreadsheet_id:
        return None
    worksheet = os.getenv("GOOGLE_SHEETS_WORKSHEET", "Results").strip() or "Results"
    try:
        return GoogleSheetsSink(spreadsheet_id, worksheet)
    except Exception as error:  # noqa: BLE001
        warnings.warn(
            "Google Sheets sync disabled: "
            f"{error}. For local runs, gcloud auth login is not enough; set up "
            "Application Default Credentials with the Sheets scope, or set "
            "GOOGLE_APPLICATION_CREDENTIALS to a credential file. See "
            "GOOGLE_SHEETS.md.",
            stacklevel=2,
        )
        return None


def _json_cell(value) -> str:
    return "" if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
