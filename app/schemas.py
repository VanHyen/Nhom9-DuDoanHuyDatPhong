"""Strict input contract derived from the supplied domain artifact."""
import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

ROOT = Path(__file__).resolve().parent.parent
BOUNDS = json.loads((ROOT / "models" / "domain_bounds.json").read_text(encoding="utf-8"))
MONTHS = {name: i + 1 for i, name in enumerate(("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"))}
# Dataset adults maximum is 55; serving deliberately applies the requested cap of 10.
API_BOUNDS = {key: {"min": max(0, value["min"]), "max": min(10, value["max"]) if key == "adults" else value["max"]} for key, value in BOUNDS["numerical"].items()}

class BookingBase(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def booking_consistency(self):
        total = self.adults + self.children + self.babies
        if not 1 <= total <= 10:
            raise ValueError("Tổng số khách (adults + children + babies) phải từ 1 đến 10; đây là giới hạn nghiệp vụ.")
        try:
            date(self.arrival_date_year, MONTHS[self.arrival_date_month], self.arrival_date_day_of_month)
        except ValueError:
            raise ValueError("Ngày đến không tồn tại trong tháng/năm đã chọn.") from None
        return self

fields = {}
for name, limits in API_BOUNDS.items():
    kind = float if name == "adr" else int
    fields[name] = (Annotated[kind, Field(ge=limits["min"], le=limits["max"], description=f"Miền API: {limits['min']:g}–{limits['max']:g}")], ...)
for name, categories in BOUNDS["categorical"].items():
    fields[name] = (Literal[tuple(categories)], ...)
BookingRequest = create_model("BookingRequest", __base__=BookingBase, **fields)
