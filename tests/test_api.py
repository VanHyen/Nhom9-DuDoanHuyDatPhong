"""Serving contract tests using synthetic bookings, never the ML Test set."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import create_app, load_pipeline, MODEL_PATH
from app.schemas import BOUNDS, BookingRequest

PAYLOAD = dict(hotel="City Hotel", lead_time=60, arrival_date_year=2016,
    arrival_date_month="July", arrival_date_week_number=29, arrival_date_day_of_month=15,
    stays_in_weekend_nights=1, stays_in_week_nights=2, adults=2, children=0, babies=0,
    meal="BB", country="PRT", market_segment="Online TA", distribution_channel="TA/TO",
    is_repeated_guest=0, previous_cancellations=0, previous_bookings_not_canceled=0,
    reserved_room_type="A", deposit_type="No Deposit", customer_type="Transient", adr=120.0,
    required_car_parking_spaces=0, total_of_special_requests=1, has_agent=1, has_company=0)

@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as instance:
        assert instance.app.state.model is not None, "Actual artifact must load successfully"
        yield instance

def test_valid_payload(client):
    response = client.post("/api/cancellation-risk", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 0 < data["probability"] < 1
    assert data["risk_level"] in ("Thấp", "Trung bình", "Cao")
    assert data["recommended_action"] and data["limitation"]
    assert data["domain_warnings"] == []
    assert len(data["threshold_decisions"]) == 3
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers

@pytest.mark.parametrize("field,value", [("adults", "2"), ("adults", True), ("lead_time", 1.5), ("adr", "120"), ("country", 123)])
def test_invalid_format(client, field, value):
    r = client.post("/api/cancellation-risk", json={**PAYLOAD, field: value})
    assert r.status_code == 422
    assert r.json()["error"]["details"]

@pytest.mark.parametrize("patch,expected", [
    ({"adults": 0, "children": 0, "babies": 0}, "Tổng số khách"),
    ({"adults": 11}, "10"), ({"adults": 8, "children": 3}, "Tổng số khách"),
    ({"lead_time": -5}, "737"), ({"lead_time": 738}, "737"),
    ({"adr": 5400.01}, "5400"), ({"adr": -1}, "5400"),
    ({"arrival_date_year": 2026}, "2017"), ({"babies": -1}, "0"),
    ({"arrival_date_month": "February", "arrival_date_day_of_month": 30}, "không tồn tại"),
    ({"stays_in_weekend_nights": 20}, "19"), ({"has_agent": 2}, "1"),
    ({"country": "INVALID"}, "danh mục")])
def test_hard_ood(client, patch, expected):
    r = client.post("/api/cancellation-risk", json={**PAYLOAD, **patch})
    assert r.status_code == 422
    assert expected in str(r.json())

def test_soft_ood_and_p99_boundary(client):
    p99 = BOUNDS["numerical"]["lead_time"]["p99"]
    r = client.post("/api/cancellation-risk", json={**PAYLOAD, "lead_time": int(p99)})
    assert r.status_code == 200 and r.json()["domain_warnings"] == []
    r = client.post("/api/cancellation-risk", json={**PAYLOAD, "lead_time": int(p99)+1, "adr": 300})
    assert r.status_code == 200
    assert {w["field"] for w in r.json()["domain_warnings"]} == {"lead_time", "adr"}
    assert all("P99" in w["message"] for w in r.json()["domain_warnings"])

@pytest.mark.parametrize("field", ["lead_time", "adr"])
def test_dataset_max_is_accepted(client, field):
    r = client.post("/api/cancellation-risk", json={**PAYLOAD, field: int(BOUNDS["numerical"][field]["max"])})
    assert r.status_code == 200
    assert field in [w["field"] for w in r.json()["domain_warnings"]]

def test_local_contributions_match_actual_model(client):
    data = client.post("/api/cancellation-risk", json=PAYLOAD).json()
    model = client.app.state.model
    frame = pd.DataFrame([PAYLOAD], columns=model.feature_names_in_)
    prep = model.named_steps["preprocessor"]
    x = prep.transform(frame)
    x = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    coef = model.named_steps["classifier"].coef_[0]
    impacts = dict(zip(prep.get_feature_names_out(), x * coef))
    for f in data["top_positive_factors"] + data["top_negative_factors"]:
        assert f["contribution"] == pytest.approx(impacts[f["feature"]])
    positives = sorted([v for v in impacts.values() if v > 0], reverse=True)[:5]
    negatives = sorted([v for v in impacts.values() if v < 0])[:5]
    assert [f["contribution"] for f in data["top_positive_factors"]] == pytest.approx(positives)
    assert [f["contribution"] for f in data["top_negative_factors"]] == pytest.approx(negatives)
    assert data["log_odds"] == pytest.approx(data["intercept"] + sum(impacts.values()))
    assert data["probability"] == pytest.approx(1/(1+np.exp(-data["log_odds"])))

def test_homepage_and_graphs(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.text.count('role="tabpanel"') == 3
    for text in ("Mẫu Thấp", "Mẫu Trung Bình", "Mẫu Cao", "Mẫu Ngoài Miền", "top_positive_factors", "domain_warnings"):
        assert text in r.text
    assert "localStorage.setItem" not in r.text and "sessionStorage" not in r.text
    for image in ("roc_pr_test_curves.png", "calibration_and_distribution.png"):
        graph = client.get("/static/"+image)
        assert graph.status_code == 200 and graph.content.startswith(b"\x89PNG")

def test_public_domain_and_model_card(client):
    d = client.get("/api/domain").json()
    assert len(d["numerical"]) == 17 and len(d["categorical"]) == 9
    assert d["numerical"]["adults"]["max"] == 55 and d["api_bounds"]["adults"]["max"] == 10
    c = client.get("/api/model-card").json()
    assert c["test"] == {"roc_auc": .8898, "average_precision": .8499, "log_loss": .3877, "brier": .1259, "n":17882}
    assert c["splits"][1]["roc_auc"] is None
    assert c["model_sha256"] == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

def test_missing_extra_malformed_and_no_value_leak(client):
    cases = [{k:v for k,v in PAYLOAD.items() if k != "adults"}, {**PAYLOAD, "email":"PRIVATE_EXAMPLE"}, {**PAYLOAD, "adults":"PRIVATE_EXAMPLE"}]
    for payload in cases:
        r=client.post("/api/cancellation-risk", json=payload)
        assert r.status_code == 422 and "PRIVATE_EXAMPLE" not in r.text
    r=client.post("/api/cancellation-risk",content='{broken PRIVATE_EXAMPLE',headers={"Content-Type":"application/json"})
    assert r.status_code == 422 and "PRIVATE_EXAMPLE" not in r.text

def test_model_loaded_once_and_invalid_input_skips_inference(monkeypatch):
    calls=[]
    def loader(path):
        calls.append(path)
        return load_pipeline(path)
    with TestClient(create_app(loader)) as c:
        model=c.app.state.model
        inference=[]
        original=model.predict_proba
        def spy(frame):
            inference.append(1)
            return original(frame)
        monkeypatch.setattr(model,"predict_proba",spy)
        assert c.post("/api/cancellation-risk",json={**PAYLOAD,"adults":100}).status_code==422
        assert inference==[]
        for _ in range(2):assert c.post("/api/cancellation-risk",json=PAYLOAD).status_code==200
        assert len(inference)==2 and len(calls)==1

@pytest.mark.parametrize("mode", ["missing","wrong","inference"])
def test_model_failures_are_private_503(mode,monkeypatch):
    def loader(path):
        if mode=="missing":raise FileNotFoundError("PRIVATE_MODEL_PATH")
        if mode=="wrong":return object()
        return load_pipeline(path)
    with TestClient(create_app(loader)) as c:
        if mode=="inference":
            def fail(frame):raise RuntimeError("PRIVATE_MODEL_PATH")
            monkeypatch.setattr(c.app.state.model,"predict_proba",fail)
        r=c.post("/api/cancellation-risk",json=PAYLOAD)
        assert r.status_code==503 and "PRIVATE_MODEL_PATH" not in r.text

@pytest.mark.parametrize("patch,display", [({"required_car_parking_spaces":8},"< 0.5%"), ({"deposit_type":"Non Refund","previous_cancellations":26},"> 99.5%")])
def test_extreme_probability_display(client,patch,display):
    r=client.post("/api/cancellation-risk",json={**PAYLOAD,**patch})
    assert r.status_code==200 and r.json()["probability_display"]==display
