"""Stateless inference and local log-odds explanations; no customer persistence."""
from contextlib import asynccontextmanager
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .schemas import BookingRequest, BOUNDS, API_BOUNDS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "logistic_pipeline.pkl"
LIMITATION = "Xác suất là ước lượng có sai số. Chỉ hỗ trợ ưu tiên xác nhận nội bộ; không tự động hủy, tăng giá hay phân biệt đối xử. Đóng góp là liên hệ trong mô hình, không phải nhân quả."


def load_pipeline(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(path)  # Only the trusted offline artifact, never uploads.


def public_error(status, code, message, details=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or []}})


def check_pipeline(model):
    if not isinstance(model, Pipeline) or set(model.feature_names_in_) != set(BookingRequest.model_fields):
        raise ValueError("Feature contract mismatch")
    prep = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression) or set(model.classes_) != {0, 1}:
        raise ValueError("Binary logistic classifier required")
    names = prep.get_feature_names_out()
    if len(names) != classifier.coef_.shape[1]:
        raise ValueError("Explanation contract mismatch")
    # Do not silently ignore categories present in bounds but absent from training.
    encoder = prep.named_transformers_["cat"]
    cat_cols = next(cols for name, _, cols in prep.transformers_ if name == "cat")
    trained = {col: set(values) for col, values in zip(cat_cols, encoder.categories_)}
    return list(model.feature_names_in_), names, trained


def create_app(model_loader=None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.model = None
        try:
            model = (model_loader or load_pipeline)(MODEL_PATH)
            columns, names, categories = check_pipeline(model)
            application.state.columns = columns
            application.state.feature_names = names
            application.state.categories = categories
            application.state.model = model
        except Exception:
            pass  # No raw exception, paths, request data or secrets in responses/logs.
        yield
        application.state.model = None

    application = FastAPI(title="Hotel Risk • Project 13", lifespan=lifespan)
    application.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

    @application.middleware("http")
    async def private_response(request, call_next):
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'"})
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = {"missing": "Thiếu trường bắt buộc.", "int_type": "Phải là số nguyên JSON, không phải chuỗi hoặc boolean.", "float_type": "Phải là số JSON.", "literal_error": "Không thuộc danh mục Dataset được hỗ trợ.", "extra_forbidden": "Trường không được phép gửi.", "json_invalid": "JSON không hợp lệ.", "finite_number": "Phải là số hữu hạn.", "model_attributes_type": "Nội dung phải là đối tượng JSON."}
        details = []
        for error in exc.errors():
            field = str(error["loc"][-1]) if len(error["loc"]) > 1 else "booking"
            if field not in BookingRequest.model_fields:
                field = "booking"
            kind, ctx = error["type"], error.get("ctx", {})
            message = messages.get(kind, "Dữ liệu không hợp lệ.")
            if kind in ("greater_than_equal", "less_than_equal") and field in API_BOUNDS:
                b = API_BOUNDS[field]
                message = f"{field}: giới hạn cho phép từ {b['min']:g} đến {b['max']:g}."
            elif kind == "value_error":
                message = str(ctx.get("error", "Booking không hợp lệ."))
            details.append({"field": field, "code": kind, "message": message})
        return public_error(422, "VALIDATION_ERROR", "Vui lòng kiểm tra dữ liệu đặt phòng.", details)

    @application.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8"))

    @application.get("/api/domain")
    def domain():
        return {**BOUNDS, "api_bounds": API_BOUNDS, "total_guests_max": 10, "provenance": "Notebook bước 4: thống kê toàn bộ df_clean trước split; chưa phải riêng Train."}

    @application.get("/api/model-card")
    def model_card():
        return json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))

    @application.post("/api/cancellation-risk")
    def predict(booking: BookingRequest):
        model = application.state.model
        if model is None:
            return public_error(503, "MODEL_UNAVAILABLE", "Mô hình chưa sẵn sàng. Kiểm tra artifact và phiên bản thư viện.")
        values = booking.model_dump()
        unseen = [name for name, categories in application.state.categories.items() if values[name] not in categories]
        if unseen:
            return public_error(422, "UNSEEN_CATEGORY", "Danh mục có trong Dataset nhưng chưa được encoder học từ Train.", [{"field": name, "code": "unseen_category", "message": "Danh mục chưa có trong mô hình huấn luyện."} for name in unseen])
        try:
            frame = pd.DataFrame([values], columns=application.state.columns)
            classes = list(model.classes_)
            probability = float(model.predict_proba(frame)[0, classes.index(1)])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid probability")
            classifier = model.named_steps["classifier"]
            transformed = model.named_steps["preprocessor"].transform(frame)
            x = transformed.toarray().ravel() if hasattr(transformed, "toarray") else np.asarray(transformed).ravel()
            orientation = 1 if classes[1] == 1 else -1
            beta = classifier.coef_[0] * orientation
            impacts = beta * x
            intercept = float(classifier.intercept_[0] * orientation)
            log_odds = intercept + float(impacts.sum())
            if not np.all(np.isfinite(impacts)) or not np.isclose(float(model.decision_function(frame)[0]) * orientation, log_odds, atol=1e-8):
                raise ValueError("Inconsistent explanation")
            factors = [{"feature": str(name), "contribution": float(impact), "beta": float(coef), "transformed_value": float(value)} for name, impact, coef, value in zip(application.state.feature_names, impacts, beta, x)]
            positive = sorted((f for f in factors if f["contribution"] > 0), key=lambda f: f["contribution"], reverse=True)[:5]
            negative = sorted((f for f in factors if f["contribution"] < 0), key=lambda f: f["contribution"])[:5]
            domain_warnings = [{"field": name, "value": values[name], "p99": b["p99"], "message": f"{name} vượt P99 = {b['p99']:g}; thuộc đuôi trên hiếm gặp theo artifact miền dữ liệu, cần thận trọng."} for name, b in BOUNDS["numerical"].items() if values[name] > b["p99"]]
            risk = "Thấp" if probability < .3 else "Trung bình" if probability < .7 else "Cao"
            action = "Theo dõi theo quy trình thông thường." if probability < .3 else "Ưu tiên nhắc xác nhận nhẹ nhàng; nhân viên xem xét bối cảnh." if probability < .7 else "Ưu tiên nhân viên liên hệ xác nhận; không tự động áp dụng điều kiện bất lợi."
            return {"probability": probability, "probability_display": "< 0.5%" if probability < .005 else "> 99.5%" if probability > .995 else f"{probability * 100:.1f}%", "risk_level": risk, "recommended_action": action, "threshold_decisions": [{"threshold": t, "predicted_cancellation": probability >= t} for t in (.3, .5, .7)], "domain_warnings": domain_warnings, "top_positive_factors": positive, "top_negative_factors": negative, "intercept": intercept, "log_odds": log_odds, "explanation_unit": "log-odds; không phải mức thay đổi phần trăm xác suất, không phải nhân quả", "limitation": LIMITATION}
        except Exception:
            return public_error(503, "PREDICTION_UNAVAILABLE", "Chưa thể suy luận an toàn. Vui lòng thử lại sau.")
    return application

app = create_app()
