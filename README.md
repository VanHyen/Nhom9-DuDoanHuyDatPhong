# Hotel Risk · Project 13

![Python](https://img.shields.io/badge/Python-3.11-2563eb) ![FastAPI](https://img.shields.io/badge/FastAPI-stateless-059669) ![ML](https://img.shields.io/badge/ML-Logistic_Regression-0f172a) [![API tests](https://github.com/VanHyen/Nhom9-DuDoanHuyDatPhong/actions/workflows/tests.yml/badge.svg)](https://github.com/VanHyen/Nhom9-DuDoanHuyDatPhong/actions/workflows/tests.yml) ![Data License](https://img.shields.io/badge/Data-CC_BY_4.0-blue)

Ứng dụng dự đoán khả năng hủy đặt phòng **ngay tại thời điểm đặt**. Bài 5: phân lớp bằng xác suất qua hồi quy Logistic. Học phần Học máy cơ bản, PGS.TS. Nguyễn Văn Hậu. Nhóm 09: **Cao Văn Hiến — 10123115**, **Lê Hồng Đăng — 10123097**.

## Mục lục

- [Cài đặt sạch](#cài-đặt-sạch)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Kiểm thử tự động](#kiểm-thử-tự-động)
- [API và OOD hai tầng](#api-và-ood-hai-tầng)
- [Giải thích cục bộ](#giải-thích-cục-bộ)
- [Thực nghiệm và nguồn số liệu](#thực-nghiệm-và-nguồn-số-liệu)
- [Quyền riêng tư và Responsible AI](#quyền-riêng-tư-và-responsible-ai)

## Cài đặt sạch

Yêu cầu Python 3.11; mở **PowerShell tại thư mục gốc dự án**. Một dòng lệnh tạo venv và cài dependencies, không cần kích hoạt môi trường:

```powershell
py -3.11 -m venv venv; if ($LASTEXITCODE -eq 0) { & .\venv\Scripts\python.exe -m pip install -r requirements.txt }
```

Pipeline mới được xuất bằng **scikit-learn 1.6.1**. Phiên bản 1.9.1 cũ không nạp được `_RemainderColsList`; requirements đã đổi về 1.6.1. Chỉ nạp artifact pickle/joblib tin cậy của nhóm. Nếu export lại trên Colab, ghi nhận đầy đủ phiên bản Python/numpy/pandas/scipy/joblib/scikit-learn và xuất requirements tương ứng; notebook hiện chưa lưu đầy đủ phiên bản môi trường export. Không có cam kết pickle tương thích chéo phiên bản.

Các tệp bắt buộc: `models/logistic_pipeline.pkl`, `models/domain_bounds.json`, `models/model_card.json`, `app/templates/index.html`, hai ảnh trong `app/static/`. Không chạy `train.py` để khởi động web: đó là công việc offline và có thể ghi đè artifact.

## Chạy ứng dụng

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Mở http://127.0.0.1:8000; API docs tại `/docs`. Ba tab gồm thuật toán, form 26 trường và dashboard/model card. Bốn mẫu minh họa có nút cố ý nhập sai để nhận **HTTP 422 từ API**. Form có min/max và thông tin Dataset/P99; `novalidate` cho phép gửi mẫu sai đến API để demo bắt lỗi phía server. Bootstrap 5 được tải từ jsDelivr, cần Internet để giao diện đầy đủ. Frontend và API cùng origin, không cần CORS mở rộng.

## Kiểm thử tự động

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_api.py
```

Kết quả kiểm tra ngày 18/10/2026: **32 passed**, một cảnh báo deprecation từ Starlette/httpx không làm thất bại kiểm thử. Đã kiểm tra trên trình duyệt cả ba mức mẫu và mẫu OOD 422.

Dùng pytest + FastAPI TestClient với lifespan. Test dùng booking tổng hợp và pipeline thật: hợp lệ 200; sai kiểu 422; hard OOD 422; GET `/` 200; soft P99 200 có cảnh báo; min/max; giải thích khớp tổng điểm tuyến tính; mô hình chỉ nạp một lần; lỗi artifact/suy luận trả 503; dữ liệu sai không gọi model; phản hồi không lộ giá trị sai; hiển thị cực trị; ảnh/model card. Không dùng tập Test học máy để chọn/tinh chỉnh mô hình.

## API và OOD hai tầng

- `GET /api/domain`: 17 biến số, 9 danh mục, giới hạn Dataset và giới hạn API.
- `GET /api/model-card`: số liệu output notebook đã lưu và checksum artifact.
- `POST /api/cancellation-risk`: đối tượng JSON đủ 26 trường. Số đếm là integer JSON, ADR là số hữu hạn; từ chối trường thừa, không tự đổi chuỗi thành số.

**Hard OOD:** chặn số âm, ngoài `[min,max]` JSON, tổng khách ngoài 1–10, adults > 10, danh mục không hợp lệ và ngày không tồn tại. Ví dụ `lead_time=738` → 422, thông báo giới hạn 0–737; `adr=5400.01` → 422, giới hạn 0–5400. `adults.max` Dataset là **55**, còn giới hạn phục vụ **10** là quy tắc nghiệp vụ bổ sung. Năm đến giới hạn 2015–2017 theo JSON, không tự mở tới năm hiện tại. Tổng đêm bằng 0 được giữ đúng miền Dataset. API cũng chặn danh mục không có trong encoder dù JSON toàn Dataset có danh mục đó.

**Soft OOD:** giá trị `> p99` và vẫn hợp lệ → HTTP 200 kèm `domain_warnings` gồm field/value/p99/message. Bằng P99 không cảnh báo. Với biến rời rạc, `> P99` không nhất thiết có đúng 1% số dòng; hợp nhiều cảnh báo cũng không suy ra booking nằm ngoài đúng 1% phân bố. Chốt chặn từng biến không phát hiện mọi tổ hợp OOD.

Phản hồi thành công có `probability`, `probability_display`, `risk_level`, `recommended_action`, `threshold_decisions`, `domain_warnings`, `top_positive_factors`, `top_negative_factors`, `intercept`, `log_odds`, `explanation_unit`, `limitation`.

Phân nhóm: p < 0.3 thấp; 0.3 ≤ p < 0.7 trung bình; p ≥ 0.7 cao. Nhãn phân lớp minh họa được trả riêng tại cả ba ngưỡng 0.3/0.5/0.7. Sai đầu vào trả JSON `error.code/message/details` tiếng Việt (422); lỗi model trả thông báo riêng tư (503).

## Giải thích cục bộ

Đóng góp từng đặc trưng sau preprocessing: **impactⱼ = βⱼ × xⱼ_transformed**. Biến số dùng giá trị StandardScaler; one-hot dùng 0/1. Tổng mọi impact cộng intercept khớp `decision_function`; sigmoid cho probability. API trả tối đa 5 đóng góp dương và 5 âm, sắp theo độ lớn; đây không phải SHAP, không phải điểm phần trăm xác suất và không phải tác động nhân quả. Các yếu tố không hiển thị vẫn thuộc tổng điểm.

OR = exp(β) của biến số tương ứng tăng một độ lệch chuẩn. Encoder giữ tất cả nhãn: đối chiếu hai nhãn cần exp(βA − βB), không giả định một nhóm tham chiếu đã bị bỏ. Không dùng hệ số quốc gia để phân biệt đối xử.

## Thực nghiệm và nguồn số liệu

`models/model_card.json` lưu số liệu đối chiếu trực tiếp output `HotelBooking_Hocmaycb.ipynb`, checksum notebook và mô hình. Giao diện lấy số liệu tại `/api/model-card`; không huấn luyện/đánh giá lại khi khởi động hoặc nhận request.

**Test mới:** ROC-AUC **0.8898**, Average Precision **0.8499**, Log Loss **0.3877**, Brier **0.1259** (N=17.882). Khác bộ KPI cũ người dùng nêu 0.8895 / 0.8505 / 0.3899 / 0.1264; không trộn kết quả giữa các lần chạy. AP không đồng nhất với diện tích PR tính bằng quy tắc hình thang. Validation Brier 0.1262.

**Minh bạch các giới hạn trong notebook hiện tại:**

1. Bước 4 tính min/P99/max và danh mục trên **toàn bộ df_clean trước split**, chưa phải riêng Train như mô tả ban đầu. JSON được giữ nguyên và nguồn được công khai. Muốn chuẩn Train-only, phải xuất lại từ `X_train`; serving không tự đọc dữ liệu thô hay tái tạo split để thay artifact.
2. Temporal chỉ được tạo với 83.446 Train và 35.763 Test theo ngày đến; chưa fit và chưa có metric Temporal. Dashboard hiển thị “Chưa có trong notebook”, không bịa số. Nhãn giai đoạn năm trong print chưa chứng minh ranh giới thực sự của phép cắt 70/30.
3. Baseline Dummy là `most_frequent`, Log Loss 13.3644; không phải `prior`.
4. Latency Logistic 0.0147 ms/mẫu, Tree 0.0005, Forest 0.0166. Dung lượng tương ứng 0.009 / 0.016 / 1.969 MB. Logistic gồm preprocessing; Tree/Forest đo classifier trên dữ liệu đã biến đổi. Đây là batch average Colab, chưa phải so sánh end-to-end hay latency HTTP.
5. Bước 8 và 9 cùng dùng Test; notebook không chứng minh Test chỉ được truy cập đúng một lần. Serving không dùng Test để tinh chỉnh.
6. Sigmoid không tự đảm bảo calibration tốt. Brier đơn lẻ không chứng minh xác suất đúng ở mọi nhóm; chia stratified không mô phỏng đầy đủ tương lai. Đánh giá sai số theo nhóm và temporal performance vẫn cần bổ sung offline.

Dashboard có baseline, ma trận 3 ngưỡng, 10 hệ số/OR, tài nguyên 3 mô hình và hai ảnh mới. Bảng Beta được đối chiếu mô hình thật; không sửa notebook, pipeline hoặc domain JSON.

Dữ liệu: António, de Almeida & Nunes, *Hotel booking demand datasets*, Data in Brief (2019), [bài báo gốc](https://doi.org/10.1016/j.dib.2018.11.126), CC BY 4.0. Dữ liệu gốc 119.390 dòng; notebook mới còn 119.209 sau làm sạch. Loại 5 biến hậu nghiệm: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`.

## Quyền riêng tư và Responsible AI

Không có database, session, cookie lưu booking, request-body log, localStorage hoặc sessionStorage. Pipeline nạp một lần **mỗi worker** bằng lifespan; dữ liệu khách chỉ ở bộ nhớ trong quá trình xử lý, không có endpoint huấn luyện. Mọi response có `Cache-Control: no-store`; lỗi không trả input thô, exception hoặc đường dẫn nội bộ. Proxy/công cụ giám sát triển khai thêm phải được cấu hình không lưu request body/query khách. Khởi động với `--no-access-log` như hướng dẫn.

Xác suất cực trị hiển thị `< 0.5%` / `> 99.5%`, giữ probability số cho xử lý. Đây là quy ước hiển thị, **không phải khoảng tin cậy**. Chỉ hỗ trợ nhân viên ưu tiên xác nhận nội bộ; không tự hủy đơn, tăng giá hoặc phân biệt đối xử. Không có API key/token/mật khẩu. Đường dẫn tài nguyên được tính tương đối từ module bằng pathlib.


## Kho mã & tái lập

Kho dự án: https://github.com/VanHyen/Nhom9-DuDoanHuyDatPhong

Pipeline tin cậy dung lượng nhỏ được đưa vào kho cùng JSON và ảnh để clone về có thể chạy test ngay. Không commit venv, cache, dữ liệu CSV hoặc bí mật. GitHub Actions tự kiểm tra trên Python 3.11 khi push/pull request.

Notebook là hồ sơ thực nghiệm chuẩn. `train.py --output-dir <thư_mục_xuất_offline>` chỉ là tiện ích huấn luyện riêng, không tái tạo mọi bảng thực nghiệm, không tự chạy khi import và từ chối ghi vào `models/` phục vụ hiện tại. Muốn cập nhật mô hình phải xuất đồng bộ pipeline, bounds, model card và kiểm tra checksum/kiểm thử.

Báo cáo Word và script báo cáo cũ vẫn giữ trên máy, được loại khỏi bản GitHub vì chưa đồng bộ notebook mới; xem `reports/README.md`. Không diễn giải phiên bản hiện tại đã đạt mọi tiêu chí thực nghiệm khi Temporal performance và bounds Train-only chưa được bổ sung.
