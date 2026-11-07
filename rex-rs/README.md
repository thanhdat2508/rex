# Rex Rust Runtime (`rex-rs`)

High-performance, encrypted, on-premise runtime for Rex AI decision and Popperian elimination models.

---

## 1. Kiến trúc Bảo mật & Nguyên tắc Zero-Disk

* **Thuật toán mã hóa:** AES-256-GCM AEAD (Authenticated Encryption with Associated Data).
* **Định dạng đóng gói (`.rexenc`):** Header magic `REXENC1\0` kèm metadata JSON kiểm chứng toàn vẹn (SHA256 checksum) và nonce 96-bit ngẫu nhiên.
* **Bảo vệ Sở hữu Trí tuệ (In-Memory IP Protection):** File mô hình chỉ được giải mã trực tiếp trên RAM. Trọng số rõ (plaintext weights) **không bao giờ được ghi xuống đĩa hoặc swap**, loại bỏ nguy cơ rò rỉ mô hình khi phân phối cho khách hàng on-premise. Khóa giải mã được xóa khỏi bộ nhớ (`Zeroize`) ngay sau khi hoàn tất nạp.

---

## 2. Công cụ Đóng gói & Mã hóa Mô hình (`rex-pack`)

CLI tool dành cho ML Ops / Model Provider để chuẩn bị các gói artifact mã hoá trước khi gửi cho khách hàng.

### 2.1. Sinh khóa AES-256 mới
```bash
cargo run --bin rex-pack -- generate-key
```

### 2.2. Mã hóa file mô hình (.onnx $\to$ .rexenc)
```bash
cargo run --bin rex-pack -- encrypt \
  --input path/to/model.onnx \
  --output ./models/model.rexenc \
  --name "rex-poe-investigator" \
  --version "1.0.0" \
  --key "<32-byte-hex-key>"
```

### 2.3. Kiểm tra metadata của gói mã hóa (Không cần khóa)
```bash
cargo run --bin rex-pack -- inspect --input ./models/model.rexenc
```

---

## 3. Triển khai On-Premise trên Server Khách hàng

### 3.1. Chạy bằng Docker Compose (Khuyến nghị)
Đặt file mô hình mã hóa vào thư mục `./models/model.rexenc` và cấu hình biến môi trường:

```bash
# Cấu hình khóa giải mã (chọn 1 trong 2 cách):
# Cách 1: Khóa tĩnh trực tiếp
export REX_MODEL_KEY="<32-byte-hex-key>"

# Cách 2: Qua License Server (Challenge Handshake)
export REX_LICENSE_KEY="<your-license-key>"
export REX_LICENSE_SERVER="https://license.yourdomain.com"

# Khởi động dịch vụ
docker compose -f compose.onprem.yml up -d --build
```

### 3.2. Chạy trực tiếp từ Binary
```bash
cargo run --release --bin rex-runtime -- \
  --model-path ./models/model.rexenc \
  --key "<32-byte-hex-key>" \
  --port 8080
```

---

## 4. Đặc tả API

### 4.1. Health Check & Trạng thái Khóa
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "rex-poe-investigator",
  "model_version": "1.0.0",
  "uptime_seconds": 124
}
```

### 4.2. Khởi chạy Điều tra Popperian (JSON)
```http
POST /api/v1/investigate
Content-Type: application/json

{
  "target_entity": "198.51.100.24",
  "initial_events": [
    {
      "timestamp_unix": 1731000100,
      "event_type": "high_volume_syn",
      "summary": "Burst of SYN traffic towards port 443",
      "source": "ids-sensor-01"
    }
  ],
  "max_rounds": 4,
  "stream": false
}
```

### 4.3. Khởi chạy Điều tra Popperian với SSE Streaming
Gửi request kèm header `Accept: text/event-stream` hoặc `"stream": true`:
```http
POST /api/v1/investigate
Accept: text/event-stream
Content-Type: application/json

{
  "target_entity": "198.51.100.24",
  "stream": true
}
```
Client sẽ nhận từng sự kiện `event: round` theo thời gian thực tương ứng với từng bước loại trừ giả thuyết Popperian ($M_j \in \{0, 1\}$).

---

## 5. Kiểm thử & Xác minh

Chạy toàn bộ unit test và integration test:
```bash
cargo test
```
