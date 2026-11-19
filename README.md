# Chạy thí nghiệm trajectory slicing với Ollama local

Đây là bản rút gọn của `trajectory-slicing-code` để chạy Qwen, DeepSeek và GLM
trên máy GPU. Cấu trúc module trong `src/`, dữ liệu `data/gold/`, cấu hình thí
nghiệm và định dạng kết quả tương thích với project gốc. Một runner duy nhất
nhận tên model và URL Ollama qua tham số.

Xem [DEPLOY.md](DEPLOY.md) để biết cách dùng Git, Git LFS, clone lên máy GPU và
đẩy kết quả về GitHub.

## Chuẩn bị trên máy GPU Linux

Cài Docker và Git LFS. Nếu chạy Ollama trong Docker với GPU NVIDIA, cần cài
NVIDIA Container Toolkit và cấu hình Docker runtime. Chạy các lệnh sau ngay
trên máy GPU để bind mount trỏ đến thư mục `outputs/` của máy đó:

```bash
git lfs pull
docker build -t trajectory-slicing-llm-local .
bash run-with-gpu.sh --model qwen3:32b --dry-run
```

Dry run đọc dữ liệu và checkpoint hiện có, in số lượt còn phải chạy cho từng
cấu hình, không gọi model và không ghi file. Runner sẽ báo lỗi nếu thiếu
`index.json` hoặc một file kết quả được liệt kê trong index, nhờ đó không vô
tình chạy lại từ đầu khi mount sai. Chỉ dùng `--allow-fresh` khi chủ ý tạo thí
nghiệm mới không có checkpoint.

## Cách 1: Ollama chạy trong Docker với GPU NVIDIA

Chạy từng model một:

```bash
bash run-with-gpu.sh --model qwen3:32b
bash run-with-gpu.sh --model deepseek-r1:32b
bash run-with-gpu.sh --model glm-4.7-flash:q8_0
```

Script kiểm tra `--gpus all` bằng `nvidia-smi`, khởi động image Ollama chính
thức, tải model vào Docker volume lưu lâu dài, rồi kiểm tra model có dùng VRAM
ở context đã chọn trước khi chạy thí nghiệm. Nếu model chỉ chạy trên CPU,
script dừng và báo lỗi. Nếu model dùng cả RAM lẫn VRAM, script in mức sử dụng
và tiếp tục chạy. Khi job kết thúc, container Ollama
được dừng nhưng model đã tải và kết quả vẫn còn.

## Cách 2: Ollama chạy trên máy host

Nếu Ollama đã chạy trực tiếp trên máy GPU, tải model và dùng runner container:

```bash
ollama pull qwen3:32b
bash run-docker.sh --model qwen3:32b --url http://localhost:11434
```

Nếu Ollama chưa chạy, khởi động `ollama serve` trong terminal khác hoặc bằng
system service. `run-docker.sh` dùng Docker host networking để truy cập
`localhost:11434`. Container Python không cần gắn GPU vì Ollama trên host thực
hiện inference.

## Kết quả và tiếp tục khi bị ngắt

Hai script đều bind mount thư mục `outputs/` của project vào `/app/outputs`
trong container. Kết quả được lưu sau mỗi sample tại:

```text
outputs/experiments/e001_latency_accuracy_tradeoff/<model>/predictions/
```

Project đã có checkpoint của cả ba model. Chạy lại cùng lệnh để tiếp tục:
sample `complete` được bỏ qua, sample `failed` được thử lại. Sample
`token_limit_exceeded` chỉ được thử lại khi tăng `--context-length`. Nếu Ollama
mất kết nối giữa chừng, job dừng và giữ checkpoint hiện có. Mỗi model có khóa
để tránh hai runner cùng ghi một bộ kết quả. Container tự thoát khi quét xong;
nếu vẫn còn sample `failed`, lệnh trả mã lỗi khác 0.

Tham số `--context-length N` ghi đè mặc định: Qwen `32768`, DeepSeek `131072`,
GLM `202752`; model khác dùng `32768`. Context lớn cần thêm VRAM. Tên model
phải khớp chính xác với model có trong Ollama.

## Dữ liệu và Git LFS

Các file trong `data/gold/` là toàn bộ dữ liệu đầu vào cần thiết. File Trace
Elephant 231 MB được quản lý bằng Git LFS; các JSONL và checkpoint nhỏ dùng Git
thường. Sau khi clone, chạy `git lfs pull` trước khi build image. Docker build
sẽ từ chối nếu file lớn vẫn chỉ là LFS pointer.

Không cần `.env` hay thông tin xác thực Cloud Run để chạy Ollama local. Các cấu
hình slicing dùng OpenAI embeddings bị bỏ qua, giống các runner model hiện có.

Nếu không dùng Docker, cài Python 3.12+, `uv` và các gói trong
`requirements.txt`, tải NLTK stopwords, rồi gọi `src/run_local.py` với cùng
tham số `--model` và `--url`.
