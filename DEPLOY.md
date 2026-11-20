# Hướng dẫn Git và chạy LLM local

Thực hiện các lệnh Git trong thư mục `trajectory-slicing-llm-local`, không phải
thư mục `Trajector-Slicing` ở cấp trên. Thư mục này đã được `git init` với nhánh
`main` và đã cấu hình remote
`https://github.com/nguyenddat/trajectory-slicing-llm-local.git`.
Nhánh `main` đã được đẩy lên GitHub; chỉ tạo commit mới khi có thay đổi.

## 1. Cập nhật dự án trên Git

Máy phát triển cần cài Git LFS. File
`data/gold/trace_elephant/trajectories.jsonl` (231 MB) đã được khai báo trong
`.gitattributes` để dùng LFS. Các JSONL và checkpoint nhỏ dùng Git thường.

```powershell
cd F:\Research\Trajector-Slicing\trajectory-slicing-llm-local
git lfs install --local
git add .
git lfs ls-files
git status --short
git commit -m "Cập nhật hướng dẫn và mã chạy Ollama"
git push
```

Khi `git lfs ls-files` chạy sau `git add`, nó phải liệt kê file
Trace Elephant. Chỉ chạy `git commit` khi `git status` có thay đổi. `.venv/`,
`.env`, cache Python và `.runner.lock` được bỏ qua.

Nếu cần tạo lại repo Git từ một bản copy chưa có `.git`, chạy trong đúng thư
mục dự án:

```bash
git init -b main
git remote add origin https://github.com/nguyenddat/trajectory-slicing-llm-local.git
```

Không chạy lại `git init` trên bản hiện tại.

## 2. Clone trên máy GPU Linux

Cài Git LFS, Docker, NVIDIA driver và NVIDIA Container Toolkit trên máy GPU;
xem lệnh cài và cấu hình Docker trong [README.md](README.md). Sau đó:

```bash
git lfs install
git clone https://github.com/nguyenddat/trajectory-slicing-llm-local.git
cd trajectory-slicing-llm-local
git lfs pull
wc -c data/gold/trace_elephant/trajectories.jsonl
docker build -t trajectory-slicing-llm-local .
```

Lệnh `wc -c` phải cho kết quả `231097073`, tức là dữ liệu đầy đủ thay vì LFS
pointer. Docker build cũng tự từ chối pointer nếu quên `git lfs pull`.

## 3. Kiểm tra checkpoint trước khi chạy

`outputs/` đã có checkpoint của Qwen, DeepSeek và GLM. Chạy dry run để xem số
sample còn thiếu. Lệnh này không gọi Ollama và không ghi file:

```bash
bash run-with-gpu.sh --model qwen3:32b --dry-run
```

Runner từ chối chạy nếu thiếu `index.json` hoặc thiếu file result được liệt kê
trong index. Không truyền `--allow-fresh` khi muốn tiếp tục từ checkpoint cũ.

## 4. Chạy model và lưu kết quả ra host

Với GPU NVIDIA được cấp cho Docker, chạy từng model một:

```bash
bash run-with-gpu.sh --model qwen3:32b
bash run-with-gpu.sh --model deepseek-r1:32b
bash run-with-gpu.sh --model glm-4.7-flash:q8_0
```

Script kiểm tra GPU, khởi động Ollama trong container, tự pull model vào Docker
volume `trajectory-slicing-ollama-models`, rồi chạy job. Kết quả được bind mount
vào thư mục `outputs/` của project trên máy host. Khi job kết thúc hoặc lỗi,
container Ollama được dừng; model trong volume và checkpoint trên host vẫn còn.

Thêm `--context-length N` nếu cần đổi context. Mặc định: Qwen 32768,
DeepSeek 131072, GLM 202752. Context lớn cần thêm VRAM; chế độ GPU kiểm tra
model có thực sự dùng VRAM trước khi bắt đầu xử lý sample.

## 5. Tiếp tục và đẩy kết quả về Git

Nếu job bị ngắt, chạy lại đúng lệnh model ở bước 4. Các sample `complete` được
bỏ qua; sample `failed` được thử lại. `token_limit_exceeded` chỉ được thử lại
khi tăng `--context-length`. Mỗi sample được ghi checkpoint ngay sau khi xong.

Để đồng bộ kết quả từ máy GPU về repo:

```bash
git status --short outputs/
git add outputs/
git commit -m "Cập nhật kết quả thí nghiệm LLM local"
git push
```

Chỉ cho một máy ghi và push cùng một bộ checkpoint của một model tại một thời
điểm để tránh xung đột Git và ghi đè kết quả. File `.env` không cần cho các
lệnh Ollama local và không được đưa lên Git.
