# Setup Environment + Kaggle Access

Tài liệu này chỉ tập trung phần chuẩn bị môi trường chạy project.

## 1. Tạo virtual environment

Chạy trong thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra nhanh:

```powershell
python --version
pip --version
```

## 2. Chuẩn bị Kaggle API token

1. Đăng nhập Kaggle.
2. Vào `Settings` -> `API Tokens`.
3. Bấm `Create Legacy API Key` để tải file `kaggle.json`.
4. Tạo thư mục token (nếu chưa có):

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
```

5. Copy file token vào đúng chỗ:

```powershell
copy "D:\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json" -Force
```

Ví dụ trên dùng đúng đường dẫn bạn đang có. Nếu file ở chỗ khác thì đổi lại path nguồn.

## 3. Tải dataset CUB-200-2011

### Cách A: tải bằng Kaggle API

```powershell
python scripts/download_cub.py --force
```

### Cách B: bạn đã tải sẵn zip từ Kaggle

```powershell
python scripts/download_cub.py --zip "D:\path\to\cub2002011.zip" --force
```

## 4. Kết quả mong đợi

Sau khi tải và giải nén xong, dataset nằm ở:

```text
data/raw/cub2002011
```

Bạn có thể kiểm tra:

```powershell
Get-ChildItem data/raw/cub2002011
```
