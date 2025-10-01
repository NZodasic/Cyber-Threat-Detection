
1. **Chuẩn bị môi trường**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torchvision torchaudio   # chọn bản phù hợp CPU/GPU
   pip install flwr==1.10.3 numpy pandas scikit-learn joblib
   ```

2. **Chuẩn bị dữ liệu**

   * Đặt file `features.csv` vào thư mục `data/` (cột cuối cùng phải là `label`).
   * Hoặc bạn có thể export `X.npy` và `y.npy` từ pipeline trích xuất features của bạn.

3. **Chạy thử**

   ```bash
   bash run_example.sh 3
   ```

   * Script sẽ tự chạy **server** và 3 **clients** (giả lập trên cùng máy).
   * Bạn sẽ thấy log server in ra accuracy/F1 của global model sau mỗi round.

---

👉 Nếu bạn muốn kiểm soát bằng tay:

* Mở 1 terminal chạy:

  ```bash
  python server.py --data_csv data/features.csv --rounds 5 --num_clients 3
  ```
* Mở 3 terminal khác chạy:

  ```bash
  python client.py --cid 0 --n_clients 3 --data_csv data/features.csv --iid
  python client.py --cid 1 --n_clients 3 --data_csv data/features.csv --iid
  python client.py --cid 2 --n_clients 3 --data_csv data/features.csv --iid
  ```

---

✅ Như vậy: **đúng, chỉ cần chạy `bash run_example.sh` là bạn có demo FedAvg hoạt động trên Ubuntu** (miễn là dữ liệu và môi trường đã sẵn).

Bạn có muốn mình viết thêm một đoạn nhỏ **preprocess_csv.py** để tự động lấy `preprocessor.joblib` + `rf_pe_model.joblib` mà bạn có sẵn, rồi xuất `features.csv` chuẩn cho demo Flower không?
