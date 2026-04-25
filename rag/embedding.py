import numpy as np
from threading import Lock


model_name = "vinai/phobert-base-v2"
model = None
_model_lock = Lock()


def _load_model_once():
    """
    Lazy-load model để tránh import nặng khi chỉ vừa khởi động API.
    Model sẽ chỉ nạp khi có truy vấn embedding đầu tiên.
    """
    global model
    if model is not None:
        return model

    with _model_lock:
        if model is not None:
            return model

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Đang tải mô hình embedding. Quá trình này có thể mất vài phút cho lần đầu tiên...")
            print(f"Sử dụng thiết bị: {device}")
            loaded_model = SentenceTransformer(model_name, device=device)

            # Đối với PhoBERT-base, văn bản dài hơn 256 token sẽ bị cắt.
            loaded_model.max_seq_length = 256

            print("✓ Tải mô hình thành công.")
            print(f"✓ Đã thiết lập max_seq_length cho mô hình là {loaded_model.max_seq_length}")
            model = loaded_model
            return model
        except Exception as e:
            raise RuntimeError(
                f"Không thể tải mô hình embedding '{model_name}': {e}. "
                "Vui lòng kiểm tra kết nối internet, cache model và môi trường Python."
            ) from e
   
def get_model():
    """
    Trả về instance của mô hình embedding đã được tải.
    Hàm này đảm bảo rằng mô hình chỉ được tải một lần.
    """
    return _load_model_once()

def get_embedding(text: str):
    """
    Hàm này nhận một đoạn văn bản (string) và trả về vector embedding của nó.
    """
    embedding_model = get_model()
    
    # Sử dụng mô hình để mã hoá văn bản
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    #normalize_embeddings=True sẽ đảm bảo rằng các vector embedding được chuẩn
    #hóa về đơn vị độ dài 1, giúp cải thiện hiệu suất tìm kiếm tương đồng.
    return embedding

def get_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Tạo embeddings cho một danh sách các đoạn văn bản.
    
    Args:
        texts (list[str]): Danh sách các đoạn văn bản cần mã hóa.
        batch_size (int): Số lượng văn bản xử lý trong một lần để tối ưu VRAM.

    Returns:
        np.ndarray: Một mảng numpy chứa các vector embedding.
    """
    embedding_model = get_model()
    if embedding_model is None:
        return np.array([])
        
    # show_progress_bar=True để hiển thị thanh tiến trình
    print(f"Bắt đầu tạo embedding cho {len(texts)} văn bản với batch size = {batch_size}...")
    embeddings = embedding_model.encode(
    texts, 
    batch_size=batch_size, 
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
    print("✓ Hoàn tất tạo embedding.")
    return embeddings

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # Câu ví dụ
    example_sentence = "Mức phạt khi vượt đèn đỏ là bao nhiêu?"
    
    # Lấy embedding cho một câu
    try:
        single_embedding = get_embedding(example_sentence)
        print(f"Vector embedding cho câu đơn lẻ có chiều: {single_embedding.shape}")

        # Danh sách câu ví dụ
        example_sentences = [
            "Người đi bộ có được đi trên đường cao tốc không?",
            "Quy định về tốc độ tối đa trong khu dân cư.",
            "Nồng độ cồn cho phép khi lái xe ô tô là gì?"
        ]
        
        # Lấy embedding cho một danh sách câu
        multiple_embeddings = get_embeddings(example_sentences)
        print(f"Vector embedding cho danh sách 3 câu có shape: {multiple_embeddings.shape}")

    except RuntimeError as e:
        print(e)

