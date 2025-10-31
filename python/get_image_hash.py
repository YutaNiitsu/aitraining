import hashlib

def get_image_hash(image):
    # 画像のハッシュ値を取得（RGB配列 → bytes → MD5）
    return hashlib.md5(image.tobytes()).hexdigest()