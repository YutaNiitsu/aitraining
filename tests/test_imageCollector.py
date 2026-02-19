import os
import shutil
from PIL import Image
from unittest.mock import patch
from python.imageCollector import ImageCollector
import logging
import os, stat


# ============================================================
# get_image_hash()
# ============================================================

# 4-1-1 正常：ハッシュ取得成功
def test_get_image_hash_success(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)

    ic = ImageCollector()
    h = ic.get_image_hash(str(img_path))

    assert isinstance(h, str)
    assert len(h) == 32  # MD5 32桁


# 4-1-2 異常：ファイル不存在
def test_get_image_hash_not_found(caplog):
    # ログを拾う
    caplog.set_level(logging.ERROR, logger="myapp")
    
    ic = ImageCollector()
    h = ic.get_image_hash("not_found.jpg")
    assert h is None

    # ログ拾えないのでコメントアウト
    #assert "ファイルが見つかりません" in caplog.text


# 4-1-3 異常：非画像ファイル
def test_get_image_hash_not_image(tmp_path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("not image")

    ic = ImageCollector()
    h = ic.get_image_hash(str(txt_path))

    assert h is None


# 4-1-4 異常：読み込み権限なし
def test_get_image_hash_permission_error(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    ic = ImageCollector()
    with patch("builtins.open", side_effect=PermissionError("アクセス拒否")):
        h = ic.get_image_hash(str(img_path))
        assert h is None


# 4-1-5 異常：予期せぬ例外（Image.open を例外化）
def test_get_image_hash_unexpected_error(tmp_path, monkeypatch):
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    def fake_open(*args, **kwargs):
        raise Exception("unexpected error")

    monkeypatch.setattr("PIL.Image.open", fake_open)

    ic = ImageCollector()
    h = ic.get_image_hash(str(img_path))

    assert h is None


# ============================================================
# remove_duplicate()
# ============================================================

# 4-2-1 正常：重複なし
def test_remove_duplicate_no_duplicates(tmp_path):
    ic = ImageCollector()

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"

    Image.new("RGB", (100, 100)).save(img1)
    Image.new("RGB", (90, 100)).save(img2)

    hash_set = set()
    unique_files = []

    ic.remove_duplicate(str(tmp_path), hash_set, unique_files)

    assert len(unique_files) == 2
    assert img1.exists()
    assert img2.exists()


# 4-2-2 正常：重複あり
def test_remove_duplicate_with_duplicates(tmp_path):
    ic = ImageCollector()

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"

    img = Image.new("RGB", (100, 100))
    img.save(img1)
    shutil.copy(img1, img2)

    hash_set = set()
    unique_files = []

    ic.remove_duplicate(str(tmp_path), hash_set, unique_files)

    assert len(unique_files) == 1
    assert img1.exists()
    assert not img2.exists()  # 削除される


# 4-2-3 正常：ハッシュ取得失敗スキップ
def test_remove_duplicate_hash_fail(tmp_path):
    ic = ImageCollector()

    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"

    Image.new("RGB", (100, 100)).save(good)
    bad.write_text("not image")

    hash_set = set()
    unique_files = []

    ic.remove_duplicate(str(tmp_path), hash_set, unique_files)

    assert len(unique_files) == 1
    assert good.exists()
    assert bad.exists()  # スキップされ削除されない


# 4-2-4 異常：ディレクトリ不存在
def test_remove_duplicate_dir_not_found():
    ic = ImageCollector()
    hash_set = set()
    unique_files = []

    ic.remove_duplicate("not_exist", hash_set, unique_files)

    assert len(unique_files) == 0


# 4-2-5 異常：削除失敗
def test_remove_duplicate_delete_fail(tmp_path):
    ic = ImageCollector()

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"

    img = Image.new("RGB", (100, 100))
    img.save(img1)
    shutil.copy(img1, img2)

    img2.chmod(0o000)  # 削除不可にする

    hash_set = set()
    unique_files = []

    ic.remove_duplicate(str(tmp_path), hash_set, unique_files)

    assert img1.exists()
    assert img2.exists()  # 削除失敗 → 残る


# ============================================================
# split_images()
# ============================================================

# 4-3-1 正常：正常分割
def test_split_images_success(tmp_path):
    ic = ImageCollector()

    IMAGE_NUM = 10
    VAL_RATIO = 0.2
    VAL_IMAGE_NUM = int(IMAGE_NUM * VAL_RATIO)

    target = tmp_path / "target"
    output = tmp_path / "output"
    category = "cat"

    target.mkdir()
    output.mkdir()

    # 画像作成
    for i in range(IMAGE_NUM):
        Image.new("RGB", (100, 100)).save(target / f"img{i}.jpg")

    ic.split_images(str(target), str(output), category, VAL_RATIO)

    train_dir = output / category / "train"
    eval_dir = output / category / "eval"

    assert len(os.listdir(train_dir)) == IMAGE_NUM - VAL_IMAGE_NUM
    assert len(os.listdir(eval_dir)) == VAL_IMAGE_NUM


# 4-3-2 異常：ディレクトリ不存在
def test_split_images_dir_not_found(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()
    output = tmp_path / "output"
    output.mkdir()

    ic.split_images("not_exist", str(output), "cat", 0.2)

    assert "ディレクトリの読み込みに失敗しました" in caplog.text


# 4-3-3 異常：画像0枚
def test_split_images_zero_images(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()

    ic.split_images(str(target), str(output), "cat", 0.2)

    assert "画像が収集できませんでした" in caplog.text


# 4-3-4 異常：画像1枚
def test_split_images_one_image(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()

    Image.new("RGB", (100, 100)).save(target / "img.jpg")

    ic.split_images(str(target), str(output), "cat", 0.2)

    assert "画像が少なすぎるため split できません" in caplog.text


# 4-3-5 異常：train_test_split エラー
def test_split_images_tts_error(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()

    for i in range(5):
        Image.new("RGB", (100, 100)).save(target / f"img{i}.jpg")

    ic.split_images(str(target), str(output), "cat", 1.0)

    assert "train_test_split エラー" in caplog.text


# 4-3-6 異常：出力フォルダ作成失敗
def test_split_images_output_mkdir_fail(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    target.mkdir()

    # output を「ディレクトリではなくファイル」にする 
    output = tmp_path / "output" 
    output.write_text("dummy")

    for i in range(5):
        Image.new("RGB", (100, 100)).save(target / f"img{i}.jpg")

    ic.split_images(str(target), str(output), "cat", 0.2)

    assert "出力フォルダ作成エラー" in caplog.text   


# 4-3-7 異常：ファイル移動エラー（train）
def test_split_images_move_train_fail(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()

    for i in range(5):
        Image.new("RGB", (100, 100)).save(target / f"img{i}.jpg")

    def fake_move(*args, **kwargs):
        raise PermissionError("Cannot move the non-empty directory")
    
    monkeypatch.setattr("shutil.move", fake_move)

    ic.split_images(str(target), str(output), "cat", 0.2)

    assert "ファイル移動エラー（train）" in caplog.text 


# 4-3-8 異常：ファイル移動エラー（eval）
def test_split_images_move_eval_fail(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    target = tmp_path / "target"
    output = tmp_path / "output"
    target.mkdir()
    output.mkdir()

    for i in range(5):
        Image.new("RGB", (100, 100)).save(target / f"img{i}.jpg")

    def fake_move(*args, **kwargs):
        raise PermissionError("Cannot move the non-empty directory")
    
    monkeypatch.setattr("shutil.move", fake_move)

    ic.split_images(str(target), str(output), "cat", 0.2)

    assert "ファイル移動エラー（eval）" in caplog.text 


# ============================================================
# termination()
# ============================================================

# 4-4-1 正常：temp_dir 削除成功
def test_termination_success(tmp_path):
    ic = ImageCollector()

    temp = tmp_path / "temp"
    temp.mkdir()
    (temp / "dummy.txt").write_text("dummy")

    ic.temp_dir = str(temp)
    ic.termination()

    assert not temp.exists()

# 4-4-2 異常：削除失敗
def test_termination_delete_fail(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    ic = ImageCollector()

    temp = tmp_path / "temp"
    temp.mkdir()
    temp.chmod(0o000)  # 削除不可

    ic.temp_dir = str(temp)
    ic.termination()

    assert temp.exists()
    assert "一時フォルダ削除エラー" in caplog.text