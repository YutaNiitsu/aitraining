import pytest
from unittest.mock import patch
from python.configManager import ConfigManager

# -----------------------------
# __init__() のテスト
# -----------------------------
# 1-1-1 正常：初期化
def test_init_success(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    # 空ファイルを作成
    learn.touch()
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))

    assert cm.learn_conf_path == learn
    assert cm.log_conf_path == log
    assert cm.labels_path == labels


# 1-1-2 異常：learn_conf_path が None
def test_init_learn_none(tmp_path):
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"
    
    log.touch()
    labels.touch()

    with pytest.raises(RuntimeError, match="learn_conf_path の初期化に失敗しました"):
        ConfigManager(None, str(log), str(labels))


# 1-1-3 異常：log_conf_path が None
def test_init_log_none(tmp_path):
    learn = tmp_path / "learn.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch()
    labels.touch()

    with pytest.raises(RuntimeError, match="log_conf_path の初期化に失敗しました"):
        ConfigManager(str(learn), None, str(labels))


# 1-1-4 異常：labels_path が None
def test_init_labels_none(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"

    learn.touch()
    log.touch()

    with pytest.raises(RuntimeError, match="labels_path の初期化に失敗しました"):
        ConfigManager(str(learn), str(log), None)


# 1-1-5〜1-1-7 異常：数値パス
@pytest.mark.parametrize("learn, log, labels, error_msg", [
    (123, "log.yaml", "labels.yaml", "learn_conf_path の初期化に失敗しました"),
    ("learn.yaml", 123, "labels.yaml", "log_conf_path の初期化に失敗しました"),
    ("learn.yaml", "log.yaml", 123, "labels_path の初期化に失敗しました"),
])
def test_init_numeric_path(tmp_path, learn, log, labels, error_msg):
    # 存在ファイルを作成
    (tmp_path / "learn.yaml").touch()
    (tmp_path / "log.yaml").touch()
    (tmp_path / "labels.yaml").touch()

    # 数値パスは Path() で例外になる
    with pytest.raises(RuntimeError, match=error_msg):
        ConfigManager(
            str(tmp_path / learn) if isinstance(learn, str) else learn,
            str(tmp_path / log) if isinstance(log, str) else log,
            str(tmp_path / labels) if isinstance(labels, str) else labels,
        )

# -----------------------------
# load_config() のテスト
# -----------------------------
# 1-2-1 正常：読み込み
def test_load_config_success(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.write_text("lr: 0.01")
    log.write_text("version: 1\nhandlers: {}")
    labels.write_text("cat: 0")

    cm = ConfigManager(str(learn), str(log), str(labels))
    cm.load_config()

    assert cm.learn_config == {"lr": 0.01}
    assert cm.log_config == {"version": 1, "handlers": {}}
    assert cm.labels == {"cat": 0}


# 1-2-2 異常：学習設定ファイルが存在しない
def test_load_config_learn_not_found(tmp_path):
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    log.touch()
    labels.touch()

    cm = ConfigManager("not_exist.yaml", str(log), str(labels))

    with pytest.raises(RuntimeError, match="学習設定ファイルが存在しません"):
        cm.load_config()


# 1-2-3 異常：学習設定ファイルの権限エラー
def test_load_config_learn_permission(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch() 
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))
    with patch("builtins.open", side_effect=PermissionError("アクセス拒否")): 
        with pytest.raises(RuntimeError, match="学習設定ファイルにアクセスできません"):
            cm.load_config()


# 1-2-4 異常：学習設定ファイル YAML 構文エラー
def test_load_config_learn_yaml_error(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.write_text("::: invalid yaml :::")
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="学習設定ファイルの YAML 構文エラー"):
        cm.load_config()


# 1-2-5 異常：学習設定ファイル その他エラー
def test_load_config_learn_other_error(tmp_path, monkeypatch):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch()
    log.touch()
    labels.touch()

    def fake_open(*args, **kwargs):
        raise Exception("unexpected error")

    monkeypatch.setattr("builtins.open", fake_open)

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="学習設定ファイルの読み込み中に予期せぬエラー"):
        cm.load_config()


# 1-2-6 異常：ログ設定ファイルが存在しない
def test_load_config_log_not_found(tmp_path):
    learn = tmp_path / "learn.yaml"
    labels = tmp_path / "labels.yaml"
    learn.touch()
    labels.touch()

    cm = ConfigManager(str(learn), "not_exist.yaml", str(labels))

    with pytest.raises(RuntimeError, match="ログ設定ファイルが存在しません"):
        cm.load_config()


# 1-2-7 異常：ログ設定ファイルの権限エラー
def test_load_config_log_permission(tmp_path, monkeypatch):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch()
    log.touch()
    labels.touch()

    # 本物の open を退避
    open_original = open

    def fake_open(path, *args, **kwargs):
        # ログ設定ファイルのときだけ例外を投げる
        if str(path) == str(log):
            raise PermissionError("アクセス拒否")
        return open_original(path, *args, **kwargs)
    
    monkeypatch.setattr("builtins.open", fake_open)

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="ログ設定ファイルにアクセスできません"):
        cm.load_config()


# 1-2-8 異常：ログ設定ファイル YAML 構文エラー
def test_load_config_log_yaml_error(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch()
    log.write_text("::: invalid yaml :::")
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="ログ設定ファイルの YAML 構文エラー"):
        cm.load_config()


# 1-2-9 異常：ログ設定ファイル その他エラー
def test_load_config_log_other_error(tmp_path, monkeypatch):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.touch()
    log.touch()
    labels.touch()

    # 本物の open を退避
    open_original = open

    def fake_open(path, *args, **kwargs):
        # ログ設定ファイルのときだけ例外を投げる
        if str(path) == str(log):
            raise Exception("unexpected error")
        return open_original(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="ログ設定ファイルの読み込み中に予期せぬエラー"):
        cm.load_config()



# 1-2-10 異常：ラベルファイルが存在しない
def test_load_config_labels_not_found(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"

    learn.write_text("lr: 0.01")
    log.write_text("version: 1\nhandlers: {}")

    cm = ConfigManager(str(learn), str(log), "not_exist.yaml")

    with pytest.raises(RuntimeError, match="ラベルファイルが存在しません"):
        cm.load_config()


# 1-2-11 異常：ラベルファイルの権限エラー
def test_load_config_labels_permission(tmp_path, monkeypatch):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.write_text("lr: 0.01")
    log.write_text("version: 1\nhandlers: {}")
    labels.write_text("::: invalid yaml :::")

    # 本物の open を退避
    open_original = open

    def fake_open(path, *args, **kwargs):
        # ラベルファイルのときだけ例外を投げる
        if str(path) == str(labels):
            raise PermissionError("アクセス拒否")
        return open_original(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    cm = ConfigManager(str(learn), str(log), str(labels))
     
    with pytest.raises(RuntimeError, match="ラベルファイルにアクセスできません"):
        cm.load_config()


# 1-2-12 異常：ラベルファイル YAML 構文エラー
def test_load_config_labels_yaml_error(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.write_text("lr: 0.01")
    log.write_text("version: 1\nhandlers: {}")
    labels.write_text("::: invalid yaml :::")

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="ラベルファイルの YAML 構文エラー"):
        cm.load_config()


# 1-2-13 異常：ラベルファイル その他エラー
def test_load_config_labels_other_error(tmp_path, monkeypatch):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"

    learn.write_text("lr: 0.01")
    log.write_text("version: 1\nhandlers: {}")
    labels.write_text("::: invalid yaml :::")

    # 本物の open を退避
    open_original = open

    def fake_open(path, *args, **kwargs):
        # ラベルファイルのときだけ例外を投げる
        if str(path) == str(labels):
            raise Exception("unexpected error")
        return open_original(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    cm = ConfigManager(str(learn), str(log), str(labels))

    with pytest.raises(RuntimeError, match="ラベルファイルの読み込み中に予期せぬエラー"):
        cm.load_config()


# -----------------------------
# is_updated() のテスト
# -----------------------------
# 1-3-1 正常：初回
def test_is_updated_first_time(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"
    learn.touch()
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))

    assert cm.is_updated() is True


# 1-3-2 正常：変更あり
def test_is_updated_changed(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"
    learn.touch()
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))
    cm.update_timestamp()

    # mtime を更新
    learn.write_text("updated")

    assert cm.is_updated() is True


# 1-3-3 正常：変更なし
def test_is_updated_not_changed(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"
    learn.touch()
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))
    cm.update_timestamp()

    assert cm.is_updated() is False


# -----------------------------
# is_updated() のテスト
# -----------------------------
# 1-4-1 正常
def test_update_timestamp(tmp_path):
    learn = tmp_path / "learn.yaml"
    log = tmp_path / "log.yaml"
    labels = tmp_path / "labels.yaml"
    learn.touch()
    log.touch()
    labels.touch()

    cm = ConfigManager(str(learn), str(log), str(labels))
    cm.update_timestamp()

    assert cm.last_timestamp == learn.stat().st_mtime
