import logging.config
import yaml
from pathlib import Path
import platform

# pytest 実行時に logging 設定を読み込む
def pytest_configure():
    file_path = None
    os_name = platform.system()

    if os_name == 'Windows':
        file_path = Path("C:/Users/yniit/Documents/aitraining").resolve()
    elif os_name == 'Linux':
        file_path = Path(__file__).resolve().parent.parent
    
    file_path = file_path / 'config' / 'logging.yaml'

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


import pytest

@pytest.fixture(autouse=True)
def _auto_caplog_level(caplog):
    caplog.set_level(logging.ERROR, logger="myapp")
