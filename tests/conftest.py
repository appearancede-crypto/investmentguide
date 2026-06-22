import pytest

from crypto_tool.config import load_config
from crypto_tool.data import synthetic


@pytest.fixture(scope="session")
def cfg():
    # Use built-in defaults (no dependence on a config.yaml during tests).
    return load_config(path="__nonexistent__.yaml")


@pytest.fixture()
def ohlcv():
    return synthetic.generate_ohlcv("BTCUSDT", interval="1h", n=400, seed=7)
