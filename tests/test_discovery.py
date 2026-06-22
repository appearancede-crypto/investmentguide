"""Discovery / movers screener tests (pure logic, no network)."""
from crypto_tool.analysis import discovery


def _coin(symbol, mcap, vol, p1=0.0, p24=0.0, p7=0.0, rank=100, ath=-50.0, price=1.0):
    return {
        "symbol": symbol, "name": symbol, "current_price": price,
        "market_cap": mcap, "total_volume": vol, "market_cap_rank": rank,
        "price_change_percentage_1h_in_currency": p1,
        "price_change_percentage_24h_in_currency": p24,
        "price_change_percentage_7d_in_currency": p7,
        "ath_change_percentage": ath,
    }


def test_microcap_illiquid_is_extreme():
    rows = discovery.screen([_coin("GEM", 5_000_000, 100_000, p24=40, rank=600)],
                            binance_listed=set(), tracked=set(), top=10, min_volume=0)
    r = rows[0]
    assert r["risk"] == "EXTREME"
    assert any("microcap" in f for f in r["flags"])
    assert any("liquidity" in f for f in r["flags"])
    assert r["onBinance"] is False and r["deepDive"] is False


def test_largecap_liquid_is_lowest_tier_and_marked():
    rows = discovery.screen([_coin("BTC", 800_000_000_000, 30_000_000_000, p24=2, rank=1, ath=-10)],
                            binance_listed={"BTC"}, tracked={"BTCUSDT"}, top=10, min_volume=0)
    r = rows[0]
    assert r["risk"] == "MEDIUM"             # screener never claims 'low' risk
    assert r["onBinance"] and r["deepDive"] and r["tracked"] and r["binanceSymbol"] == "BTCUSDT"


def test_on_binance_but_untracked_is_deepdivable():
    """A Binance mover we don't track yet is still deep-divable (fetched live)."""
    rows = discovery.screen([_coin("PEPE", 200_000_000, 50_000_000, p24=12)],
                            binance_listed={"PEPE"}, tracked=set(), top=10, min_volume=0)
    r = rows[0]
    assert r["onBinance"] and r["deepDive"] and r["binanceSymbol"] == "PEPEUSDT"
    assert r["tracked"] is False


def test_min_volume_filter_and_sort():
    market = [_coin("A", 100_000_000, 500_000, p24=5),
              _coin("B", 100_000_000, 100_000, p24=50)]   # B gains more but is too illiquid
    rows = discovery.screen(market, set(), set(), top=10, sort="gainers", min_volume=250_000)
    assert [r["symbol"] for r in rows] == ["A"]


def test_not_on_binance_flagged():
    rows = discovery.screen([_coin("XYZ", 200_000_000, 5_000_000, p24=5)],
                            binance_listed=set(), tracked=set(), top=10, min_volume=0)
    assert any("not on Binance" in f.lower() or "not on binance" in f.lower()
               for f in rows[0]["flags"])
