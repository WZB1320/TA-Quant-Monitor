"""BaoStock 数据源复权映射单元测试

背景 (2026-10-14 复权口径审计):
    baostock 的 adjustflag 为 "1"=后复权, "2"=前复权, "3"=不复权,
    而修正前的代码写成 {"qfq": "1", "hfq": "2"} —— qfq/hfq 完全写反.
    因 DATA_SOURCE_ORDER 中 akshare 优先, 该 bug 未被触发; 但 akshare 失败
    由 baostock 接管时, 会把后复权数据写入 qfq 缓存 (后复权价格量级可达数千),
    导致后续信号与回测全错.

本测试用 fake baostock 模块替换真实依赖, 不发起网络请求,
验证"传给 baostock 的 adjustflag 参数"确实正确.
"""
import sys

import pytest

from src.data_fetcher.baostock_source import BaoStockSource


class _FakeRowSet:
    """模拟 baostock 的 ResultSet"""

    def __init__(self, rows):
        self.fields = ["date", "open", "high", "low", "close", "volume", "amount"]
        self.error_code = "0"
        self.error_msg = "success"
        self._rows = rows
        self._i = 0

    def next(self):
        if self._i < len(self._rows):
            self._i += 1
            return True
        return False

    def get_row_data(self):
        return self._rows[self._i - 1]


class _FakeLoginResult:
    error_code = "0"
    error_msg = "success"


@pytest.fixture
def fake_bs(monkeypatch):
    """替换 sys.modules['baostock'], 捕获调用参数"""
    captured = {}

    class FakeBS:
        @staticmethod
        def login():
            return _FakeLoginResult()

        @staticmethod
        def logout():
            pass

        @staticmethod
        def query_history_k_data_plus(code, fields, **kwargs):
            captured["code"] = code
            captured["fields"] = fields
            captured.update(kwargs)
            # 返回一行合法数据, 使 _standardize 能跑通
            return _FakeRowSet([["2026-01-05", "1.0", "1.0", "1.0", "1.0", "100", "1000"]])

    monkeypatch.setitem(sys.modules, "baostock", FakeBS)
    return captured


# ── 复权映射 ──

class TestAdjustMapping:
    def test_qfq_maps_to_flag_2(self, fake_bs):
        """qfq (前复权) → adjustflag="2" """
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust="qfq")
        assert fake_bs["adjustflag"] == "2"

    def test_hfq_maps_to_flag_1(self, fake_bs):
        """hfq (后复权) → adjustflag="1" """
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust="hfq")
        assert fake_bs["adjustflag"] == "1"

    def test_empty_maps_to_flag_3(self, fake_bs):
        """不复权 → adjustflag="3" """
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust="")
        assert fake_bs["adjustflag"] == "3"

    def test_none_maps_to_flag_3(self, fake_bs):
        """None → adjustflag="3" """
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust=None)
        assert fake_bs["adjustflag"] == "3"

    def test_default_is_qfq(self, fake_bs):
        """不传 adjust 时默认前复权 ("2")"""
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31")
        assert fake_bs["adjustflag"] == "2"

    def test_unknown_adjust_falls_back_to_qfq(self, fake_bs):
        """未知复权取值回退到前复权, 避免静默返回后复权量级数据"""
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust="weird")
        assert fake_bs["adjustflag"] == "2"

    def test_qfq_and_hfq_are_not_the_same_flag(self):
        """回归护栏: qfq 与 hfq 必须映射到不同 flag (曾因写反而相同语义互换)"""
        m = BaoStockSource.ADJUST_MAP
        assert m["qfq"] != m["hfq"]
        assert m["qfq"] == "2"
        assert m["hfq"] == "1"


# ── 代码格式转换 ──

class TestSymbolFormat:
    @pytest.mark.parametrize("raw,expected", [
        ("600887", "sh.600887"),
        ("000001", "sz.000001"),
        ("300308", "sz.300308"),
        ("688981", "sh.688981"),
        ("sh.600887", "sh.600887"),
        ("sz.000001", "sz.000001"),
    ])
    def test_format(self, raw, expected):
        assert BaoStockSource._format_symbol(raw) == expected

    def test_symbol_passed_through(self, fake_bs):
        """转换后的代码应传给 baostock"""
        BaoStockSource().get_daily_kline("600887", "2024-01-01", "2024-01-31", adjust="qfq")
        assert fake_bs["code"] == "sh.600887"
        assert fake_bs["frequency"] == "d"
