"""公司行动事件表 (corporate action)

职责:
1. 持久化除权除息事件 (送股 / 转增 / 现金分红) —— 落 data/kline_cache.db 的
   corporate_action 表, 与行情同库, 便于按 (symbol, date) 一条 SQL join.
2. 提供除权调整计算: 股数、成本基准、最高价的下移, 以及现金分红金额.
3. 提供过期哨兵: 事件表是低频变更数据, 长期不更新会静默漏掉新分红.

术语 (详见 outputs/adjust_reform_project_plan_20261022.md 附录 A):
    raw   不复权, 真实成交价, 唯一物理事实
    qfq   前复权, 供信号层看图
    D     每股派息 (元). dps_pre 含税 / dps_after 税后
    s     送转比例 (每 1 股送转几股) = share_ratio + transfer_ratio
    除权参考价  (前收 − D) / (1 + s), 除权日涨跌停判定基准
    ref_price  除权调整后的成本基准, 用于止损/止盈百分比计算

设计约束 (易错, 勿改):
    - 派息基数必须是含税 dps_pre. 公告的"扣税后"值恒定按 10% 计算,
      与本策略实际资本加权税率 13.49% 不符.
      例外: s > 0 时送股已按面值 1 元/股代扣个税且内扣于现金派息,
            此时改以 dps_after 为基数, 不再叠加自算税率.
    - baostock 的 foreAdjustFactor 不能用 (不是逐事件乘数), 只能用
      backAdjustFactor 的逐行比值.
"""
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.config.settings import CACHE_DB

# 建表语句 (单一事实来源: EventBook.ensure_table 与 KLineCache._init_db 共用,
# 避免同一份 DDL 散落两处产生漂移)
CORPORATE_ACTION_SCHEMA = """
    CREATE TABLE IF NOT EXISTS corporate_action (
        symbol          TEXT NOT NULL,
        ex_date         TEXT NOT NULL,
        record_date     TEXT,
        dps_pre         REAL,
        dps_after        REAL,
        share_ratio      REAL DEFAULT 0,
        transfer_ratio  REAL DEFAULT 0,
        multiplier      REAL,
        mult_baostock    REAL,
        mult_em          REAL,
        verified         INTEGER DEFAULT 0,
        reason           TEXT,
        checked_at       TEXT,
        PRIMARY KEY (symbol, ex_date)
    )
"""


@dataclass
class CorporateAction:
    """单条除权除息事件

    dps_pre / dps_after: 每股派息 (元), 不是每 10 股. 数据源给的是每 10 股,
    写入前必须除以 10 — 这里存的是"每股", 避免消费端每次换算出错.
    """

    symbol: str
    ex_date: str
    record_date: Optional[str] = None
    dps_pre: float = 0.0
    dps_after: float = 0.0
    share_ratio: float = 0.0
    transfer_ratio: float = 0.0
    multiplier: Optional[float] = None
    mult_baostock: Optional[float] = None
    mult_em: Optional[float] = None
    verified: int = 0
    reason: Optional[str] = None
    checked_at: Optional[str] = None

    @property
    def s(self) -> float:
        """送转合计比例 (每 1 股送转几股)"""
        return self.share_ratio + self.transfer_ratio

    def ex_reference_price(self, prev_close: float) -> float:
        """除权参考价: 除权日涨跌停判定基准 = (前收 − D) / (1 + s)"""
        return (prev_close - self.dps_pre) / (1 + self.s)

    def adjust_prev_close(self, raw_prev_close: float) -> float:
        """把 raw 前收盘价换算成除权日的涨跌停判定基准

        优先用市场乘数 (已含"回购股不参与分配"等实际口径), 缺失时退回公告
        口径 (前收 − D)/(1+s). 交易所按分公布, 故统一四舍五入到 2 位.
        """
        if self.multiplier and self.multiplier > 0:
            return round(raw_prev_close / self.multiplier, 2)
        return round(self.ex_reference_price(raw_prev_close), 2)


def dividend_tax_rate(holding_days: int) -> float:
    """股息红利税税率 (财税〔2015〕101号, 2015-09-08 起)

    持股 ≤1 年时派息环节不扣缴, 待卖出时按持股期限补收 —— 所以税率在
    卖出那一刻才确定, 回测里 holding_days 已知, 直接按此分档即可.

    | 持股期限      | 税率 |
    |--------------|------|
    | ≤ 1 个月(含) | 20%  |
    | 1 个月 ~ 1 年 | 10%  |
    | > 1 年       | 0%   |

    实测 (229 笔基线): 持仓 max 103 天 → 0% 档本策略永不触发.
    资本加权综合税率 13.49%, 而非公告隐含的 10%.
    """
    if holding_days <= 30:
        return 0.20
    if holding_days <= 365:
        return 0.10
    return 0.0


class EventBook:
    """公司行动事件表读写与除权调整计算"""

    def __init__(self, db_path: str = CACHE_DB):
        self.db_path = db_path
        self._ensure_table()

    # ---------------------------------------------------------------- 建表

    @staticmethod
    def ensure_table(conn: sqlite3.Connection) -> None:
        conn.execute(CORPORATE_ACTION_SCHEMA)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ca_symbol_date "
            "ON corporate_action(symbol, ex_date)"
        )

    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            self.ensure_table(conn)

    # ---------------------------------------------------------------- 写入

    def upsert(self, action: CorporateAction) -> None:
        """写入单条事件 (主键 (symbol, ex_date) 冲突时覆盖)"""
        self.upsert_many([action])

    def upsert_many(self, actions: List[CorporateAction]) -> int:
        if not actions:
            return 0
        sql = """
            INSERT OR REPLACE INTO corporate_action
                (symbol, ex_date, record_date, dps_pre, dps_after,
                 share_ratio, transfer_ratio, multiplier, mult_baostock, mult_em,
                 verified, reason, checked_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with sqlite3.connect(self.db_path) as conn:
            for a in actions:
                conn.execute(
                    sql,
                    (
                        a.symbol,
                        a.ex_date,
                        a.record_date,
                        a.dps_pre,
                        a.dps_after,
                        a.share_ratio,
                        a.transfer_ratio,
                        a.multiplier,
                        a.mult_baostock,
                        a.mult_em,
                        a.verified,
                        a.reason,
                        a.checked_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
        return len(actions)

    # ---------------------------------------------------------------- 读取

    @staticmethod
    def _row_to_action(row: Tuple) -> CorporateAction:
        return CorporateAction(
            symbol=row[0],
            ex_date=row[1],
            record_date=row[2],
            dps_pre=row[3] if row[3] is not None else 0.0,
            dps_after=row[4] if row[4] is not None else 0.0,
            share_ratio=row[5] if row[5] is not None else 0.0,
            transfer_ratio=row[6] if row[6] is not None else 0.0,
            multiplier=row[7],
            mult_baostock=row[8],
            mult_em=row[9],
            verified=row[10] if row[10] is not None else 0,
            reason=row[11],
            checked_at=row[12],
        )

    _SELECT_COLS = (
        "symbol, ex_date, record_date, dps_pre, dps_after, share_ratio, "
        "transfer_ratio, multiplier, mult_baostock, mult_em, verified, reason, checked_at"
    )

    def get(self, symbol: str, ex_date: str) -> Optional[CorporateAction]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM corporate_action "
                "WHERE symbol = ? AND ex_date = ?",
                (symbol, ex_date),
            ).fetchone()
        return self._row_to_action(row) if row else None

    def events_between(self, symbol: str, start: str, end: str) -> List[CorporateAction]:
        """取某标的在 [start, end] 内的全部除权事件 (含端点, 按日期升序)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM corporate_action "
                "WHERE symbol = ? AND ex_date >= ? AND ex_date <= ? ORDER BY ex_date ASC",
                (symbol, start, end),
            ).fetchall()
        return [self._row_to_action(r) for r in rows]

    def all_events(self, symbol: str = None) -> List[CorporateAction]:
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM corporate_action "
                    "WHERE symbol = ? ORDER BY ex_date ASC",
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM corporate_action "
                    "ORDER BY symbol, ex_date ASC"
                ).fetchall()
        return [self._row_to_action(r) for r in rows]

    def delete_symbol(self, symbol: str) -> int:
        """删除某标的的全部事件 (重跑校验前先清, 避免留下上次的过期行)

        事件表是校验脚本的**唯一**写入方; 重跑时若不先清, 上次检出而这次没检出
        的事件 (例如阈值调整后合并/消失的事件) 会作为幽灵行留在库里被成交层读到.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM corporate_action WHERE symbol = ?", (symbol,))
            return cur.rowcount

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM corporate_action").fetchone()[0]

    # ------------------------------------------------- 除权调整 (核心计算)

    def apply_to_position(
        self,
        symbol: str,
        ex_date: str,
        shares: int,
        ref_price: float,
        highest_price: Optional[float] = None,
    ) -> Tuple[int, float, Optional[float], float, float]:
        """对一笔持仓施加除权调整

        返回 (shares_new, ref_price_new, highest_price_new, cash_dividend, s)

        要点:
        - 现金分红按**除权日持有股数**计算 (送转新增的股不参与本次派息),
          故用调整前的 shares.
        - **现金**用公告派息 dps_pre/dps_after —— 这是股东实际到账的钱.
        - **成本基准与最高价**用市场乘数 multiplier 下移 (ref / mult). 不用公告
          D/s 推导, 因为"回购股份不参与利润分配"时两者不等:
          实测 600519 2025-06-26 公告 27.6730 vs 行情价格调整 27.6007.
          用 ref/mult 还能保证 ref/price 的比值在除权日前后连续, 止损语义不变.
        - ref_price / highest_price 必须与价格同步下移, 否则除权日的价格
          跳空会被当成暴跌 —— 这是"除权日误止损"的根因.
        - s == 0 时派息基数取含税 dps_pre; s > 0 时取 dps_after (见模块文档).
        """
        ev = self.get(symbol, ex_date)
        if ev is None:
            return shares, ref_price, highest_price, 0.0, 0.0

        s = ev.s
        dps = ev.dps_pre if s == 0 else ev.dps_after

        # 现金: 除权日持有的股数 × 每股派息 (含税, 税在平仓时才扣)
        cash_dividend = dps * shares

        # 价格侧调整: 三选一, 优先级从上到下
        if ev.multiplier and ev.multiplier > 0:
            # 市场实际乘数 —— ref/price 比值在除权日前后连续, 止损语义不变
            ref_price_new = ref_price / ev.multiplier
            highest_new = highest_price / ev.multiplier if highest_price is not None else None
        elif s:
            # 无市场乘数, 退回公告口径: 含送转
            ref_price_new = (ref_price - dps) / (1 + s)
            highest_new = (highest_price - dps) / (1 + s) if highest_price is not None else None
        else:
            # 无市场乘数, 退回公告口径: 纯派息
            ref_price_new = ref_price - dps
            highest_new = highest_price - dps if highest_price is not None else None

        return int(round(shares * (1 + s))), ref_price_new, highest_new, cash_dividend, s

    def adjusted_prev_close(self, symbol: str, date: str, raw_prev_close: float) -> float:
        """涨跌停判定基准价: 除权日返回除权参考价, 其余日期原样返回

        除权日 raw 序列存在真实跳空 (派息导致), 若沿用 raw 前收做涨跌停基准,
        会把除权跳空误判成跌停而禁止交易.
        """
        ev = self.get(symbol, date)
        if ev is None:
            return raw_prev_close
        return ev.adjust_prev_close(raw_prev_close)

    def bulk(self, symbols: List[str], start: str = None, end: str = None
             ) -> Dict[str, Dict[str, "CorporateAction"]]:
        """一次性预加载多只标的的事件, 返回 {symbol: {ex_date: action}}

        成交层按日循环调用, 若每次 get() 都查一次 SQLite, 42 只 × 664 天
        就是 ~2.8 万次查询 —— 全量回测会被拖垮. 故开局一次性载入内存.
        """
        if not symbols:
            return {}
        ph = ",".join("?" * len(symbols))
        sql = f"SELECT {self._SELECT_COLS} FROM corporate_action WHERE symbol IN ({ph})"
        args = list(symbols)
        if start:
            sql += " AND ex_date >= ?"
            args.append(start)
        if end:
            sql += " AND ex_date <= ?"
            args.append(end)
        sql += " ORDER BY symbol, ex_date ASC"

        out: Dict[str, Dict[str, "CorporateAction"]] = {s: {} for s in symbols}
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(sql, args):
                a = self._row_to_action(row)
                out.setdefault(a.symbol, {})[a.ex_date] = a
        return out

    # ---------------------------------------------------------------- 哨兵

    def staleness(self, symbols: List[str]) -> dict:
        """事件表新鲜度检查 (本地查询, 零网络, 不阻断)

        返回:
            max_age_days  最旧的 checked_at 距今天数 (无数据则为 None)
            missing       表内 0 行的标的 (从未校验过)
            count         命中行数
        """
        if not symbols:
            return {"max_age_days": None, "missing": [], "count": 0}
        ph = ",".join("?" * len(symbols))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT symbol, MIN(checked_at) FROM corporate_action "
                f"WHERE symbol IN ({ph}) GROUP BY symbol",
                symbols,
            ).fetchall()
        present = {r[0]: r[1] for r in rows}
        missing = [s for s in symbols if s not in present]

        ages = []
        for _, ts in present.items():
            if not ts:
                continue
            try:
                ages.append((datetime.now() - datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")).days)
            except ValueError:
                continue
        return {
            "max_age_days": max(ages) if ages else None,
            "missing": missing,
            "count": len(present),
        }


def check_staleness(symbols: List[str], db_path: str = CACHE_DB, max_age_days: int = 90):
    """过期哨兵入口: 超龄或缺失时返回告警串, 正常返回 None (warn 不阻断)"""
    st = EventBook(db_path).staleness(symbols)
    warns = []
    if st["max_age_days"] is not None and st["max_age_days"] > max_age_days:
        warns.append(
            f"事件表最旧校验记录距今 {st['max_age_days']} 天 (> {max_age_days} 天), "
            "可能存在未入库的新分红"
        )
    if st["missing"]:
        warns.append(f"{len(st['missing'])} 只标的事件表无记录 (从未校验): "
                     f"{', '.join(st['missing'][:10])}"
                     + ("..." if len(st["missing"]) > 10 else ""))
    return "; ".join(warns) if warns else None
