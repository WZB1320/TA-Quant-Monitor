"""诊断: 检查 df["date"] 的实际类型和内容"""
from src.data_fetcher import DataManager, Watchlist
import pandas as pd

dm = DataManager()
wl = Watchlist()

for s in wl.get_all():
    code = s["code"]
    df = dm.get_daily_kline(code, start_date="2024-01-01")
    if df is not None:
        print(f"\n{code}:")
        print(f"  date dtype: {df['date'].dtype}")
        print(f"  date 前3行: {df['date'].head(3).tolist()}")
        print(f"  date 类型: {[type(x) for x in df['date'].head(3)]}")
        print(f"  date 后3行: {df['date'].tail(3).tolist()}")
        
        # 测试比较
        test_date = pd.Timestamp("2025-07-18").date()
        target_str = test_date.strftime("%Y-%m-%d")
        mask = df["date"] == target_str
        print(f"  比较 '{target_str}' == df['date']: {mask.any()}, 匹配行数={mask.sum()}")
        
        # 手动查找
        found = False
        for j in range(len(df)):
            if pd.Timestamp(df["date"].iloc[j]).date() == test_date:
                print(f"  手动查找找到: idx={j}, date={df['date'].iloc[j]}")
                found = True
                break
        if not found:
            print(f"  手动查找: 未找到 {test_date}")
        
        # 只检查一只股票就够了
        break