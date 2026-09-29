"""收盘复盘大屏 · 渲染（离线 fallback 版）

⚠️ DEPRECATED（2026-09-29）: 权威实现已迁至 web/frontend/src/pages/Dashboard/（应用内 /dashboard 路由）。
   本脚本冻结不再迭代，仅用于生成 web/dashboard/dashboard.html 离线单页
   （双击即开、零常驻进程，适合无后端场景）。

读取 data/dashboard.json → 生成 web/dashboard/dashboard.html（自包含、离线可开）
设计: 专业量化终端风 — 深蓝黑底 + 琥珀金强调 + 等宽数字表格
用法: python scripts/dashboard_render.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_PATH = os.path.join(ROOT, "data", "dashboard.json")
OUT_PATH = os.path.join(ROOT, "web", "dashboard", "dashboard.html")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=1920">
<link rel="icon" href="data:,">
<title>收盘复盘 · 量化信号终端</title>
<style>
/* ═══ 设计令牌 ═══ */
:root{
  --bg:#0B1220;
  --surface:#141E2E; --surface-2:#1A2537; --surface-3:#1E293B;
  --line:#26334A; --line-soft:rgba(38,51,74,.55);
  --txt:#F8FAFC; --txt-2:#CBD5E1; --txt-3:#94A3B8; --txt-4:#64748B;
  --gold:#F59E0B; --gold-2:#FBBF24; --gold-dim:rgba(245,158,11,.14);
  --up:#F6465D; --up-soft:rgba(246,70,93,.13);
  --down:#0ECB81; --down-soft:rgba(14,203,129,.13);
  --mono:"Cascadia Mono","JetBrains Mono",Consolas,"SF Mono",ui-monospace,monospace;
  --sans:"Microsoft YaHei UI","Microsoft YaHei","PingFang SC","Segoe UI",system-ui,sans-serif;
  --r:12px;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--txt);
  font-family:var(--sans);-webkit-font-smoothing:antialiased}
/* 背景氛围: 双径向光晕 + 极淡网格 */
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(1100px 560px at 12% -12%, rgba(245,158,11,.085), transparent 62%),
    radial-gradient(900px 480px at 88% -8%, rgba(56,189,248,.06), transparent 62%);}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.32;
  background-image:linear-gradient(rgba(148,163,184,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(148,163,184,.05) 1px,transparent 1px);
  background-size:48px 48px;}
#stage{position:relative;z-index:1;width:1920px;height:1200px;transform-origin:top left;
  display:grid;grid-template-rows:84px 1fr 40px;gap:12px;padding:12px 14px}

/* ═══ 面板 ═══ */
.panel{position:relative;display:flex;flex-direction:column;min-height:0;overflow:hidden;
  border:1px solid var(--line);border-radius:var(--r);
  background:linear-gradient(180deg,var(--surface),#111A28);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035), 0 10px 30px -18px rgba(0,0,0,.9);
  animation:rise .55s cubic-bezier(.16,1,.3,1) both}
.panel.d1{animation-delay:.04s}.panel.d2{animation-delay:.10s}
.panel.d3{animation-delay:.16s}.panel.d4{animation-delay:.22s}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.p-head{display:flex;align-items:center;gap:12px;padding:11px 16px 10px;border-bottom:1px solid var(--line-soft)}
.p-head h2{font-size:16px;font-weight:600;letter-spacing:1.5px;display:flex;align-items:center;gap:9px}
.p-head h2::before{content:"";width:3px;height:15px;border-radius:2px;
  background:linear-gradient(180deg,var(--gold-2),var(--gold));box-shadow:0 0 10px rgba(245,158,11,.55)}
.tag{font-size:11px;color:var(--txt-3);border:1px solid var(--line);border-radius:5px;
  padding:2px 8px;letter-spacing:.4px;white-space:nowrap}
.tag.warn{color:var(--gold-2);border-color:rgba(245,158,11,.32);background:var(--gold-dim)}

/* ═══ 顶栏 ═══ */
.top{display:flex;align-items:stretch;gap:14px}
.brand{display:flex;flex-direction:column;justify-content:center;padding-right:20px;
  border-right:1px solid var(--line-soft);min-width:230px}
.brand h1{font-size:25px;font-weight:700;letter-spacing:7px;line-height:1.15}
.brand h1 em{font-style:normal;color:var(--gold-2)}
.brand .sub{font-family:var(--mono);font-size:10.5px;letter-spacing:2.6px;color:var(--txt-4);margin-top:6px}
.idx-wrap{display:flex;gap:12px}
.idx{position:relative;display:flex;align-items:center;gap:14px;padding:0 16px;width:348px;
  border:1px solid var(--line);border-radius:var(--r);overflow:hidden;
  background:linear-gradient(135deg,var(--surface-2),#121B29)}
.idx .l{display:flex;flex-direction:column;gap:6px;min-width:92px}
.idx .nm{font-size:13.5px;color:var(--txt-2);letter-spacing:.5px;white-space:nowrap}
.idx .pct{font-family:var(--mono);font-size:12.5px;font-weight:600;padding:2px 7px;border-radius:4px;
  width:fit-content;font-variant-numeric:tabular-nums;white-space:nowrap}
.idx .v{font-family:var(--mono);font-size:30px;font-weight:700;letter-spacing:-.5px;
  line-height:1;font-variant-numeric:tabular-nums}
.idx .d{font-family:var(--mono);font-size:12.5px;color:var(--txt-3);margin-top:5px;
  font-variant-numeric:tabular-nums}
.idx .spark{margin-left:auto;opacity:.95}
.fresh{margin-left:auto;display:flex;flex-direction:column;justify-content:center;align-items:flex-end;gap:8px}
.fresh .row{display:flex;align-items:center;gap:9px;font-size:12.5px;color:var(--txt-3)}
.fresh b{color:var(--txt);font-family:var(--mono);font-weight:600;font-size:13.5px}
.badge-lag{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;padding:3px 10px;border-radius:20px;
  color:var(--gold-2);background:var(--gold-dim);border:1px solid rgba(245,158,11,.3)}
.badge-lag.ok{color:var(--down);background:var(--down-soft);border-color:rgba(14,203,129,.3)}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor}

/* ═══ 主体 ═══ */
.main{display:grid;grid-template-columns:62fr 38fr;gap:12px;min-height:0}
.col{display:grid;gap:12px;min-height:0}
.col.right{grid-template-rows:56fr 44fr}

/* ═══ 信号表 ═══ */
.sum{display:flex;gap:8px;margin-left:auto}
.chip{display:flex;align-items:center;gap:7px;padding:3px 10px;border-radius:6px;font-size:12px;
  background:var(--surface-3);border:1px solid var(--line-soft);color:var(--txt-3)}
.chip b{font-family:var(--mono);font-size:14px;font-weight:700;font-variant-numeric:tabular-nums}
.chip.u b{color:var(--up)}.chip.d b{color:var(--down)}.chip.g b{color:var(--gold-2)}
.tbl{flex:1;display:flex;flex-direction:column;min-height:0}
.ths,.tr{display:grid;grid-template-columns:3px 26px 112px 84px 72px 76px 100px 96px 92px 30px 1fr;
  align-items:center;padding:0 14px 0 0}
.ths{height:34px;font-size:10.5px;letter-spacing:1.2px;color:var(--txt-4);
  background:rgba(30,41,59,.55);border-bottom:1px solid var(--line-soft)}
.ths>div,.tr>div{padding-left:10px}
.tr>.bar{padding-left:0}
.tr .no{color:var(--txt-4);font-size:11px}
.rows{flex:1;min-height:0}
.tr{position:relative;border-bottom:1px solid rgba(38,51,74,.4);
  font-family:var(--mono);transition:background .15s ease}
.tr .bar{width:3px;height:100%;border-radius:0 2px 2px 0;opacity:.9}
.tr.focus{height:40px;font-size:15px;background:rgba(245,158,11,.035)}
.tr:not(.focus){height:24.2px;font-size:12.5px;color:#B9C5D6}
.tr:hover{background:rgba(148,163,184,.09)}
.tr .idx{color:var(--txt-4);font-size:11px}
.tr .code{font-weight:600;letter-spacing:.3px}
.tr .code small{display:block;font-family:var(--sans);font-size:10.5px;font-weight:400;color:var(--txt-4);
  letter-spacing:0;margin-top:1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:104px}
.tr .code small.inl{display:inline;margin:0 0 0 5px;max-width:66px;vertical-align:baseline}
.tr .grp{font-family:var(--sans);font-size:11px;color:var(--txt-4);overflow:hidden;white-space:nowrap}
.tr .num{font-variant-numeric:tabular-nums;text-align:right;padding-right:6px}
.tr .pct{font-variant-numeric:tabular-nums;text-align:right;padding-right:6px;font-weight:600}
.badge{display:inline-block;padding:2px 9px;border-radius:5px;font-family:var(--sans);font-size:12px;
  font-weight:700;color:#0B1220;letter-spacing:.5px;text-align:center;min-width:62px}
.tr:not(.focus) .badge{font-size:11px;padding:1px 7px;min-width:54px}
.score{display:flex;align-items:center;gap:7px}
.score .track{position:relative;width:40px;height:4px;border-radius:2px;background:rgba(148,163,184,.16);overflow:hidden}
.score .fill{position:absolute;top:0;left:0;height:100%;border-radius:2px}
.score .v{font-variant-numeric:tabular-nums;min-width:40px;text-align:right}
.exec{font-family:var(--sans);font-size:11.5px;letter-spacing:.2px;white-space:nowrap}
.exec.ok{color:var(--down);font-weight:600}
.hold{display:flex;justify-content:center}
.hold svg{filter:drop-shadow(0 0 5px rgba(245,158,11,.6))}
.inds{font-size:11px;color:var(--txt-4);text-align:right;padding-right:2px;
  font-variant-numeric:tabular-nums;overflow:hidden;white-space:nowrap}

/* ═══ 净值 ═══ */
.nav-body{flex:1;display:flex;flex-direction:column;padding:10px 16px 12px;min-height:0}
.nav-top{display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap}
.nav-top .big{font-family:var(--mono);font-size:45px;font-weight:700;line-height:1;
  color:var(--gold-2);letter-spacing:-1px;font-variant-numeric:tabular-nums;
  text-shadow:0 0 26px rgba(245,158,11,.35)}
.nav-top .unit{font-size:11.5px;color:var(--txt-4);letter-spacing:1px;padding-bottom:5px}
.nav-top .money{font-family:var(--mono);font-size:19px;color:var(--txt-2);padding-bottom:4px;
  font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:11px}
.kpi{padding:7px 10px;border-radius:8px;background:rgba(30,41,59,.55);border:1px solid var(--line-soft)}
.kpi .k{font-size:10.5px;color:var(--txt-4);letter-spacing:.6px}
.kpi .v{font-family:var(--mono);font-size:16px;font-weight:700;margin-top:3px;
  font-variant-numeric:tabular-nums}
.kpi.hl{background:var(--gold-dim);border-color:rgba(245,158,11,.28)}
.kpi.hl .v{color:var(--gold-2)}
.legend{display:flex;gap:20px;font-size:11.5px;color:var(--txt-4);padding:6px 2px 0;align-items:center}
.legend i{display:inline-block;width:20px;height:3px;border-radius:2px;vertical-align:middle;margin-right:7px}
#chart{flex:1;min-height:0;margin-top:6px}

/* ═══ 持仓 ═══ */
.pos-top{display:flex;align-items:center;gap:22px;padding:12px 18px 10px}
.pos-top .kv{display:flex;flex-direction:column;gap:3px}
.pos-top .k{font-size:10.5px;color:var(--txt-4);letter-spacing:.8px}
.pos-top .v{font-family:var(--mono);font-size:19px;font-weight:700;font-variant-numeric:tabular-nums}
.expo{flex:1;display:flex;flex-direction:column;gap:6px;min-width:120px}
.expo .track{height:8px;border-radius:4px;background:rgba(148,163,184,.14);overflow:hidden;display:flex}
.expo .track i{height:100%}
.expo .lg{display:flex;gap:14px;font-size:10.5px;color:var(--txt-4)}
.ptbl{flex:1;padding:0 14px 10px;overflow:hidden}
.phd,.prow{display:grid;grid-template-columns:150px 76px 124px 74px 90px 72px 1fr;
  align-items:center;padding:0 6px}
.phd{height:28px;font-size:10.5px;letter-spacing:1.1px;color:var(--txt-4);
  border-bottom:1px solid var(--line-soft)}
.prow{height:52px;border-bottom:1px solid rgba(38,51,74,.4);font-family:var(--mono);font-size:15px;
  font-variant-numeric:tabular-nums;transition:background .15s ease}
.prow:hover{background:rgba(148,163,184,.07)}
.prow .code{font-weight:600}
.prow .code small{display:block;font-family:var(--sans);font-size:11px;font-weight:400;
  color:var(--txt-4);margin-top:3px}
.prow .grp{font-family:var(--sans);font-size:11.5px;color:var(--txt-4)}
.prow .num{text-align:right;padding-right:8px}
.prow .w{display:flex;align-items:center;gap:9px;justify-content:flex-end;padding-right:4px}
.prow .w .track{width:64px;height:5px;border-radius:3px;background:rgba(148,163,184,.16);overflow:hidden}
.prow .w .track i{display:block;height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--gold),var(--gold-2))}
.empty{padding:38px;text-align:center;color:var(--txt-4);font-size:14px}

/* ═══ 底栏 ═══ */
.foot{display:flex;align-items:center;gap:12px;font-size:11.5px;color:var(--txt-4);
  border-top:1px solid var(--line-soft);padding-top:9px}
.foot b{color:var(--txt-2);font-family:var(--mono);font-weight:600}
.fchip{padding:2px 9px;border-radius:20px;border:1px solid var(--line-soft);background:rgba(30,41,59,.5);
  letter-spacing:.3px;white-space:nowrap}
.fchip.gold{color:var(--gold-2);border-color:rgba(245,158,11,.28);background:var(--gold-dim)}
.spacer{margin-left:auto}
.u{color:var(--up)}.d{color:var(--down)}.n{color:var(--txt-3)}

@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div id="stage">
  <!-- ═══ 顶栏 ═══ -->
  <div class="top">
    <div class="brand">
      <h1>收盘<em>复盘</em></h1>
      <div class="sub">QUANT SIGNAL TERMINAL</div>
    </div>
    <div class="idx-wrap" id="indices"></div>
    <div class="fresh">
      <div class="row"><span>数据日期</span><b id="dataDate"></b><span id="lagBadge"></span></div>
      <div class="row" id="metaRow"></div>
    </div>
  </div>

  <!-- ═══ 主体 ═══ -->
  <div class="main">
    <!-- 左: 信号表 -->
    <div class="panel d1">
      <div class="p-head">
        <h2>自选股信号</h2>
        <span class="tag">信号强度降序</span>
        <span class="tag">前 8 重点区</span>
        <div class="sum" id="sum"></div>
      </div>
      <div class="tbl">
        <div class="ths">
          <div></div><div>#</div><div>代码</div><div>分组</div><div style="text-align:right;padding-left:0">收盘</div>
          <div style="text-align:right">涨跌幅</div><div>信号</div><div>得分</div><div>执行状态</div>
          <div style="text-align:center">持仓</div><div style="text-align:right">MA60 / RSI / ATR%</div>
        </div>
        <div class="rows" id="rows"></div>
      </div>
    </div>

    <!-- 右: 净值 + 持仓 -->
    <div class="col right">
      <div class="panel d2">
        <div class="p-head"><h2>净值</h2><span class="tag" id="navRange"></span>
          <span class="tag warn" id="paramTag"></span></div>
        <div class="nav-body">
          <div class="nav-top">
            <span class="big" id="navLatest"></span>
            <span class="unit">净值</span>
            <span class="money" id="navMoney"></span>
          </div>
          <div class="kpis" id="kpis"></div>
          <div id="chart"></div>
          <div class="legend">
            <span><i style="background:linear-gradient(90deg,#F59E0B,#FBBF24)"></i>策略（回测口径）</span>
            <span><i style="background:#64748B"></i>沪深300</span>
            <span style="margin-left:auto" id="rangeText"></span>
          </div>
        </div>
      </div>

      <div class="panel d3">
        <div class="p-head"><h2>当前持仓</h2><span class="tag">策略模拟持仓 · 非实盘</span>
          <span class="tag" id="posCount"></span></div>
        <div class="pos-top" id="posTop"></div>
        <div class="ptbl" id="posTbl"></div>
      </div>
    </div>
  </div>

  <!-- ═══ 底栏 ═══ -->
  <div class="foot">
    <span>生成于 <b id="genAt"></b></span>
    <span>耗时 <b id="elapsed"></b></span>
    <span class="fchip gold">回测口径 · 模拟本金 ¥100,000</span>
    <span class="fchip gold">参数固化日 2026-08-07（此前区间含前视）</span>
    <span class="fchip">体制 auto</span>
    <span class="fchip">持仓/净值为策略模拟，非实盘</span>
    <span class="spacer"></span>
    <span>数据仅供复盘参考，不构成投资建议</span>
  </div>
</div>

<script>
const S = __SNAPSHOT__;
const LVL = {
  STRONG_BUY:{c:"#F6465D",t:"强买入"}, BUY:{c:"#FF7A8A",t:"买入"},
  WEAK_BUY:{c:"#F59E0B",t:"关注"},     NEUTRAL:{c:"#64748B",t:"观望"},
  WEAK_SELL:{c:"#4ADE80",t:"注意"},    SELL:{c:"#0ECB81",t:"卖出"},
  STRONG_SELL:{c:"#0B9E66",t:"强卖出"}
};
const sgn = v => (v > 0 ? "+" : "");
const cls = v => v > 0 ? "u" : (v < 0 ? "d" : "n");
const arrow = v => v > 0 ? "\u25B2" : (v < 0 ? "\u25BC" : "\u2014");
const f = (v, d = 2) => Number(v).toFixed(d);
const pc = v => sgn(v) + f(v) + "%";

/* ── 顶栏 ── */
const lag = S.data_lag_trading_days === undefined ? 0 : S.data_lag_trading_days;
document.getElementById("dataDate").textContent = S.data_date;
document.getElementById("lagBadge").innerHTML = lag > 1
  ? '<span class="badge-lag"><span class="dot"></span>距今 ' + lag + ' 个交易日 · 非最新</span>'
  : '<span class="badge-lag ok"><span class="dot"></span>最新交易日</span>';
document.getElementById("metaRow").innerHTML =
  '<span>体制</span><b>' + S.regime_mode + '</b><span>·</span><span>净值起点</span><b>' + S.nav_start + '</b>';

document.getElementById("indices").innerHTML = S.indices.map(i => {
  const c = cls(i.chg_pct), col = i.chg_pct >= 0 ? "#F6465D" : "#0ECB81";
  const bg = i.chg_pct >= 0 ? "var(--up-soft)" : "var(--down-soft)";
  const sp = i.spark || [];
  let svg = "";
  if (sp.length > 1) {
    const lo = Math.min.apply(null, sp), hi = Math.max.apply(null, sp);
    const H = 34, W = 128, span = (hi - lo) || 1;
    const px = k => (W * k / (sp.length - 1)).toFixed(1);
    const py = v => (H - 3 - (H - 9) * (v - lo) / span).toFixed(1);
    const pts = sp.map((v, k) => px(k) + "," + py(v)).join(" ");
    svg = '<svg class="spark" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">' +
      '<defs><linearGradient id="sg' + i.code.replace(/[^0-9]/g, "") + '" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0" stop-color="' + col + '" stop-opacity=".3"/>' +
      '<stop offset="1" stop-color="' + col + '" stop-opacity="0"/></linearGradient></defs>' +
      '<polygon points="0,' + H + ' ' + pts + ' ' + W + ',' + H + '" fill="url(#sg' + i.code.replace(/[^0-9]/g, "") + ')"/>' +
      '<polyline points="' + pts + '" fill="none" stroke="' + col + '" stroke-width="1.6" ' +
      'stroke-linejoin="round" stroke-linecap="round"/></svg>';
  }
  return '<div class="idx"><div class="l"><span class="nm">' + i.name + '</span>' +
    '<span class="pct ' + c + '" style="background:' + bg + '">' + arrow(i.chg_pct) + ' ' + pc(i.chg_pct) + '</span></div>' +
    '<div><div class="v ' + c + '">' + f(i.close) + '</div>' +
    '<div class="d">' + sgn(i.chg) + f(i.chg) + '</div></div>' + svg + '</div>';
}).join("");

/* ── 信号表 ── */
const sg = S.signals.summary;
document.getElementById("sum").innerHTML =
  '<span class="chip u">看涨<b>' + sg.bullish + '</b></span>' +
  '<span class="chip">中性<b>' + sg.neutral + '</b></span>' +
  '<span class="chip d">看空<b>' + sg.bearish + '</b></span>' +
  '<span class="chip g">可执行<b>' + sg.actionable + '</b></span>';

const HOLD_ICON = '<svg width="13" height="13" viewBox="0 0 24 24" fill="#FBBF24">' +
  '<path d="M12 2 3 7v10l9 5 9-5V7l-9-5zm0 2.3 6.5 3.6v7.2L12 18.7 5.5 15.1V7.9L12 4.3z"/></svg>';
const maxAbs = Math.max.apply(null, S.signals.rows.map(r => Math.abs(r.score)).concat([1]));

document.getElementById("rows").innerHTML = S.signals.rows.map((r, i) => {
  const L = LVL[r.level] || {c:"#64748B", t:r.label || r.level};
  const c = cls(r.chg_pct), ind = r.indicators || {};
  const sc = r.score, sw = Math.max(3, Math.min(100, Math.abs(sc) / maxAbs * 100));
  const sColor = sc > 0 ? "#F6465D" : (sc < 0 ? "#0ECB81" : "#64748B");
  return '<div class="tr ' + (i < 8 ? "focus" : "") + '">' +
    '<div class="bar" style="background:' + L.c + '"></div>' +
    '<div class="no">' + (i + 1) + '</div>' +
    '<div class="code">' + r.code +
    (i < 8 ? '<small>' + r.name + '</small>' : '<small class="inl">' + r.name + '</small>') + '</div>' +
    '<div class="grp">' + (r.group || "\u2014") + '</div>' +
    '<div class="num">' + f(r.close) + '</div>' +
    '<div class="pct ' + c + '">' + arrow(r.chg_pct) + f(r.chg_pct) + '%</div>' +
    '<div><span class="badge" style="background:' + L.c + '">' + L.t + '</span></div>' +
    '<div class="score"><span class="track"><i class="fill" style="width:' + sw + '%;background:' + sColor + '"></i></span>' +
    '<span class="v" style="color:' + sColor + '">' + f(sc, 1) + '</span></div>' +
    '<div class="exec ' + (r.execution.executable ? "ok" : "") + '">' + r.execution.status + '</div>' +
    '<div class="hold">' + (r.held ? HOLD_ICON : "") + '</div>' +
    '<div class="inds">' + (ind.ma60_dir || "\u2014") + ' · ' + (ind.rsi === undefined ? "\u2014" : ind.rsi) +
    ' · ' + (ind.atr_pct === undefined ? "\u2014" : ind.atr_pct) + '</div></div>';
}).join("");

/* ── 净值 ── */
const N = S.nav, RC = v => (v >= 0 ? "u" : "d");
document.getElementById("navRange").textContent =
  S.nav_start + " 起 · " + N.strategy_curve.dates.length + " 个交易日";
document.getElementById("paramTag").textContent = "参数固化 " + S.param_fixed_date;
document.getElementById("navLatest").textContent = f(N.latest, 4);
document.getElementById("navMoney").textContent =
  "\u00A5" + Number(N.latest_value).toLocaleString("zh-CN", {minimumFractionDigits: 2});
document.getElementById("kpis").innerHTML =
  '<div class="kpi"><div class="k">区间收益</div><div class="v ' + RC(N.total_return) + '">' + pc(N.total_return * 100) + '</div></div>' +
  '<div class="kpi"><div class="k">沪深300</div><div class="v ' + RC(N.benchmark_return) + '">' + pc(N.benchmark_return * 100) + '</div></div>' +
  '<div class="kpi hl"><div class="k">超额收益</div><div class="v">' + sgn(N.alpha) + f(N.alpha * 100) + '%</div></div>' +
  '<div class="kpi"><div class="k">最大回撤</div><div class="v d">' + f(N.max_drawdown * 100) + '%</div></div>';
document.getElementById("rangeText").textContent =
  N.strategy_curve.dates[0] + " → " + N.strategy_curve.dates[N.strategy_curve.dates.length - 1];

function drawChart() {
  const box = document.getElementById("chart");
  if (!box) return;
  const W = box.clientWidth, H = box.clientHeight;
  if (W < 60 || H < 60) { requestAnimationFrame(drawChart); return; }
  const a = N.strategy_curve, b = N.benchmark_curve || {values: [], dates: []};
  const all = a.values.concat(b.values.length ? b.values : [1]);
  const lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
  const pad = (hi - lo) * .1 || .01, y0 = lo - pad, y1 = hi + pad;
  const L = 44, Rg = 10, T = 12, B = 20;
  const px = (i, n) => L + (W - L - Rg) * i / Math.max(n - 1, 1);
  const py = v => T + (H - T - B) * (1 - (v - y0) / (y1 - y0));
  const line = c => c.values.map((v, i) =>
    (i ? "L" : "M") + px(i, c.values.length).toFixed(1) + "," + py(v).toFixed(1)).join(" ");
  let grid = "";
  for (let g = 0; g <= 3; g++) {
    const v = y0 + (y1 - y0) * g / 3, y = py(v);
    grid += '<line x1="' + L + '" y1="' + y.toFixed(1) + '" x2="' + (W - Rg) + '" y2="' + y.toFixed(1) +
      '" stroke="rgba(38,51,74,.75)"/>' +
      '<text x="' + (L - 8) + '" y="' + (y + 3.5).toFixed(1) + '" fill="#64748B" font-size="10.5"' +
      ' text-anchor="end" font-family="Consolas,monospace">' + v.toFixed(2) + '</text>';
  }
  const last = a.values.length - 1;
  const ticks = [0, Math.floor(last / 2), last].filter((v, i, arr) => arr.indexOf(v) === i);
  const xlab = ticks.map(i => '<text x="' + px(i, a.values.length).toFixed(1) + '" y="' + (H - 5) +
    '" fill="#64748B" font-size="10.5" text-anchor="middle" font-family="Consolas,monospace">' +
    a.dates[i] + '</text>').join("");
  const base = py(y0).toFixed(1);
  const area = "M" + px(0, a.values.length).toFixed(1) + "," + base + " " +
    a.values.map((v, i) => "L" + px(i, a.values.length).toFixed(1) + "," + py(v).toFixed(1)).join(" ") +
    " L" + px(last, a.values.length).toFixed(1) + "," + base + " Z";
  const ex = px(last, a.values.length), ey = py(a.values[last]);
  box.innerHTML = '<svg width="' + W + '" height="' + H + '">' +
    '<defs><linearGradient id="ga" x1="0" y1="0" x2="0" y2="1">' +
    '<stop offset="0" stop-color="#F59E0B" stop-opacity=".26"/>' +
    '<stop offset="1" stop-color="#F59E0B" stop-opacity="0"/></linearGradient>' +
    '<linearGradient id="gl" x1="0" y1="0" x2="1" y2="0">' +
    '<stop offset="0" stop-color="#F59E0B"/><stop offset="1" stop-color="#FBBF24"/></linearGradient></defs>' +
    '<style>@keyframes draw{to{stroke-dashoffset:0}}</style>' +
    grid +
    '<path d="' + area + '" fill="url(#ga)"/>' +
    (b.values.length ? '<path d="' + line(b) + '" fill="none" stroke="#64748B" stroke-width="1.5" ' +
      'stroke-dasharray="5 4" stroke-linecap="round"/>' : "") +
    '<path d="' + line(a) + '" fill="none" stroke="url(#gl)" stroke-width="2.4" stroke-linejoin="round" ' +
    'stroke-linecap="round" style="stroke-dasharray:4000;stroke-dashoffset:4000;' +
    'animation:draw 1.4s cubic-bezier(.16,1,.3,1) .25s forwards"/>' +
    '<circle cx="' + ex.toFixed(1) + '" cy="' + ey.toFixed(1) + '" r="8" fill="#FBBF24" opacity=".16"/>' +
    '<circle cx="' + ex.toFixed(1) + '" cy="' + ey.toFixed(1) + '" r="3.4" fill="#FBBF24" ' +
    'stroke="#0B1220" stroke-width="1.4"/>' + xlab + '</svg>';
}

/* ── 持仓 ── */
const P = S.positions, held = P.position_pct;
document.getElementById("posCount").textContent = "持仓 " + P.count + " 只";
document.getElementById("posTop").innerHTML =
  '<div class="kv"><span class="k">总仓位</span><span class="v" style="color:var(--gold-2)">' + f(held, 1) + '%</span></div>' +
  '<div class="kv"><span class="k">现金</span><span class="v" style="color:var(--txt-3)">' + f(P.cash_pct, 1) + '%</span></div>' +
  '<div class="kv"><span class="k">持仓数</span><span class="v">' + P.count + '</span></div>' +
  '<div class="expo"><div class="track"><i style="width:' + held + '%;background:linear-gradient(90deg,#F59E0B,#FBBF24)"></i>' +
  '<i style="width:' + P.cash_pct + '%;background:rgba(100,116,139,.45)"></i></div>' +
  '<div class="lg"><span>\u25A0 持仓 ' + f(held, 1) + '%</span><span>\u25A1 现金 ' + f(P.cash_pct, 1) + '%</span></div></div>';
document.getElementById("posTbl").innerHTML = P.rows.length
  ? '<div class="phd"><div>代码 / 名称</div><div>分组</div><div style="text-align:right">买入日 · 成本</div>' +
    '<div style="text-align:right">现价</div><div style="text-align:right">浮动盈亏</div>' +
    '<div style="text-align:right">收益率</div><div style="text-align:right">权重</div></div>' +
    P.rows.map(p => {
      const c = RC(p.pnl_pct), wbar = Math.max(2, p.weight_pct);
      return '<div class="prow">' +
        '<div class="code">' + p.code + '<small>' + p.name + '</small></div>' +
        '<div class="grp">' + (p.group || "\u2014") + '</div>' +
        '<div class="num" style="color:var(--txt-3)">' + p.entry_date + '<br>@' + f(p.entry_price) + '</div>' +
        '<div class="num">' + f(p.current_price) + '</div>' +
        '<div class="num ' + c + '">' + sgn(p.pnl) + f(p.pnl) + '</div>' +
        '<div class="num ' + c + '" style="font-weight:700">' + sgn(p.pnl_pct) + f(p.pnl_pct) + '%</div>' +
        '<div class="w"><span class="track"><i style="width:' + wbar + '%"></i></span>' +
        '<span>' + f(p.weight_pct, 1) + '%</span></div></div>';
    }).join("")
  : '<div class="empty">当前空仓 · 无持仓标的</div>';

/* ── 底栏 ── */
document.getElementById("genAt").textContent = S.generated_at;
document.getElementById("elapsed").textContent = S.elapsed_sec + "s";

/* ── 1920×1200 等比适配 ── */
function fit() {
  document.getElementById("stage").style.transform =
    "scale(" + Math.min(innerWidth / 1920, innerHeight / 1200) + ")";
  requestAnimationFrame(drawChart);
}
addEventListener("resize", fit);
requestAnimationFrame(() => { fit(); drawChart(); });
</script>
</body>
</html>
"""


def main():
    with open(SNAP_PATH, encoding="utf-8") as f:
        snap = json.load(f)
    # 数据滞后交易日: 工作日近似（与 DataManager._expected_last_trade_date 同口径）
    from datetime import date, timedelta
    try:
        import numpy as np
        d0 = date.fromisoformat(snap["data_date"])
        lag = int(np.busday_count(d0, date.today() + timedelta(days=1)))
    except Exception:
        lag = 0
    snap["data_lag_trading_days"] = lag

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    html = TEMPLATE.replace("__SNAPSHOT__", json.dumps(snap, ensure_ascii=False))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[render] {OUT_PATH}  ({os.path.getsize(OUT_PATH)//1024} KB)")


if __name__ == "__main__":
    main()
