/** 当前持仓面板：策略模拟持仓（非实盘），含总仓位/现金占比条 + 每行权重条 */
import type { PositionRow } from './shared'
import { fx, sgn } from './shared'

interface Props {
  positionPct: number
  cashPct: number
  rows: PositionRow[]
}

const RC = (v: number) => (v >= 0 ? 'dash-u' : 'dash-d')

export default function PositionPanel({ positionPct, cashPct, rows }: Props) {
  return (
    <div className="dash-panel d3">
      <div className="dash-p-head">
        <h2>当前持仓</h2>
        <span className="dash-tag">策略模拟持仓 · 非实盘</span>
        <span className="dash-tag">持仓 {rows.length} 只</span>
      </div>

      <div className="dash-pos-top">
        <div className="kv">
          <span className="k">总仓位</span>
          <span className="v" style={{ color: 'var(--d-gold-2)' }}>{fx(positionPct, 1)}%</span>
        </div>
        <div className="kv">
          <span className="k">现金</span>
          <span className="v" style={{ color: 'var(--d-txt-3)' }}>{fx(cashPct, 1)}%</span>
        </div>
        <div className="kv">
          <span className="k">持仓数</span>
          <span className="v">{rows.length}</span>
        </div>
        <div className="dash-expo">
          <div className="track">
            <i style={{ width: `${positionPct}%`, background: 'linear-gradient(90deg,#F59E0B,#FBBF24)' }} />
            <i style={{ width: `${cashPct}%`, background: 'rgba(100,116,139,.45)' }} />
          </div>
          <div className="lg">
            <span>■ 持仓 {fx(positionPct, 1)}%</span>
            <span>□ 现金 {fx(cashPct, 1)}%</span>
          </div>
        </div>
      </div>

      <div className="dash-ptbl">
        {rows.length ? (
          <>
            <div className="dash-phd">
              <div>代码 / 名称</div>
              <div>分组</div>
              <div style={{ textAlign: 'right' }}>买入日 · 成本</div>
              <div style={{ textAlign: 'right' }}>现价</div>
              <div style={{ textAlign: 'right' }}>浮动盈亏</div>
              <div style={{ textAlign: 'right' }}>收益率</div>
              <div style={{ textAlign: 'right' }}>权重</div>
            </div>
            {rows.map(p => (
              <div className="dash-prow" key={p.code}>
                <div className="code">{p.code}<small>{p.name}</small></div>
                <div className="grp">{p.group || '—'}</div>
                <div className="num" style={{ color: 'var(--d-txt-3)' }}>
                  {p.entry_date}<br />@{fx(p.entry_price)}
                </div>
                <div className="num">{fx(p.current_price)}</div>
                <div className={`num ${RC(p.pnl_pct)}`}>{sgn(p.pnl)}{fx(p.pnl)}</div>
                <div className={`num ${RC(p.pnl_pct)}`} style={{ fontWeight: 700 }}>
                  {sgn(p.pnl_pct)}{fx(p.pnl_pct)}%
                </div>
                <div className="w">
                  <span className="track">
                    <i style={{ width: `${Math.max(2, p.weight_pct)}%` }} />
                  </span>
                  <span>{fx(p.weight_pct, 1)}%</span>
                </div>
              </div>
            ))}
          </>
        ) : (
          <div className="dash-empty">当前空仓 · 无持仓标的</div>
        )}
      </div>
    </div>
  )
}
