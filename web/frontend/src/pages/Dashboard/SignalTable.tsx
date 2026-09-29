/** 自选股信号表：信号强度降序，前 8 行为重点区（行高放大） */
import type { SignalRow } from './shared'
import { LEVEL_META, HOLD_ICON, arrow, cls, fx } from './shared'

interface Props {
  rows: SignalRow[]
  summary: { bullish: number; neutral: number; bearish: number; actionable: number; total: number; missing?: number }
  focusCount?: number
}

export default function SignalTable({ rows, summary, focusCount = 8 }: Props) {
  const maxAbs = Math.max(...rows.map(r => Math.abs(r.score ?? 0)), 1)

  return (
    <div className="dash-panel d1">
      <div className="dash-p-head">
        <h2>自选股信号</h2>
        <span className="dash-tag">信号强度降序</span>
        <span className="dash-tag">前 {focusCount} 重点区</span>
        <div className="dash-sum">
          <span className="dash-chip u">看涨<b>{summary.bullish}</b></span>
          <span className="dash-chip">中性<b>{summary.neutral}</b></span>
          <span className="dash-chip d">看空<b>{summary.bearish}</b></span>
          <span className="dash-chip g">可操作<b>{summary.actionable}</b></span>
          {!!summary.missing && (
            <span className="dash-chip m">数据缺失<b>{summary.missing}</b></span>
          )}
        </div>
      </div>

      <div className="dash-tbl">
        <div className="dash-ths">
          <div />
          <div>#</div>
          <div>代码</div>
          <div>分组</div>
          <div style={{ textAlign: 'right', paddingLeft: 0 }}>收盘</div>
          <div style={{ textAlign: 'right' }}>涨跌幅</div>
          <div>信号</div>
          <div>得分</div>
          <div>执行状态</div>
          <div style={{ textAlign: 'center' }}>持仓</div>
          <div style={{ textAlign: 'right' }}>MA60 / RSI / ATR%</div>
        </div>

        <div className="dash-rows">
          {rows.map((r, i) => {
            const L = LEVEL_META[r.level] || { c: '#64748B', t: r.label || r.level }
            const ind = r.indicators || {}
            const hasScore = r.score !== null && r.score !== undefined
            const sw = hasScore ? Math.max(3, Math.min(100, (Math.abs(r.score as number) / maxAbs) * 100)) : 0
            const sColor = !hasScore ? '#475569'
              : (r.score as number) > 0 ? '#F6465D' : (r.score as number) < 0 ? '#0ECB81' : '#64748B'
            const focus = i < focusCount
            return (
              <div className={`dash-tr ${focus ? 'focus' : ''} ${r.missing ? 'miss' : ''}`} key={r.code}>
                <div className="bar" style={{ background: L.c }} />
                <div className="no">{i + 1}</div>
                <div className="code">
                  {r.code}
                  <small className={focus ? '' : 'inl'}>{r.name}</small>
                </div>
                <div className="grp">{r.group || '—'}</div>
                <div className="num">{fx(r.close)}</div>
                <div className={`pct ${r.chg_pct === null || r.chg_pct === undefined ? 'dash-n' : cls(r.chg_pct)}`}>
                  {r.chg_pct === null || r.chg_pct === undefined ? '—' : `${arrow(r.chg_pct)}${fx(r.chg_pct)}%`}
                </div>
                <div><span className="dash-badge" style={{ background: L.c }}>{L.t}</span></div>
                <div className="dash-score">
                  <span className="track">
                    <i className="fill" style={{ width: `${sw}%`, background: sColor }} />
                  </span>
                  <span className="v" style={{ color: sColor }}>{hasScore ? fx(r.score, 1) : '—'}</span>
                </div>
                <div className={`dash-exec ${r.execution?.executable ? 'ok' : ''} ${r.missing ? 'miss' : ''}`}
                  title={r.execution?.reason || undefined}>
                  {r.execution?.status || '—'}
                </div>
                <div className="dash-hold">{r.held ? HOLD_ICON : null}</div>
                <div className="dash-inds">
                  {ind.ma60_dir ?? '—'} · {ind.rsi ?? '—'} · {ind.atr_pct ?? '—'}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
