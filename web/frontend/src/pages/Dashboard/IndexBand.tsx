/** 顶栏：品牌 + 指数卡（含 20 日 sparkline）+ 数据新鲜度 */
import type { ReactNode } from 'react'
import type { IndexRow, Freshness } from './shared'
import { Sparkline, arrow, cls, fx, pc, sgn } from './shared'

interface Props {
  indices: IndexRow[]
  freshness: Freshness
  regimeMode: string
  navStart: string
  actions?: ReactNode
}

export default function IndexBand({ indices, freshness, regimeMode, navStart, actions }: Props) {
  const lag = freshness.stale_trade_days ?? 0
  const dateLabel = freshness.probed_latest_trade_date
    ? `${freshness.probed_latest_trade_date}（源端最新）`
    : '—'

  return (
    <div className="dash-top">
      <div className="dash-brand">
        <h1>收盘<em>复盘</em></h1>
        <div className="sub">QUANT SIGNAL TERMINAL</div>
      </div>

      <div className="dash-idx-wrap">
        {indices.map(i => {
          const up = i.chg_pct >= 0
          const bg = up ? 'var(--d-up-soft)' : 'var(--d-down-soft)'
          return (
            <div className="dash-idx" key={i.code}>
              <div className="l">
                <span className="nm">{i.name}</span>
                <span className={`pct ${cls(i.chg_pct)}`} style={{ background: bg }}>
                  {arrow(i.chg_pct)} {pc(i.chg_pct)}
                </span>
              </div>
              <div>
                <div className={`v ${cls(i.chg_pct)}`}>{fx(i.close)}</div>
                <div className="d">{sgn(i.chg)}{fx(i.chg)}</div>
              </div>
              <Sparkline data={i.spark} color={up ? '#F6465D' : '#0ECB81'} />
            </div>
          )
        })}
      </div>

      <div className="dash-fresh">
        <div className="row">
          <span>数据日期</span>
          <b>{freshness.data_date || '—'}</b>
          {lag > 1 ? (
            <span className="dash-badge-lag"><span className="dash-dot" />距今 {lag} 个交易日 · 非最新</span>
          ) : (
            <span className="dash-badge-lag ok"><span className="dash-dot" />最新交易日</span>
          )}
        </div>
        <div className="row">
          <span>源端</span><b>{dateLabel}</b>
          <span>·</span>
          <span>体制</span><b>{regimeMode}</b>
          <span>·</span>
          <span>净值起点</span><b>{navStart}</b>
        </div>
        {actions && <div className="dash-actions">{actions}</div>}
      </div>
    </div>
  )
}
