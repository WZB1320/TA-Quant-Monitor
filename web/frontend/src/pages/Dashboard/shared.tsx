/** 大屏共享类型与格式化工具 */

export interface IndexRow {
  code: string
  name: string
  close: number
  chg: number
  chg_pct: number
  spark: number[]
}

export interface SignalRow {
  code: string
  name: string
  group: string
  close: number
  chg_pct: number
  level: string
  label: string
  score: number
  execution: { executable?: boolean; status: string; reason?: string }
  held: boolean
  indicators: Record<string, any>
}

export interface PositionRow {
  code: string
  name: string
  group: string
  entry_date: string
  entry_price: number
  shares: number
  cost: number
  current_price: number
  pnl: number
  pnl_pct: number
  weight_pct: number
}

export interface Curve { dates: string[]; values: number[] }

export interface DashboardSnapshot {
  generated_at: string
  data_date: string
  probed_latest_trade_date?: string
  elapsed_sec: number
  regime_mode: string
  nav_start: string
  param_fixed_date: string
  initial_capital: number
  indices: IndexRow[]
  signals: {
    summary: { bullish: number; neutral: number; bearish: number; actionable: number; total: number }
    rows: SignalRow[]
  }
  positions: { count: number; position_pct: number; cash_pct: number; rows: PositionRow[] }
  nav: {
    latest: number
    latest_value: number
    total_return: number
    max_drawdown: number
    benchmark_return: number
    alpha: number
    strategy_curve: Curve
    benchmark_curve: Curve
  }
}

export interface Freshness {
  data_date: string | null
  expected_last_trade_date?: string | null
  probed_latest_trade_date?: string | null
  stale_trade_days: number
  is_fresh: boolean
  has_snapshot: boolean
  generated_at?: string
}

export interface GenStatus {
  state: 'idle' | 'running' | 'success' | 'failed'
  stage: string
  progress: number
  error: string | null
  started_at: string | null
  finished_at: string | null
}

/** 7 级信号配色（全屏统一中国惯例: 买=红 / 卖=绿） */
export const LEVEL_META: Record<string, { c: string; t: string }> = {
  STRONG_BUY: { c: '#F6465D', t: '强买入' },
  BUY: { c: '#FF7A8A', t: '买入' },
  WEAK_BUY: { c: '#F59E0B', t: '关注' },
  NEUTRAL: { c: '#64748B', t: '观望' },
  WEAK_SELL: { c: '#4ADE80', t: '注意' },
  SELL: { c: '#0ECB81', t: '卖出' },
  STRONG_SELL: { c: '#0B9E66', t: '强卖出' },
}

export const sgn = (v: number) => (v > 0 ? '+' : '')
export const cls = (v: number) => (v > 0 ? 'dash-u' : v < 0 ? 'dash-d' : 'dash-n')
export const arrow = (v: number) => (v > 0 ? '▲' : v < 0 ? '▼' : '—')
export const fx = (v: number, d = 2) => Number(v).toFixed(d)
export const pc = (v: number) => sgn(v) + fx(v) + '%'
export const round = (v: number, d: number) => {
  const p = Math.pow(10, d)
  return Math.round(v * p) / p
}

/** 持仓标记图标 */
export const HOLD_ICON = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="#FBBF24">
    <path d="M12 2 3 7v10l9 5 9-5V7l-9-5zm0 2.3 6.5 3.6v7.2L12 18.7 5.5 15.1V7.9L12 4.3z" />
  </svg>
)

/** 迷你走势图（20 日 sparkline） */
export function Sparkline({ data, color, width = 128, height = 34 }: {
  data: number[]; color: string; width?: number; height?: number
}) {
  if (!data || data.length < 2) return null
  const lo = Math.min(...data)
  const hi = Math.max(...data)
  const span = hi - lo || 1
  const px = (k: number) => (width * k) / (data.length - 1)
  const py = (v: number) => height - 3 - ((height - 9) * (v - lo)) / span
  const pts = data.map((v, k) => `${px(k).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
  const gid = 'sg-' + color.replace(/[^0-9a-zA-Z]/g, '')
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity=".3" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${pts} ${width},${height}`} fill={`url(#${gid})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.6"
        strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
