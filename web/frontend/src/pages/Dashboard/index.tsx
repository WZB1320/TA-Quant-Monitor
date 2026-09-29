/** 收盘复盘大屏 — 主页面
 *
 * 数据链路: GET /api/dashboard/snapshot → 快照新鲜则秒开渲染
 *           过期/缺失 → POST /api/dashboard/refresh（异步）→ 轮询 /status → 重取快照
 * 四态: 无快照(首次生成) / 生成中 / 有快照(可能非最新) / 生成失败
 * 口径: 持仓=策略模拟持仓(非实盘) · 净值=回测口径(模拟本金 ¥100,000) · 体制 auto
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { dashboardApi } from '../../api'
import './dashboard.css'
import IndexBand from './IndexBand'
import SignalTable from './SignalTable'
import PositionPanel from './PositionPanel'
import NavPanel from './NavPanel'
import type { DashboardSnapshot, Freshness, GenStatus } from './shared'
import { fx } from './shared'
import { useFitScale } from './useFitScale'

const POLL_MS = 4000

/** 容器高度 = 视口 − 顶栏(56) − Content 内边距(24×2) */
const ROOT_STYLE: CSSProperties = { height: 'calc(100vh - 104px)' }

export default function DashboardPage() {
  const [snap, setSnap] = useState<DashboardSnapshot | null>(null)
  const [freshness, setFreshness] = useState<Freshness | null>(null)
  const [gen, setGen] = useState<GenStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)

  const { wrapRef, stageRef } = useFitScale()

  const stopPoll = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current)
      timer.current = null
    }
  }, [])

  const load = useCallback(async () => {
    try {
      const res = await dashboardApi.getSnapshot()
      setSnap(res.data.snapshot)
      setFreshness(res.data.freshness)
      setGen(res.data.status)
      setError(null)
      return res.data
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || '快照读取失败')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const startPoll = useCallback(() => {
    stopPoll()
    timer.current = window.setInterval(async () => {
      try {
        const res = await dashboardApi.getStatus()
        const st: GenStatus = res.data
        setGen(st)
        if (st.state !== 'running') {
          stopPoll()
          const data = await load()
          if (st.state === 'failed') {
            setError(st.error || '生成失败')
          } else if (data?.snapshot) {
            setError(null)
          }
        }
      } catch (e: any) {
        // 后端未启动等：停止轮询并提示
        stopPoll()
        setError(e?.message || '状态查询失败')
      }
    }, POLL_MS)
  }, [load, stopPoll])

  const refresh = useCallback(async () => {
    setError(null)
    try {
      await dashboardApi.refresh()
      setGen(prev => ({ ...(prev as GenStatus), state: 'running', progress: 0, stage: '已提交任务...' }))
      startPoll()
    } catch (e: any) {
      const detail = e?.response?.data?.detail
      if (e?.response?.status === 409) {
        // 已有生成在跑：直接进入轮询，不重复触发
        startPoll()
        return
      }
      setError(detail || e?.message || '触发生成失败')
    }
  }, [startPoll])

  useEffect(() => {
    load().then(data => {
      if (data?.status?.state === 'running') startPoll()   // 打开页面时正在生成
    })
    return stopPoll
  }, [load, startPoll, stopPoll])

  // 生成中：给一个轮询入口（首次进入时若后端在跑，status 已在 snapshot 里返回）
  useEffect(() => {
    if (gen?.state === 'running' && timer.current === null) startPoll()
  }, [gen, startPoll])

  const running = gen?.state === 'running'
  const actions = (
    <>
      {running ? (
        <>
          <span className="dash-progress">
            <i style={{ width: `${gen?.progress ?? 0}%` }} />
          </span>
          <span style={{ fontSize: 11.5, color: 'var(--d-txt-3)' }}>
            {gen?.stage || '生成中'} {gen?.progress ?? 0}%
          </span>
        </>
      ) : (
        <button className="dash-btn" onClick={refresh} disabled={loading}>
          ↻ {snap ? '重新生成' : '生成大屏数据'}
        </button>
      )}
    </>
  )

  /* 单一稳定根节点：任何状态下 .dash-root / .dash-stage 的 DOM 身份不变，
     保证 useFitScale 的 ResizeObserver 始终挂在活节点上（条件换根会导致缩放失效） */
  const N = snap?.nav

  const stateView = (
    <div className="dash-state">
      {loading ? (
        <h3>读取快照…</h3>
      ) : running ? (
        <>
          <h3>正在生成大屏数据</h3>
          <div className="big-progress"><i style={{ width: `${gen?.progress ?? 0}%` }} /></div>
          <p className="stage">{gen?.stage || '准备中'} · {gen?.progress ?? 0}%</p>
          <p>生成约需 1–4 分钟（取决于当日缓存是否命中），期间可离开此页。</p>
        </>
      ) : (
        <>
          <h3>尚无大屏快照</h3>
          <p>
            大屏数据需在收盘后生成一次（回测 + 35 只自选股信号，约 1–4 分钟）。
            <br />
            生成后同一交易日内再次打开将直接复用快照、秒开。
          </p>
          {error && <p className="err">{error}</p>}
          <button className="dash-btn" style={{ fontSize: 14, padding: '9px 22px' }} onClick={refresh}>
            ▶ 生成大屏数据
          </button>
        </>
      )}
    </div>
  )

  return (
    <div className="dash-root" ref={wrapRef} style={{ ...ROOT_STYLE, display: 'flex' }}>
      <div className="dash-stage" ref={stageRef} style={{ display: snap ? 'grid' : 'none' }}>
        {snap && N && (
          <>
            <IndexBand
              indices={snap.indices}
              freshness={freshness || { data_date: snap.data_date, stale_trade_days: 0, is_fresh: true, has_snapshot: true }}
              regimeMode={snap.regime_mode}
              navStart={snap.nav_start}
              actions={actions}
            />

            {error && (
              <div className="dash-banner dash-banner-float err">
                ⚠ {running ? '生成中，正在展示上次成功快照' : error}
              </div>
            )}

            <div className="dash-main">
              <SignalTable rows={snap.signals.rows} summary={snap.signals.summary} />

              <div className="dash-col right">
                <NavPanel
                  latest={N.latest}
                  latestValue={N.latest_value}
                  totalReturn={N.total_return}
                  benchmarkReturn={N.benchmark_return}
                  alpha={N.alpha}
                  maxDrawdown={N.max_drawdown}
                  strategy={N.strategy_curve}
                  benchmark={N.benchmark_curve}
                  navStart={snap.nav_start}
                  paramFixedDate={snap.param_fixed_date}
                />
                <PositionPanel
                  positionPct={snap.positions.position_pct}
                  cashPct={snap.positions.cash_pct}
                  rows={snap.positions.rows}
                />
              </div>
            </div>

            <div className="dash-foot">
              <span>生成于 <b>{snap.generated_at}</b></span>
              <span>耗时 <b>{fx(snap.elapsed_sec, 1)}s</b></span>
              <span className="dash-fchip gold">回测口径 · 模拟本金 ¥{snap.initial_capital.toLocaleString('zh-CN')}</span>
              <span className="dash-fchip gold">参数固化日 {snap.param_fixed_date}（此前区间含前视）</span>
              <span className="dash-fchip">体制 {snap.regime_mode}</span>
              <span className="dash-fchip">持仓/净值为策略模拟，非实盘</span>
              <span className="dash-spacer" />
              <span>数据仅供复盘参考，不构成投资建议</span>
            </div>
          </>
        )}
      </div>

      {!snap && stateView}
    </div>
  )
}
