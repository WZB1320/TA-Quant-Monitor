/** 净值面板：最新净值 + KPI 四卡 + 策略/沪深300 双线（echarts） */
import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { Curve } from './shared'
import { fx, pc, sgn } from './shared'

interface Props {
  latest: number
  latestValue: number
  totalReturn: number
  benchmarkReturn: number
  alpha: number
  maxDrawdown: number
  strategy: Curve
  benchmark: Curve
  navStart: string
  paramFixedDate: string
}

const RC = (v: number) => (v >= 0 ? 'dash-u' : 'dash-d')

export default function NavPanel(p: Props) {
  const option = useMemo(() => {
    const dates = p.strategy.dates
    const bench = p.benchmark.values.length === dates.length
      ? p.benchmark.values
      : dates.map(() => null)   // 基准缺失/长度不一致时不画错位线
    const tickGap = Math.max(1, Math.floor(dates.length / 4))

    return {
      animationDuration: 900,
      animationEasing: 'cubicOut',
      grid: { left: 46, right: 14, top: 14, bottom: 24 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(15,23,42,.95)',
        borderColor: '#334155',
        borderWidth: 1,
        textStyle: { color: '#E2E8F0', fontSize: 12, fontFamily: 'Consolas, monospace' },
        axisPointer: { type: 'line', lineStyle: { color: 'rgba(148,163,184,.45)' } },
        valueFormatter: (v: any) => (v == null ? '—' : Number(v).toFixed(4)),
      },
      xAxis: {
        type: 'category',
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#26334A' } },
        axisTick: { show: false },
        axisLabel: {
          color: '#64748B', fontSize: 10.5, fontFamily: 'Consolas, monospace',
          interval: tickGap, hideOverlap: true,
        },
      },
      yAxis: {
        type: 'value',
        scale: true,
        splitNumber: 4,
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: 'rgba(38,51,74,.75)' } },
        axisLabel: {
          color: '#64748B', fontSize: 10.5, fontFamily: 'Consolas, monospace',
          formatter: (v: number) => v.toFixed(2),
        },
      },
      series: [
        {
          name: '策略',
          type: 'line',
          data: p.strategy.values,
          showSymbol: false,
          smooth: false,
          lineStyle: { width: 2.4, color: '#F59E0B' },
          itemStyle: { color: '#FBBF24' },
          emphasis: { focus: 'series' },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(245,158,11,.26)' },
                { offset: 1, color: 'rgba(245,158,11,0)' },
              ],
            },
          },
          markPoint: {
            symbol: 'circle',
            symbolSize: 9,
            data: [{ coord: [dates.length - 1, p.strategy.values[dates.length - 1]] }],
            itemStyle: { color: '#FBBF24', borderColor: '#0B1220', borderWidth: 1.4 },
            label: { show: false },
          },
        },
        {
          name: '沪深300',
          type: 'line',
          data: bench,
          showSymbol: false,
          smooth: false,
          connectNulls: false,
          lineStyle: { width: 1.5, color: '#64748B', type: 'dashed' },
          itemStyle: { color: '#64748B' },
        },
      ],
    }
  }, [p.strategy, p.benchmark])

  const dates = p.strategy.dates

  return (
    <div className="dash-panel d2">
      <div className="dash-p-head">
        <h2>净值</h2>
        <span className="dash-tag">{p.navStart} 起 · {dates.length} 个交易日</span>
        <span className="dash-tag warn">参数固化 {p.paramFixedDate}</span>
      </div>

      <div className="dash-nav-body">
        <div className="dash-nav-top">
          <span className="dash-big">{fx(p.latest, 4)}</span>
          <span className="dash-unit">净值</span>
          <span className="dash-money">
            ¥{Number(p.latestValue).toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
          </span>
        </div>

        <div className="dash-kpis">
          <div className="dash-kpi">
            <div className="k">区间收益</div>
            <div className={`v ${RC(p.totalReturn)}`}>{pc(p.totalReturn * 100)}</div>
          </div>
          <div className="dash-kpi">
            <div className="k">沪深300</div>
            <div className={`v ${RC(p.benchmarkReturn)}`}>{pc(p.benchmarkReturn * 100)}</div>
          </div>
          <div className="dash-kpi hl">
            <div className="k">超额收益</div>
            <div className="v">{sgn(p.alpha)}{fx(p.alpha * 100)}%</div>
          </div>
          <div className="dash-kpi">
            <div className="k">最大回撤</div>
            <div className="v dash-d">{fx(p.maxDrawdown * 100)}%</div>
          </div>
        </div>

        <div className="dash-chart">
          <ReactECharts option={option} style={{ height: '100%', width: '100%' }} notMerge lazyUpdate />
        </div>

        <div className="dash-legend">
          <span><i style={{ background: 'linear-gradient(90deg,#F59E0B,#FBBF24)' }} />策略（回测口径）</span>
          <span><i style={{ background: '#64748B' }} />沪深300</span>
          <span style={{ marginLeft: 'auto' }}>
            {dates[0]} → {dates[dates.length - 1]}
          </span>
        </div>
      </div>
    </div>
  )
}
