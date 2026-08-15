import { useEffect, useMemo, useState } from 'react'
import {
  Card, Button, DatePicker, Select, Typography, Space, Table,
  Tag, Row, Col, Statistic, Empty, Tooltip, App, Spin,
} from 'antd'
import {
  PlayCircleOutlined, ArrowUpOutlined, ArrowDownOutlined,
  MinusOutlined, StarOutlined,
} from '@ant-design/icons'
import { stockBacktestApi } from '../../api'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

interface StockItem {
  code: string
  name: string
  market?: string
  group: string
}

interface DayResult {
  date: string
  code: string
  name: string
  group: string
  close: number
  level: string
  label: string
  score: number
  confidence: number
  reason: string
  details: string
  block_detail: string
  hard_filter_blocked: boolean
  block_reason: string
  initial_level: string
  demotion_chain: string[]
  execution: {
    executable: boolean
    status: string
    reason: string | null
  }
  indicators: {
    ma60?: number
    ma60_dir?: string
    rsi?: number
    dif?: number
    dea?: number
    adx?: number
    volume_ratio?: number
    atr_pct?: number
  }
  // 持仓状态机字段
  position_state: 'IDLE' | 'HOLDING' | 'COOLDOWN'
  action: 'NONE' | 'BUY' | 'HOLD' | 'SELL' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'COOLDOWN_BLOCKED'
  entry_price: number | null
  stop_loss_price: number | null
  take_profit_price: number | null
  trailing_stop_price: number | null
  highest_price: number | null
  holding_pnl_pct: number | null
  holding_days: number | null
  cooldown_remaining: number | null
  exit_reason: string | null
  exit_pnl_pct: number | null
}

interface TradeSummary {
  total_trades: number
  win_count: number
  loss_count: number
  stop_loss_count: number
  take_profit_count: number
  signal_exit_count: number
  max_pnl_pct: number
  min_pnl_pct: number
  avg_holding_days: number
}

interface BacktestResponse {
  status: string
  message?: string
  stock?: { code: string; name: string; group: string }
  mode?: string
  summary?: {
    total: number
    bullish: number
    neutral: number
    bearish: number
    actionable: number
  }
  trade_summary?: TradeSummary
  results?: DayResult[]
}

// 信号级别颜色映射 (与信号看板一致)
const levelColorMap: Record<string, string> = {
  STRONG_BUY: '#22c55e',
  BUY: '#4ade80',
  WEAK_BUY: '#60a5fa',
  NEUTRAL: '#94a3b8',
  WEAK_SELL: '#f97316',
  SELL: '#ef4444',
  STRONG_SELL: '#dc2626',
}

const levelBgMap: Record<string, string> = {
  STRONG_BUY: '#052e16',
  BUY: '#052e16',
  WEAK_BUY: '#0c1f3d',
  NEUTRAL: '#1a1a2e',
  WEAK_SELL: '#2d1b0e',
  SELL: '#2d0f0f',
  STRONG_SELL: '#2d0f0f',
}

const actionColorMap: Record<string, string> = {
  '可执行': '#fbbf24',
  '不可执行': '#64748b',
  '无需操作': '#64748b',
  '硬过滤': '#ef4444',
  '得分不达标': '#60a5fa',
  '冷却期内': '#94a3b8',
  '连亏暂停': '#f87171',
  '信号去重': '#94a3b8',
}

// 持仓状态颜色映射
const positionStateColorMap: Record<string, string> = {
  IDLE: '#64748b',
  HOLDING: '#22c55e',
  COOLDOWN: '#f59e0b',
}

const positionStateBgMap: Record<string, string> = {
  IDLE: '#1a1a2e',
  HOLDING: '#052e16',
  COOLDOWN: '#2d1b0e',
}

const positionStateLabelMap: Record<string, string> = {
  IDLE: '空仓',
  HOLDING: '持仓',
  COOLDOWN: '冷却',
}

// 操作类型颜色 + 标签
const actionTypeColorMap: Record<string, string> = {
  BUY: '#22c55e',
  HOLD: '#94a3b8',
  SELL: '#3b82f6',
  STOP_LOSS: '#dc2626',
  TAKE_PROFIT: '#fbbf24',
  NONE: '#475569',
  COOLDOWN_BLOCKED: '#f59e0b',
}

const actionTypeLabelMap: Record<string, string> = {
  BUY: '买入',
  HOLD: '持有',
  SELL: '卖出',
  STOP_LOSS: '止损',
  TAKE_PROFIT: '止盈',
  NONE: '-',
  COOLDOWN_BLOCKED: '冷却中',
}

const MODE_LABELS: Record<string, string> = { base: '基础', trending: '趋势上涨', ranging: '震荡' }
const MODE_COLORS: Record<string, string> = { base: '#64748b', trending: '#3b82f6', ranging: '#f59e0b' }

const marketLabel = (m?: string) => {
  if (m === 'sh') return '沪'
  if (m === 'sz') return '深'
  if (m === 'bj') return '京'
  return m || ''
}

const marketColor = (m?: string) => {
  if (m === 'sh') return '#ef4444'
  if (m === 'sz') return '#3b82f6'
  if (m === 'bj') return '#f59e0b'
  return '#64748b'
}

export default function StockBacktestPage() {
  const { message } = App.useApp()
  const [stocks, setStocks] = useState<StockItem[]>([])
  const [selectedCode, setSelectedCode] = useState<string>('')
  const [selectedMode, setSelectedMode] = useState<string>('base')
  const [dateRange, setDateRange] = useState<[any, any]>([null, null])
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<BacktestResponse | null>(null)

  useEffect(() => {
    loadStocks()
  }, [])

  const loadStocks = async () => {
    try {
      const { data } = await stockBacktestApi.getStocks()
      setStocks(data.stocks || [])
    } catch (e: any) {
      message.error('获取自选股列表失败: ' + (e?.message || ''))
    }
  }

  // 股票选项: 按分组归类
  const stockOptions = useMemo(() => {
    const groupMap: Record<string, StockItem[]> = {}
    stocks.forEach(s => {
      if (!groupMap[s.group]) groupMap[s.group] = []
      groupMap[s.group].push(s)
    })
    return Object.entries(groupMap).map(([group, items]) => ({
      label: `${group} (${items.length})`,
      options: items.map(s => ({
        label: (
          <Space size={4}>
            <Text style={{ fontFamily: 'monospace', color: '#93c5fd', fontSize: 12 }}>{s.code}</Text>
            <Text style={{ color: '#e2e8f0', fontSize: 12 }}>{s.name}</Text>
            <Tag color={marketColor(s.market)} style={{ fontSize: 10, lineHeight: '16px', borderRadius: 3 }}>
              {marketLabel(s.market)}
            </Tag>
          </Space>
        ),
        value: s.code,
      })),
    }))
  }, [stocks])

  // 筛选
  const [levelFilter, setLevelFilter] = useState<'all' | 'bullish' | 'bearish' | 'neutral' | 'actionable' | 'holding' | 'trades'>('all')
  const filteredResults = useMemo(() => {
    let list = result?.results || []
    if (levelFilter === 'bullish') list = list.filter(r => ['STRONG_BUY', 'BUY', 'WEAK_BUY'].includes(r.level))
    if (levelFilter === 'bearish') list = list.filter(r => ['WEAK_SELL', 'SELL', 'STRONG_SELL'].includes(r.level))
    if (levelFilter === 'neutral') list = list.filter(r => r.level === 'NEUTRAL')
    if (levelFilter === 'actionable') list = list.filter(r => r.execution.executable)
    if (levelFilter === 'holding') list = list.filter(r => r.position_state === 'HOLDING')
    if (levelFilter === 'trades') list = list.filter(r => ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT'].includes(r.action))
    return list
  }, [result, levelFilter])

  const handleRun = async () => {
    if (!selectedCode) {
      message.warning('请先选择一只股票')
      return
    }
    setRunning(true)
    try {
      const startDate = dateRange?.[0]?.format('YYYY-MM-DD')
      const endDate = dateRange?.[1]?.format('YYYY-MM-DD')
      const { data } = await stockBacktestApi.run({
        code: selectedCode,
        mode: selectedMode,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      })
      if (data.status === 'error') {
        message.error(data.message)
        return
      }
      setResult(data)
      message.success(`分析完成 — ${data.stock?.name || selectedCode}，共 ${data.summary?.total || 0} 天`)
    } catch (e: any) {
      message.error('分析失败: ' + (e?.message || '未知错误'))
    } finally {
      setRunning(false)
    }
  }

  // 表格列 (与信号看板一致)
  const columns = [
    {
      title: '日期',
      dataIndex: 'date',
      width: 100,
      fixed: 'left' as const,
      render: (v: string) => <Text style={{ fontFamily: 'monospace', color: '#e2e8f0', fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '代码',
      dataIndex: 'code',
      width: 100,
      render: (code: string, r: DayResult) => (
        <Space size={4}>
          <Text style={{ color: '#93c5fd', fontFamily: 'monospace', fontSize: 12 }}>{code}</Text>
          <Tag color={marketColor(r.group === '_default' ? '' : 'sz')} style={{ borderRadius: 3, fontSize: 10, lineHeight: '16px' }}>
            {r.group === '_default' ? '默认' : r.group.slice(0, 2)}
          </Tag>
        </Space>
      ),
    },
    {
      title: '名称',
      dataIndex: 'name',
      width: 90,
      render: (name: string) => <Text style={{ color: '#e2e8f0', fontSize: 12 }}>{name}</Text>,
    },
    {
      title: '收盘价',
      dataIndex: 'close',
      width: 80,
      align: 'right' as const,
      render: (v: number) => (
        <Text style={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 12 }}>{v.toFixed(2)}</Text>
      ),
    },
    {
      title: '持仓状态',
      dataIndex: 'position_state',
      width: 100,
      align: 'center' as const,
      render: (state: string, r: DayResult) => {
        const color = positionStateColorMap[state] || '#64748b'
        const bg = positionStateBgMap[state] || '#1a1a2e'
        const label = positionStateLabelMap[state] || state
        let suffix = ''
        if (state === 'HOLDING' && r.holding_pnl_pct != null) {
          suffix = ` ${r.holding_pnl_pct > 0 ? '+' : ''}${r.holding_pnl_pct}%`
        } else if (state === 'COOLDOWN' && r.cooldown_remaining != null) {
          suffix = ` ${r.cooldown_remaining}天`
        }
        return (
          <Tag style={{ background: bg, color, border: `1px solid ${color}40`, borderRadius: 4, fontSize: 11, fontWeight: 600 }}>
            {label}{suffix}
          </Tag>
        )
      },
    },
    {
      title: '操作',
      dataIndex: 'action',
      width: 80,
      align: 'center' as const,
      render: (action: string, r: DayResult) => {
        if (!action || action === 'NONE') return <Text style={{ color: '#475569', fontSize: 12 }}>-</Text>
        const color = actionTypeColorMap[action] || '#64748b'
        const label = actionTypeLabelMap[action] || action
        const tooltip = r.exit_reason
          ? `${r.exit_reason}${r.exit_pnl_pct != null ? ` (${r.exit_pnl_pct > 0 ? '+' : ''}${r.exit_pnl_pct}%)` : ''}`
          : label
        return (
          <Tooltip title={tooltip} overlayStyle={{ maxWidth: 300 }}>
            <Text style={{ color, fontWeight: 700, fontSize: 12 }}>{label}</Text>
          </Tooltip>
        )
      },
    },
    {
      title: '止损价',
      dataIndex: 'stop_loss_price',
      width: 95,
      align: 'right' as const,
      render: (v: number | null, r: DayResult) => {
        if (v == null) return <Text style={{ color: '#475569', fontSize: 12 }}>-</Text>
        const space = ((r.close - v) / r.close * 100).toFixed(1)
        const spaceColor = parseFloat(space) < 3 ? '#ef4444' : '#64748b'
        const parts = [`止损: ${v.toFixed(2)}`]
        if (r.take_profit_price) parts.push(`目标止盈: ${r.take_profit_price.toFixed(2)}`)
        if (r.trailing_stop_price) parts.push(`移动止盈: ${r.trailing_stop_price.toFixed(2)}`)
        if (r.highest_price) parts.push(`最高: ${r.highest_price.toFixed(2)}`)
        const tip = parts.join(' | ')
        return (
          <Tooltip title={tip} overlayStyle={{ maxWidth: 280 }}>
            <Text style={{ color: '#f87171', fontFamily: 'monospace', fontSize: 12 }}>{v.toFixed(2)}</Text>
            <Text style={{ color: spaceColor, fontFamily: 'monospace', fontSize: 10, marginLeft: 4 }}>({space}%)</Text>
          </Tooltip>
        )
      },
    },
    {
      title: '信号',
      dataIndex: 'level',
      width: 90,
      align: 'center' as const,
      render: (level: string, r: DayResult) => (
        <Tag
          style={{
            background: levelBgMap[level] || '#1a1a2e',
            color: levelColorMap[level] || '#94a3b8',
            border: `1px solid ${levelColorMap[level] || '#94a3b8'}40`,
            borderRadius: 4,
            fontWeight: 600,
            fontSize: 11,
          }}
        >
          {r.label}
        </Tag>
      ),
    },
    {
      title: '得分',
      dataIndex: 'score',
      width: 75,
      align: 'right' as const,
      sorter: (a: DayResult, b: DayResult) => a.score - b.score,
      render: (score: number) => (
        <Text style={{
          color: score > 0 ? '#4ade80' : score < 0 ? '#f87171' : '#94a3b8',
          fontFamily: 'monospace',
          fontWeight: 600,
          fontSize: 12,
        }}>
          {score > 0 ? '+' : ''}{score}
        </Text>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      width: 75,
      align: 'right' as const,
      render: (v: number) => (
        <Text style={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: 12 }}>
          {(v * 100).toFixed(0)}%
        </Text>
      ),
    },
    {
      title: '执行状态',
      dataIndex: 'execution',
      width: 100,
      align: 'center' as const,
      render: (_: any, r: DayResult) => {
        const exec = r.execution
        if (!exec) return <Text style={{ color: '#64748b', fontSize: 12 }}>-</Text>
        const isExecutable = exec.executable
        const status = exec.status
        const color = isExecutable ? '#fbbf24' : actionColorMap[status] || '#64748b'
        return (
          <Tooltip title={exec.reason || (isExecutable ? '可执行' : '不可执行')} overlayStyle={{ maxWidth: 300 }}>
            <Text style={{ color, fontWeight: isExecutable ? 700 : 400, fontSize: 12 }}>
              {isExecutable ? <StarOutlined style={{ marginRight: 4 }} /> : null}
              {status}
            </Text>
          </Tooltip>
        )
      },
    },
    {
      title: '判断缘由',
      dataIndex: 'block_detail',
      width: 300,
      render: (_: string, r: DayResult) => {
        let summary = ''
        let color = '#94a3b8'

        const scoreStr = `得分${r.score >= 0 ? '+' : ''}${r.score.toFixed(0)}`
        const confStr = `置信度${(r.confidence * 100).toFixed(0)}%`

        if (r.hard_filter_blocked && r.block_reason) {
          summary = `硬过滤拦截: ${r.block_reason} (${scoreStr})`
          color = '#ef4444'
        } else if (r.block_detail) {
          summary = `${r.block_detail} (${scoreStr}, ${confStr})`
          color = '#fbbf24'
        } else if (r.level !== 'NEUTRAL') {
          const demotionInfo = r.demotion_chain && r.demotion_chain.length > 0
            ? ` | 降级: ${r.demotion_chain.join(' → ')}`
            : ''
          summary = `${r.reason}${demotionInfo} | ${confStr}`
          color = r.level.includes('BUY') ? '#22c55e' : r.level.includes('SELL') ? '#ef4444' : '#94a3b8'
        } else {
          summary = r.reason || '-'
        }

        if (!summary || summary === '-') {
          return <Text style={{ color: '#475569', fontSize: 12 }}>-</Text>
        }

        const tooltipContent = r.details ? (
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 12, lineHeight: 1.6, color: '#e2e8f0' }}>
            {r.details}
          </pre>
        ) : summary

        return (
          <Tooltip title={tooltipContent} overlayStyle={{ maxWidth: 520 }}>
            <Text style={{ color, fontSize: 12, display: 'block', lineHeight: 1.4 }}>
              {summary}
            </Text>
          </Tooltip>
        )
      },
    },
  ]

  // 展开行: 指标明细
  const expandedRowRender = (r: DayResult) => {
    const ind = r.indicators || {}
    return (
      <div style={{ padding: '8px 16px', background: '#0a1628', borderRadius: 6 }}>
        <Row gutter={[24, 8]}>
          <Col span={24}>
            <Text style={{ color: '#e2e8f0', fontSize: 13 }}>{r.details}</Text>
          </Col>
          {ind.rsi != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>RSI</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{ind.rsi}</Text></Col>
          )}
          {ind.dif != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>DIF/DEA</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>
                {ind.dif} / {ind.dea}
              </Text></Col>
          )}
          {ind.ma60 != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>MA60</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>
                {ind.ma60}
                <Tag color={ind.ma60_dir === '多头' ? 'green' : 'red'} style={{ marginLeft: 6, borderRadius: 3, fontSize: 10 }}>
                  {ind.ma60_dir}
                </Tag>
              </Text></Col>
          )}
          {ind.adx != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>ADX</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{ind.adx}</Text></Col>
          )}
          {ind.volume_ratio != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>量比</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{ind.volume_ratio}</Text></Col>
          )}
          {ind.atr_pct != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>ATR%</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: 'monospace' }}>{ind.atr_pct}%</Text></Col>
          )}
          {r.hard_filter_blocked && (
            <Col span={24}>
              <Tag color="red" style={{ borderRadius: 3 }}>硬过滤拦截: {r.block_reason}</Tag>
            </Col>
          )}
        </Row>
      </div>
    )
  }

  const s = result?.summary

  return (
    <div style={{ maxWidth: 1400 }}>
      {/* 顶部 */}
      <div style={{ marginBottom: 20 }}>
        <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>个股信号回测</Title>
        <Text style={{ color: '#64748b', fontSize: 13 }}>
          选择自选股中的单只个股, 查看策略在指定时间段内逐日产生的信号结果
        </Text>
      </div>

      {/* 参数栏 */}
      <Card
        size="small"
        style={{ background: '#0f1729', borderColor: '#1e2d45', marginBottom: 16 }}
        styles={{ header: { borderBottomColor: '#1e2d45' } }}
      >
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>选择股票</Text>
            <Select
              showSearch
              placeholder="从自选股中选择"
              style={{ width: 280 }}
              value={selectedCode || undefined}
              onChange={setSelectedCode}
              options={stockOptions}
              filterOption={(input, option) => {
                const v = (option as { value?: string } | undefined)?.value
                return v?.toLowerCase().includes(input.toLowerCase()) ?? false
              }}
              disabled={running}
              size="small"
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>策略模式</Text>
            <Select
              size="small"
              value={selectedMode}
              onChange={setSelectedMode}
              style={{ width: 120 }}
              disabled={running}
              options={[
                { value: 'base', label: '基础' },
                { value: 'trending', label: '趋势上涨' },
                { value: 'ranging', label: '震荡' },
              ]}
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>回测区间</Text>
            <RangePicker
              size="small"
              style={{ width: 260 }}
              value={dateRange}
              onChange={(dates) => setDateRange(dates as [any, any])}
              disabled={running}
            />
          </div>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleRun}
            loading={running}
            size="small"
            style={{ height: 32, minWidth: 100 }}
          >
            {running ? '分析中...' : '开始分析'}
          </Button>
        </div>
      </Card>

      {/* 结果区域 */}
      {running && (
        <Card style={{ background: '#0f1729', borderColor: '#1e2d45', marginBottom: 16, textAlign: 'center' }}>
          <Spin tip="逐日运行信号分析中, 请稍候..." size="large">
            <div style={{ height: 80 }} />
          </Spin>
        </Card>
      )}

      {result && s && !running ? (
        <>
          {/* 股票信息 + 汇总 */}
          <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            {result.stock && (
              <>
                <Text style={{ color: '#e2e8f0', fontSize: 15, fontWeight: 600 }}>
                  {result.stock.name}
                </Text>
                <Text style={{ color: '#93c5fd', fontFamily: 'monospace', fontSize: 13 }}>
                  {result.stock.code}
                </Text>
                <Tag color="#1e2d45" style={{ color: '#93c5fd', borderRadius: 4, fontSize: 11 }}>
                  {result.stock.group}
                </Tag>
              </>
            )}
            <Tag color={MODE_COLORS[selectedMode]} style={{ fontSize: 12, padding: '0 8px' }}>
              {MODE_LABELS[selectedMode]}
            </Tag>
          </div>

          {/* 统计概览 */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={5}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>分析天数</Text>}
                  value={s.total}
                  suffix="天"
                  valueStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 20 }}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>看多信号</Text>}
                  value={s.bullish}
                  suffix="天"
                  valueStyle={{ color: '#22c55e', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<ArrowUpOutlined />}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>观望</Text>}
                  value={s.neutral}
                  suffix="天"
                  valueStyle={{ color: '#94a3b8', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<MinusOutlined />}
                />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>看空信号</Text>}
                  value={s.bearish}
                  suffix="天"
                  valueStyle={{ color: '#ef4444', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<ArrowDownOutlined />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>可执行</Text>}
                  value={s.actionable}
                  suffix="天"
                  valueStyle={{ color: '#fbbf24', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<StarOutlined />}
                />
              </Card>
            </Col>
          </Row>

          {/* 交易摘要 (持仓状态机) */}
          {result.trade_summary && result.trade_summary.total_trades > 0 && (
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>总交易</Text>}
                    value={result.trade_summary.total_trades}
                    suffix="笔"
                    valueStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 20 }}
                  />
                </Card>
              </Col>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>止损触发</Text>}
                    value={result.trade_summary.stop_loss_count}
                    suffix="笔"
                    valueStyle={{ color: '#dc2626', fontFamily: 'monospace', fontSize: 20 }}
                  />
                </Card>
              </Col>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>止盈平仓</Text>}
                    value={result.trade_summary.take_profit_count}
                    suffix="笔"
                    valueStyle={{ color: '#fbbf24', fontFamily: 'monospace', fontSize: 20 }}
                  />
                </Card>
              </Col>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>信号卖出</Text>}
                    value={result.trade_summary.signal_exit_count}
                    suffix="笔"
                    valueStyle={{ color: '#3b82f6', fontFamily: 'monospace', fontSize: 20 }}
                  />
                </Card>
              </Col>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>最大盈/亏</Text>}
                    value={`${result.trade_summary.max_pnl_pct > 0 ? '+' : ''}${result.trade_summary.max_pnl_pct}% / ${result.trade_summary.min_pnl_pct}%`}
                    valueStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 14 }}
                  />
                </Card>
              </Col>
              <Col span={4}>
                <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                  <Statistic
                    title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>平均持仓</Text>}
                    value={result.trade_summary.avg_holding_days}
                    suffix="天"
                    valueStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: 20 }}
                  />
                </Card>
              </Col>
            </Row>
          )}
          <div style={{ marginBottom: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {[
              { key: 'all', label: `全部 (${s.total})` },
              { key: 'bullish', label: `看多 (${s.bullish})` },
              { key: 'bearish', label: `看空 (${s.bearish})` },
              { key: 'neutral', label: `观望 (${s.neutral})` },
              { key: 'actionable', label: `可执行 (${s.actionable})` },
              { key: 'holding', label: `持仓中 (${(result.results || []).filter(r => r.position_state === 'HOLDING').length})` },
              { key: 'trades', label: `交易点 (${(result.results || []).filter(r => ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT'].includes(r.action)).length})` },
            ].map(opt => (
              <Button
                key={opt.key}
                size="small"
                type={levelFilter === opt.key ? 'primary' : 'default'}
                onClick={() => setLevelFilter(opt.key as any)}
                style={levelFilter === opt.key ? {} : { background: '#1e2d45', borderColor: '#1e2d45', color: '#94a3b8' }}
              >
                {opt.label}
              </Button>
            ))}
          </div>

          {/* 信号结果表格 */}
          <Card
            size="small"
            title={<Text style={{ color: '#e2e8f0' }}>逐日信号结果 ({filteredResults.length}条)</Text>}
            style={{ background: '#0f1729', borderColor: '#1e2d45' }}
            styles={{ header: { borderBottomColor: '#1e2d45' } }}
          >
            <Table
              dataSource={filteredResults}
              columns={columns}
              rowKey={(record) => record.date}
              size="small"
              pagination={{ pageSize: 20, size: 'small', showSizeChanger: true, pageSizeOptions: ['20', '50', '100'] }}
              style={{ background: 'transparent' }}
              scroll={{ x: 1300 }}
              expandable={{ expandedRowRender, rowExpandable: () => true }}
              rowClassName={(record) => {
                if (record.execution.executable) return 'actionable-row'
                return ''
              }}
            />
          </Card>
        </>
      ) : (
        !running && (
          <Card style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
            <Empty description="选择股票并点击「开始分析」查看逐日信号结果" />
          </Card>
        )
      )}
    </div>
  )
}
