import { useEffect, useMemo, useState } from 'react'
import {
  Card, Button, DatePicker, InputNumber, Select, Typography, Space, Table,
  Tag, Statistic, Row, Col, Progress, App, Spin,
} from 'antd'
import {
  PlayCircleOutlined, DownloadOutlined, ArrowUpOutlined,
  ArrowDownOutlined, TrophyOutlined, WarningOutlined,
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { useWatchlistStore } from '../../stores/watchlist'
import { backtestApi } from '../../api'

const { Title, Text } = Typography
const { RangePicker } = DatePicker

export default function BacktestPage() {
  const { message } = App.useApp()
  const MODE_LABELS: Record<string, string> = { base: '基础', trending: '趋势上涨', ranging: '震荡' }
  const MODE_COLORS: Record<string, string> = { base: '#64748b', trending: '#3b82f6', ranging: '#f59e0b' }
  const { groups, fetchWatchlist } = useWatchlistStore()
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressText, setProgressText] = useState('')
  const [result, setResult] = useState<any>(null)
  const [allTrades, setAllTrades] = useState<any[]>([])
  const [selectedGroups, setSelectedGroups] = useState<string[]>([])
  const [selectedMode, setSelectedMode] = useState<string>('base')
  const [dateRange, setDateRange] = useState<[any, any]>([null, null])
  const [initialCapital, setInitialCapital] = useState<number>(100000)

  useEffect(() => {
    fetchWatchlist()
  }, [])

  // 根据选中分组过滤交易记录
  const filteredTrades = useMemo(() => {
    if (selectedGroups.length === 0) return allTrades
    return allTrades.filter(t => selectedGroups.includes(t.group))
  }, [allTrades, selectedGroups])

  // 从后端返回的 daily_values 构建图表数据
  const chartData = useMemo(() => {
    if (!result?.daily_values) return null
    const { dates, values } = result.daily_values
    if (!dates || !values || dates.length === 0) return null

    // 计算回撤曲线 (从真实净值序列推导)
    const drawdown: number[] = []
    let peak = values[0]
    for (let i = 0; i < values.length; i++) {
      if (values[i] > peak) peak = values[i]
      const dd = ((values[i] - peak) / peak) * 100
      drawdown.push(Math.round(dd * 100) / 100)
    }

    // 基准净值 (如果后端返回了基准数据, 否则用初始资金直线)
    const benchmark = values.map((_, i) => {
      // 简单基准: 从初始资金按 benchmark_return 线性增长
      const benchReturn = result.metrics?.benchmark_return || 0
      const progress = i / (values.length - 1 || 1)
      return Math.round(result.metrics?.initial_capital * (1 + benchReturn * progress))
    })

    return { dates, values, benchmark, drawdown }
  }, [result])

  // 净值曲线
  const equityOption = useMemo(() => {
    if (!chartData) return {}
    const { dates, values, benchmark } = chartData
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f1729',
        borderColor: '#1e2d45',
        textStyle: { color: '#e2e8f0', fontSize: 12 },
      },
      legend: {
        data: ['策略净值', '基准净值'],
        textStyle: { color: '#94a3b8' },
        top: 0,
      },
      grid: { top: 40, right: 20, bottom: 30, left: 60 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#1e2d45' } },
        axisLabel: { color: '#64748b', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#1e2d45' } },
        axisLabel: { color: '#64748b', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e2d45', type: 'dashed' } },
      },
      series: [
        {
          name: '策略净值',
          type: 'line',
          data: values,
          smooth: true,
          lineStyle: { color: '#3b82f6', width: 2 },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(59,130,246,0.3)' },
                { offset: 1, color: 'rgba(59,130,246,0)' },
              ],
            },
          },
          itemStyle: { color: '#3b82f6' },
        },
        {
          name: '基准净值',
          type: 'line',
          data: benchmark,
          smooth: true,
          lineStyle: { color: '#64748b', width: 1, type: 'dashed' },
          itemStyle: { color: '#64748b' },
        },
      ],
    }
  }, [chartData])

  // 回撤曲线
  const drawdownOption = useMemo(() => {
    if (!chartData) return {}
    const { dates, drawdown } = chartData
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f1729',
        borderColor: '#1e2d45',
        textStyle: { color: '#e2e8f0', fontSize: 12 },
        formatter: (params: any) => {
          const p = params[0]
          return `${p.axisValue}<br/>回撤: ${p.value.toFixed(2)}%`
        },
      },
      grid: { top: 10, right: 20, bottom: 30, left: 60 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#1e2d45' } },
        axisLabel: { color: '#64748b', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#1e2d45' } },
        axisLabel: { color: '#64748b', fontSize: 10, formatter: '{value}%' },
        splitLine: { lineStyle: { color: '#1e2d45', type: 'dashed' } },
      },
      series: [
        {
          type: 'line',
          data: drawdown,
          lineStyle: { color: '#ef4444', width: 1.5 },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(239,68,68,0)' },
                { offset: 1, color: 'rgba(239,68,68,0.2)' },
              ],
            },
          },
          itemStyle: { color: '#ef4444' },
        },
      ],
    }
  }, [chartData])

  const handleRun = async () => {
    setRunning(true)
    setProgress(5)
    setProgressText('提交回测请求...')

    // 进度轮询: 后端回测是同步阻塞的, 轮询 /status 获取实时进度
    let pollTimer: ReturnType<typeof setInterval> | null = null
    const startPolling = () => {
      pollTimer = setInterval(async () => {
        try {
          const statusRes = await backtestApi.getStatus()
          const s = statusRes.data
          if (s.running) {
            setProgress(Math.max(s.progress || 15, 15))
            setProgressText(s.progress_text || '计算中...')
          }
        } catch {
          // 轮询失败不中断主流程
        }
      }, 2000)
    }
    const stopPolling = () => {
      if (pollTimer) {
        clearInterval(pollTimer)
        pollTimer = null
      }
    }

    try {
      const startDate = dateRange?.[0]?.format('YYYY-MM-DD')
      const endDate = dateRange?.[1]?.format('YYYY-MM-DD')

      setProgress(15)
      setProgressText('后端正在加载股票数据...')
      startPolling()

      const res = await backtestApi.run({
        groups: selectedGroups,
        mode: selectedMode,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        initial_capital: initialCapital,
      })

      stopPolling()
      setProgress(95)
      setProgressText('接收回测结果...')

      if (res.data.status === 'error') {
        message.error(res.data.message)
        return
      }

      setResult(res.data)
      setAllTrades(res.data.trades || [])
      setProgress(100)
      setProgressText('完成')
      message.success(`回测完成 — ${MODE_LABELS[selectedMode]}模式，共 ${res.data.trades?.length || 0} 笔交易`)
    } catch (e: any) {
      message.error('回测失败: ' + (e?.message || '未知错误'))
    } finally {
      stopPolling()
      setRunning(false)
    }
  }

  // 交易明细列
  const tradeColumns = [
    {
      title: '#',
      width: 40,
      render: (_: any, __: any, i: number) => <Text style={{ color: '#64748b' }}>{i + 1}</Text>,
    },
    {
      title: '代码',
      dataIndex: 'symbol',
      width: 80,
      render: (v: string) => <Text style={{ fontFamily: 'monospace', color: '#93c5fd' }}>{v}</Text>,
    },
    {
      title: '名称',
      dataIndex: 'name',
      width: 90,
    },
    {
      title: '分组',
      dataIndex: 'group',
      width: 90,
      render: (v: string) => (
        <Tag color="#1e2d45" style={{ color: '#93c5fd', borderRadius: 4, fontSize: 11 }}>{v}</Tag>
      ),
    },
    {
      title: '买入日',
      dataIndex: 'entry_date',
      width: 100,
      render: (v: string) => <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '买入价',
      dataIndex: 'entry_price',
      width: 75,
      render: (v: number) => <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{v?.toFixed(2)}</Text>,
    },
    {
      title: '卖出日',
      dataIndex: 'exit_date',
      width: 100,
      render: (v: string) => <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{v || '-'}</Text>,
    },
    {
      title: '卖出价',
      dataIndex: 'exit_price',
      width: 75,
      render: (v: number) => <Text style={{ fontFamily: 'monospace', fontSize: 12 }}>{v ? v.toFixed(2) : '-'}</Text>,
    },
    {
      title: '股数',
      dataIndex: 'shares',
      width: 70,
      render: (v: number) => <Text style={{ fontFamily: 'monospace' }}>{v?.toLocaleString()}</Text>,
    },
    {
      title: '成本',
      dataIndex: 'cost',
      width: 90,
      render: (v: number) => <Text style={{ fontFamily: 'monospace' }}>{v?.toLocaleString()}</Text>,
    },
    {
      title: '盈亏%',
      dataIndex: 'pnl_pct',
      width: 80,
      render: (v: number) => (
        <Text style={{ color: v >= 0 ? '#22c55e' : '#ef4444', fontFamily: 'monospace', fontWeight: 600 }}>
          {v >= 0 ? '+' : ''}{v.toFixed(1)}%
        </Text>
      ),
      sorter: (a: any, b: any) => a.pnl_pct - b.pnl_pct,
    },
    {
      title: '盈亏额',
      dataIndex: 'pnl',
      width: 90,
      render: (v: number) => (
        <Text style={{ color: v >= 0 ? '#22c55e' : '#ef4444', fontFamily: 'monospace' }}>
          {v >= 0 ? '+' : ''}{v.toLocaleString()}
        </Text>
      ),
      sorter: (a: any, b: any) => a.pnl - b.pnl,
    },
    {
      title: '持仓',
      dataIndex: 'holding_days',
      width: 60,
      render: (v: number) => <Text style={{ fontFamily: 'monospace' }}>{v}d</Text>,
    },
    {
      title: '卖出原因',
      dataIndex: 'exit_signal',
      width: 140,
      render: (v: string, record: any) => {
        const isOpen = record.status === 'open' || !record.exit_date
        return (
          <Space size={4}>
            <Tag color={isOpen ? 'gold' : 'default'} style={{ borderRadius: 4, fontSize: 11, margin: 0 }}>
              {isOpen ? '持仓中' : '已平仓'}
            </Tag>
            {!isOpen && <Text style={{ fontSize: 11, color: '#94a3b8' }}>{v || '-'}</Text>}
          </Space>
        )
      },
    },
  ]

  // 从 metrics 提取展示数据
  const m = result?.metrics

  return (
    <div style={{ maxWidth: 1400 }}>
      {/* 顶部 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>策略回测</Title>
          <Text style={{ color: '#64748b', fontSize: 13 }}>
            配置回测参数，运行历史模拟，分析策略表现
          </Text>
        </div>
      </div>

      {/* 回测参数 */}
      <Card
        size="small"
        style={{ background: '#0f1729', borderColor: '#1e2d45', marginBottom: 16 }}
        styles={{ header: { borderBottomColor: '#1e2d45' } }}
      >
        <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>回测区间</Text>
            <RangePicker
              size="small"
              style={{ width: 260 }}
              value={dateRange}
              onChange={(dates) => setDateRange(dates as [any, any])}
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>初始资金</Text>
            <InputNumber
              size="small"
              value={initialCapital}
              onChange={(v) => setInitialCapital(v || 100000)}
              min={10000}
              step={10000}
              style={{ width: 130 }}
              formatter={v => `${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>参与分组</Text>
            <Select
              size="small"
              mode="multiple"
              placeholder="全部分组"
              value={selectedGroups}
              onChange={setSelectedGroups}
              style={{ minWidth: 200 }}
              options={groups.map(g => ({ value: g.name, label: g.name }))}
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>策略模式</Text>
            <Select
              size="small"
              value={selectedMode}
              onChange={setSelectedMode}
              style={{ width: 120 }}
              options={[
                { value: 'base', label: '基础' },
                { value: 'trending', label: '趋势上涨' },
                { value: 'ranging', label: '震荡' },
              ]}
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
            {running ? '回测中...' : '开始回测'}
          </Button>
        </div>

        {running && (
          <div style={{ marginTop: 12 }}>
            <Progress
              percent={progress}
              size="small"
              strokeColor="#3b82f6"
              format={() => progressText}
            />
          </div>
        )}
      </Card>

      {/* 结果区域 */}
      {result && m && (
        <>
          {/* 回测模式标识 */}
          <div style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Text style={{ color: '#94a3b8', fontSize: 13 }}>回测模式：</Text>
            <Tag color={MODE_COLORS[selectedMode]} style={{ fontSize: 13, padding: '0 8px' }}>
              {MODE_LABELS[selectedMode]}
            </Tag>
            {selectedGroups.length > 0 && (
              <>
                <Text style={{ color: '#94a3b8', fontSize: 13 }}>参与分组：</Text>
                {selectedGroups.map(g => (
                  <Tag key={g} color="#1e2d45" style={{ color: '#93c5fd', fontSize: 12 }}>{g}</Tag>
                ))}
              </>
            )}
            <Text style={{ color: '#64748b', fontSize: 12, marginLeft: 'auto' }}>
              初始资金: ¥{m.initial_capital?.toLocaleString()} → 最终: ¥{m.final_value?.toLocaleString()}
            </Text>
          </div>

          {/* 绩效概览 */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>总收益率</Text>}
                  value={m.total_return * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: m.total_return >= 0 ? '#22c55e' : '#ef4444', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={m.total_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>年化收益</Text>}
                  value={m.annual_return * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: m.annual_return >= 0 ? '#22c55e' : '#ef4444', fontFamily: 'monospace', fontSize: 20 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>最大回撤</Text>}
                  value={m.max_drawdown * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: '#ef4444', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<WarningOutlined />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>夏普比率</Text>}
                  value={m.sharpe_ratio}
                  precision={2}
                  valueStyle={{ color: '#3b82f6', fontFamily: 'monospace', fontSize: 20 }}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>胜率</Text>}
                  value={m.win_rate * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#f59e0b', fontFamily: 'monospace', fontSize: 20 }}
                  prefix={<TrophyOutlined />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card size="small" style={{ background: '#0f1729', borderColor: '#1e2d45' }}>
                <Statistic
                  title={<Text style={{ color: '#94a3b8', fontSize: 12 }}>盈亏比</Text>}
                  value={m.profit_factor}
                  precision={2}
                  valueStyle={{ color: '#8b5cf6', fontFamily: 'monospace', fontSize: 20 }}
                />
              </Card>
            </Col>
          </Row>

          {/* 净值曲线 */}
          <Card
            size="small"
            title={
              <Text style={{ color: '#e2e8f0' }}>
                净值曲线
                <Text style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                  {MODE_LABELS[selectedMode]} · {selectedGroups.length > 0 ? selectedGroups.join('、') : '全部分组'}
                </Text>
              </Text>
            }
            style={{ background: '#0f1729', borderColor: '#1e2d45', marginBottom: 16 }}
            styles={{ header: { borderBottomColor: '#1e2d45' } }}
          >
            <ReactECharts
              option={equityOption}
              style={{ height: 300 }}
              opts={{ renderer: 'canvas' }}
            />
          </Card>

          {/* 回撤曲线 */}
          <Card
            size="small"
            title={
              <Text style={{ color: '#e2e8f0' }}>
                回撤曲线
                <Text style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                  {MODE_LABELS[selectedMode]} · {selectedGroups.length > 0 ? selectedGroups.join('、') : '全部分组'}
                </Text>
              </Text>
            }
            style={{ background: '#0f1729', borderColor: '#1e2d45', marginBottom: 16 }}
            styles={{ header: { borderBottomColor: '#1e2d45' } }}
          >
            <ReactECharts
              option={drawdownOption}
              style={{ height: 180 }}
              opts={{ renderer: 'canvas' }}
            />
          </Card>

          {/* 交易明细 */}
          <Card
            size="small"
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                <Text style={{ color: '#e2e8f0' }}>交易明细 ({filteredTrades.length}笔)</Text>
                <Button size="small" icon={<DownloadOutlined />} type="text">导出CSV</Button>
              </div>
            }
            style={{ background: '#0f1729', borderColor: '#1e2d45' }}
            styles={{ header: { borderBottomColor: '#1e2d45' } }}
          >
            <Table
              dataSource={filteredTrades}
              columns={tradeColumns}
              rowKey={(record) => `${record.symbol}-${record.entry_date}-${record.shares}`}
              size="small"
              pagination={{ pageSize: 10, size: 'small' }}
              style={{ background: 'transparent' }}
              scroll={{ x: 1200 }}
            />
          </Card>
        </>
      )}
    </div>
  )
}
