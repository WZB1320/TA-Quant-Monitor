import { useEffect, useMemo, useState } from 'react'
import {
  Card, Button, Table, Tag, Space, Typography, Row, Col, Progress,
  Select, Empty, Spin, Tooltip, Radio,
} from 'antd'
import {
  ThunderboltOutlined, ReloadOutlined, ArrowUpOutlined,
  ArrowDownOutlined, MinusOutlined, StarOutlined,
} from '@ant-design/icons'
import { useSignalsStore } from '../../stores/signals'
import type { SignalResult } from '../../stores/signals'

const { Title, Text } = Typography

// 信号级别颜色映射
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

const marketLabel = (m: string) => {
  if (m === 'sh') return '沪'
  if (m === 'sz') return '深'
  if (m === 'bj') return '京'
  return m
}

const marketColor = (m: string) => {
  if (m === 'sh') return '#ef4444'
  if (m === 'sz') return '#3b82f6'
  if (m === 'bj') return '#f59e0b'
  return '#64748b'
}

export default function SignalsPage() {
  const {
    results, summary, analyzedAt, analyzedGroup, userRegime, running, progress, total, hasResult,
    groups, runAnalysis, fetchResult, fetchGroups,
  } = useSignalsStore()

  const [filter, setFilter] = useState<'all' | 'bullish' | 'neutral' | 'bearish'>('all')
  const [groupFilter, setGroupFilter] = useState<string>('')
  const [selectedGroup, setSelectedGroup] = useState<string>('')
  const [selectedRegime, setSelectedRegime] = useState<string>('auto')

  // 首次加载时尝试获取已有结果和分组列表
  useEffect(() => {
    fetchResult()
    fetchGroups()
  }, [])

  // 获取所有分组
  const allGroups = useMemo(() => {
    const groups = new Set<string>()
    results.forEach(r => { if (r.group) groups.add(r.group) })
    return Array.from(groups)
  }, [results])

  // 筛选后的结果
  const filteredResults = useMemo(() => {
    let list = results
    if (filter === 'bullish') list = list.filter(r => ['STRONG_BUY', 'BUY', 'WEAK_BUY'].includes(r.level))
    if (filter === 'neutral') list = list.filter(r => r.level === 'NEUTRAL')
    if (filter === 'bearish') list = list.filter(r => ['WEAK_SELL', 'SELL', 'STRONG_SELL'].includes(r.level))
    if (groupFilter) list = list.filter(r => r.group === groupFilter)
    return list
  }, [results, filter, groupFilter])

  const handleRun = async () => {
    await runAnalysis(selectedGroup || undefined, selectedRegime)
  }

  // 切换分组时重置体制选择
  const handleGroupChange = (v: string) => {
    setSelectedGroup(v || '')
    setSelectedRegime('auto')
  }

  // 表格列
  const columns = [
    {
      title: '代码',
      dataIndex: 'code',
      width: 100,
      render: (code: string, r: SignalResult) => (
        <Space size={4}>
          <Text style={{ color: '#93c5fd', fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
            {code}
          </Text>
          <Tag color={marketColor(r.market)} style={{ borderRadius: 3, fontSize: 10, lineHeight: '16px' }}>
            {marketLabel(r.market)}
          </Tag>
        </Space>
      ),
    },
    {
      title: '名称',
      dataIndex: 'name',
      width: 110,
      render: (name: string) => <Text style={{ color: '#e2e8f0' }}>{name}</Text>,
    },
    {
      title: '收盘价',
      dataIndex: 'close',
      width: 90,
      align: 'right' as const,
      render: (v: number) => (
        <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>
          {v.toFixed(2)}
        </Text>
      ),
    },
    {
      title: '信号',
      dataIndex: 'level',
      width: 100,
      align: 'center' as const,
      render: (level: string, r: SignalResult) => (
        <Tag
          style={{
            background: levelBgMap[level] || '#1a1a2e',
            color: levelColorMap[level] || '#94a3b8',
            border: `1px solid ${levelColorMap[level] || '#94a3b8'}40`,
            borderRadius: 4,
            fontWeight: 600,
            fontSize: 12,
          }}
        >
          {r.label}
        </Tag>
      ),
    },
    {
      title: '得分',
      dataIndex: 'score',
      width: 80,
      align: 'right' as const,
      sorter: (a: SignalResult, b: SignalResult) => a.score - b.score,
      render: (score: number) => (
        <Text style={{
          color: score > 0 ? '#4ade80' : score < 0 ? '#f87171' : '#94a3b8',
          fontFamily: "'JetBrains Mono', monospace",
          fontWeight: 600,
        }}>
          {score > 0 ? '+' : ''}{score}
        </Text>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      width: 80,
      align: 'right' as const,
      render: (v: number) => (
        <Text style={{ color: '#94a3b8', fontFamily: "'JetBrains Mono', monospace" }}>
          {(v * 100).toFixed(0)}%
        </Text>
      ),
    },
    {
      title: '执行状态',
      dataIndex: 'execution',
      width: 110,
      align: 'center' as const,
      render: (_: any, r: SignalResult) => {
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
      title: '判断原因',
      dataIndex: 'block_detail',
      width: 280,
      render: (_: string, r: SignalResult) => {
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
          // 有明确信号级别时, 展示级别 + 降级轨迹
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

        // Tooltip 展示完整分析报告 (details 字段)
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

  // 展开行
  const expandedRowRender = (r: SignalResult) => {
    const ind = r.indicators
    return (
      <div style={{ padding: '8px 16px', background: '#0a1628', borderRadius: 6 }}>
        <Row gutter={[24, 8]}>
          <Col span={24}>
            <Text style={{ color: '#e2e8f0', fontSize: 13 }}>{r.details}</Text>
          </Col>
          {ind.rsi != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>RSI</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{ind.rsi}</Text></Col>
          )}
          {ind.dif != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>DIF/DEA</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>
                {ind.dif} / {ind.dea}
              </Text></Col>
          )}
          {ind.ma60 != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>MA60</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>
                {ind.ma60}
                <Tag color={ind.ma60_dir === '多头' ? 'green' : 'red'} style={{ marginLeft: 6, borderRadius: 3, fontSize: 10 }}>
                  {ind.ma60_dir}
                </Tag>
              </Text></Col>
          )}
          {ind.adx != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>ADX</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{ind.adx}</Text></Col>
          )}
          {ind.volume_ratio != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>量比</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{ind.volume_ratio}</Text></Col>
          )}
          {ind.atr_pct != null && (
            <Col><Text style={{ color: '#64748b', fontSize: 12 }}>ATR%</Text><br />
              <Text style={{ color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{ind.atr_pct}%</Text></Col>
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

  return (
    <div style={{ padding: 0 }}>
      {/* 顶部操作栏 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Title level={4} style={{ color: '#e2e8f0', margin: 0 }}>信号看板</Title>
          {analyzedAt && (
            <Text style={{ color: '#64748b', fontSize: 12 }}>
              最近分析: {analyzedAt}
              {analyzedGroup && <Tag color="blue" style={{ marginLeft: 6, borderRadius: 3, fontSize: 10 }}>{analyzedGroup}</Tag>}
              {userRegime !== 'auto' && (
                <Tag color={userRegime === 'trending' ? 'green' : 'orange'} style={{ marginLeft: 4, borderRadius: 3, fontSize: 10 }}>
                  {{ trending: '趋势上涨', ranging: '震荡' }[userRegime]}
                </Tag>
              )}
            </Text>
          )}
        </div>
        <Space size={12}>
          <Select
            placeholder="分析范围: 全部自选股"
            allowClear
            style={{ width: 200 }}
            value={selectedGroup || undefined}
            onChange={handleGroupChange}
            options={groups.map(g => ({ label: `${g.name} (${g.count}只)`, value: g.name }))}
            disabled={running}
          />
          <Radio.Group
            value={selectedRegime}
            onChange={e => setSelectedRegime(e.target.value)}
            disabled={running}
            size="small"
            optionType="button"
            buttonStyle="solid"
            style={{ whiteSpace: 'nowrap' }}
          >
            <Radio.Button value="auto">自动判断</Radio.Button>
            <Radio.Button value="trending" style={{ color: selectedRegime === 'trending' ? '#4ade80' : undefined }}>
              趋势上涨
            </Radio.Button>
            <Radio.Button value="ranging" style={{ color: selectedRegime === 'ranging' ? '#f97316' : undefined }}>
              震荡
            </Radio.Button>
          </Radio.Group>
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={handleRun}
            loading={running}
            size="large"
            style={{
              background: running ? undefined
                : selectedRegime === 'trending' ? '#16a34a'
                : selectedRegime === 'ranging' ? '#ea580c'
                : '#3b82f6',
            }}
          >
            {running ? '分析中...' : '运行分析'}
          </Button>
          {!running && hasResult && (
            <Button icon={<ReloadOutlined />} onClick={fetchResult}>刷新</Button>
          )}
        </Space>
      </div>

      {/* 进度条 */}
      {running && (
        <Card size="small" style={{ marginBottom: 16, background: '#0f1729', border: '1px solid #1e2d45' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Progress
              percent={total > 0 ? Math.round(progress / total * 100) : 0}
              size="small"
              style={{ flex: 1 }}
              strokeColor="#3b82f6"
            />
            <Text style={{ color: '#94a3b8', fontSize: 13, fontFamily: "'JetBrains Mono', monospace" }}>
              {progress}/{total}
            </Text>
          </div>
        </Card>
      )}

      {/* 统计卡片 */}
      {hasResult && (
        <Row gutter={16} style={{ marginBottom: 20 }}>
          <Col span={6}>
            <Card
              size="small"
              style={{ background: '#052e16', border: '1px solid #166534', cursor: 'pointer' }}
              onClick={() => setFilter(filter === 'bullish' ? 'all' : 'bullish')}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ArrowUpOutlined style={{ color: '#4ade80', fontSize: 18 }} />
                <div>
                  <Text style={{ color: '#4ade80', fontSize: 24, fontWeight: 700 }}>{summary.bullish}</Text>
                  <br />
                  <Text style={{ color: '#86efac', fontSize: 12 }}>看多</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card
              size="small"
              style={{ background: '#1a1a2e', border: '1px solid #334155', cursor: 'pointer' }}
              onClick={() => setFilter(filter === 'neutral' ? 'all' : 'neutral')}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <MinusOutlined style={{ color: '#94a3b8', fontSize: 18 }} />
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 24, fontWeight: 700 }}>{summary.neutral}</Text>
                  <br />
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>观望</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card
              size="small"
              style={{ background: '#2d0f0f', border: '1px solid #7f1d1d', cursor: 'pointer' }}
              onClick={() => setFilter(filter === 'bearish' ? 'all' : 'bearish')}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ArrowDownOutlined style={{ color: '#f87171', fontSize: 18 }} />
                <div>
                  <Text style={{ color: '#f87171', fontSize: 24, fontWeight: 700 }}>{summary.bearish}</Text>
                  <br />
                  <Text style={{ color: '#fca5a5', fontSize: 12 }}>看空</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ background: '#2d1b0e', border: '1px solid #92400e' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <StarOutlined style={{ color: '#fbbf24', fontSize: 18 }} />
                <div>
                  <Text style={{ color: '#fbbf24', fontSize: 24, fontWeight: 700 }}>{summary.actionable}</Text>
                  <br />
                  <Text style={{ color: '#fcd34d', fontSize: 12 }}>可操作</Text>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* 筛选栏 */}
      {hasResult && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 16, alignItems: 'center' }}>
          <Space>
            {(['all', 'bullish', 'neutral', 'bearish'] as const).map(f => (
              <Button
                key={f}
                size="small"
                type={filter === f ? 'primary' : 'default'}
                onClick={() => setFilter(f)}
                style={filter === f ? { background: '#3b82f6' } : undefined}
              >
                {{ all: '全部', bullish: '看多', neutral: '观望', bearish: '看空' }[f]}
              </Button>
            ))}
          </Space>
          {allGroups.length > 0 && (
            <Select
              placeholder="按分组筛选"
              allowClear
              style={{ width: 160 }}
              value={groupFilter || undefined}
              onChange={v => setGroupFilter(v || '')}
              options={allGroups.map(g => ({ label: g, value: g }))}
            />
          )}
          <Text style={{ color: '#64748b', fontSize: 12, marginLeft: 'auto' }}>
            共 {filteredResults.length} 条
          </Text>
        </div>
      )}

      {/* 主表格 */}
      {!hasResult && !running ? (
        <Card style={{ background: '#0f1729', border: '1px solid #1e2d45', textAlign: 'center', padding: 60 }}>
          <Empty
            description={<Text style={{ color: '#64748b' }}>暂无分析结果，请点击「运行分析」</Text>}
          />
        </Card>
      ) : (
        <Card style={{ background: '#0f1729', border: '1px solid #1e2d45' }}>
          <Table
            dataSource={filteredResults}
            columns={columns}
            rowKey="code"
            size="small"
            pagination={false}
            expandable={{ expandedRowRender }}
            rowClassName={(r) => {
              if (r.level === 'STRONG_BUY' || r.level === 'BUY') return 'row-bullish'
              if (r.level === 'STRONG_SELL' || r.level === 'SELL') return 'row-bearish'
              return ''
            }}
            style={{ background: 'transparent' }}
          />
        </Card>
      )}
    </div>
  )
}
