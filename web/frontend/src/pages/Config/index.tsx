import { useEffect, useState } from 'react'
import {
  Card, Select, InputNumber, Slider, Button, Typography, Space, App,
  Collapse, Switch, Divider, Spin, Radio, Tag,
} from 'antd'
import {
  SaveOutlined, UndoOutlined, SettingOutlined,
  ThunderboltOutlined, LineChartOutlined,
  BarChartOutlined, DotChartOutlined, GlobalOutlined,
  RiseOutlined, FallOutlined,
} from '@ant-design/icons'
import { configApi } from '../../api'
import { useWatchlistStore } from '../../stores/watchlist'

const { Title, Text } = Typography

type ModeType = 'base' | 'trending' | 'ranging'

const MODE_OPTIONS: { value: ModeType; label: string; icon: React.ReactNode }[] = [
  { value: 'base', label: '基础', icon: <SettingOutlined /> },
  { value: 'trending', label: '趋势上涨', icon: <RiseOutlined /> },
  { value: 'ranging', label: '震荡', icon: <FallOutlined /> },
]

// 指标分类定义
const INDICATOR_CATEGORIES = [
  {
    key: 'trend',
    label: '趋势指标',
    icon: <LineChartOutlined />,
    indicators: [
      {
        name: 'MA60', label: 'MA60 均线',
        params: [
          { key: 'period', label: '周期', type: 'number', default: 60, min: 10, max: 200 },
          { key: 'direction_buffer', label: '方向缓冲', type: 'percent', default: 0.015, min: 0, max: 0.1, step: 0.001 },
        ],
      },
      {
        name: 'EMA_DUAL', label: 'EMA 双线',
        params: [
          { key: 'fast', label: '快线', type: 'number', default: 12, min: 2, max: 50 },
          { key: 'slow', label: '慢线', type: 'number', default: 26, min: 5, max: 100 },
        ],
      },
      {
        name: 'MACD', label: 'MACD',
        params: [
          { key: 'fast', label: '快线', type: 'number', default: 12, min: 2, max: 50 },
          { key: 'slow', label: '慢线', type: 'number', default: 26, min: 5, max: 100 },
          { key: 'signal', label: '信号线', type: 'number', default: 9, min: 2, max: 30 },
          { key: 'mode', label: '模式', type: 'select', default: 'standard', options: [
            { value: 'standard', label: '标准' },
            { value: 'divergence_zero', label: '背离+零轴' },
          ]},
        ],
      },
    ],
  },
  {
    key: 'strength',
    label: '强度指标',
    icon: <BarChartOutlined />,
    indicators: [
      {
        name: 'ADX', label: 'ADX 趋势强度',
        params: [
          { key: 'period', label: '周期', type: 'number', default: 14, min: 5, max: 50 },
          { key: 'weak', label: '弱趋势阈值', type: 'number', default: 20, min: 10, max: 30 },
          { key: 'trending', label: '趋势阈值', type: 'number', default: 25, min: 20, max: 40 },
          { key: 'strong', label: '强趋势阈值', type: 'number', default: 45, min: 30, max: 60 },
        ],
      },
      {
        name: 'ATR', label: 'ATR 波动率',
        params: [
          { key: 'period', label: '周期', type: 'number', default: 14, min: 5, max: 50 },
        ],
      },
    ],
  },
  {
    key: 'momentum',
    label: '动量指标',
    icon: <ThunderboltOutlined />,
    indicators: [
      {
        name: 'RSI', label: 'RSI 相对强弱',
        params: [
          { key: 'period', label: '周期', type: 'number', default: 21, min: 5, max: 50 },
          { key: 'oversold', label: '超卖', type: 'number', default: 30, min: 10, max: 40 },
          { key: 'overbought', label: '超买', type: 'number', default: 70, min: 60, max: 90 },
          { key: 'neutral_zone', label: '中性区', type: 'range', default: [40, 60], min: 20, max: 80 },
        ],
      },
      {
        name: 'KDJ', label: 'KDJ 随机指标',
        params: [
          { key: 'k_period', label: 'K周期', type: 'number', default: 9, min: 3, max: 30 },
          { key: 'k_smooth', label: 'K平滑', type: 'number', default: 3, min: 1, max: 10 },
          { key: 'd_smooth', label: 'D平滑', type: 'number', default: 3, min: 1, max: 10 },
          { key: 'oversold_j', label: 'J超卖', type: 'number', default: 20, min: 0, max: 40 },
          { key: 'overbought_j', label: 'J超买', type: 'number', default: 80, min: 60, max: 100 },
        ],
      },
    ],
  },
  {
    key: 'volume',
    label: '量价指标',
    icon: <DotChartOutlined />,
    indicators: [
      {
        name: 'OBV', label: 'OBV 能量潮',
        params: [
          { key: 'lookback', label: '回看天数', type: 'number', default: 10, min: 5, max: 50 },
        ],
      },
      {
        name: 'VOL_RATIO', label: '量比',
        params: [
          { key: 'ma_period', label: '均量周期', type: 'number', default: 5, min: 2, max: 20 },
          { key: 'effective_threshold', label: '有效放量', type: 'number', default: 1.2, min: 0.5, max: 3, step: 0.1 },
          { key: 'strong_threshold', label: '强放量', type: 'number', default: 2.0, min: 1, max: 5, step: 0.1 },
        ],
      },
    ],
  },
]

// 体制权重模板
const REGIME_TYPES = ['trending', 'transition', 'ranging', 'trend_fading']
const REGIME_LABELS: Record<string, string> = {
  trending: '趋势市', transition: '过渡期', ranging: '震荡市', trend_fading: '趋势衰退',
}
const CATEGORY_KEYS = ['trend', 'strength', 'momentum', 'volume']
const CATEGORY_LABELS: Record<string, string> = {
  trend: '趋势', strength: '强度', momentum: '动量', volume: '量价',
}

// 默认分组配置
const DEFAULT_GROUP_CONFIG: Record<string, any> = {
  score_threshold: 25,
  score_ceiling: 0,
  cooldown_days: 4,
  consecutive_loss_suspend: 0,
  max_consecutive_losses: 0,
  vol_ratio_threshold: 0.6,
  atr_price_ratio_max: 0,
  require_macd_dif_above_zero: false,
  price_ma20_max_deviation: 0,
  rsi_overbought: 0,
  atr_stop_mult: 2.5,
  max_per_stock_boost: 1.0,
}

const DEFAULT_INDICATOR_WEIGHTS: Record<string, number> = {
  MA60: 0.15, EMA_DUAL: 0.12, MACD: 0.10,
  RSI: 0.18, KDJ: 0.10, ADX: 0.15,
  OBV: 0.12, VOL_RATIO: 0.08,
}

const DEFAULT_REGIME_WEIGHTS: Record<string, Record<string, number>> = {
  trending: { trend: 0.40, strength: 0.25, momentum: 0.15, volume: 0.20 },
  ranging: { trend: 0.15, strength: 0.10, momentum: 0.50, volume: 0.25 },
  transition: { trend: 0.28, strength: 0.18, momentum: 0.32, volume: 0.22 },
}

// 缓存数据结构
interface CachedConfig {
  config: Record<string, any>
  indicatorParams: Record<string, any>
  indicatorWeights: Record<string, number>
  regimeWeights: Record<string, Record<string, number>>
}

function cacheKey(group: string, mode: ModeType): string {
  return `${group}:${mode}`
}

export default function ConfigPage() {
  const { message } = App.useApp()
  const { groups: watchlistGroups, fetchWatchlist } = useWatchlistStore()
  const [selectedGroup, setSelectedGroup] = useState<string>('_default')
  const [selectedMode, setSelectedMode] = useState<ModeType>('base')
  const [config, setConfig] = useState<Record<string, any>>({ ...DEFAULT_GROUP_CONFIG })
  const [indicatorParams, setIndicatorParams] = useState<Record<string, any>>({})
  const [indicatorWeights, setIndicatorWeights] = useState<Record<string, number>>({ ...DEFAULT_INDICATOR_WEIGHTS })
  const [regimeWeights, setRegimeWeights] = useState<Record<string, Record<string, number>>>({ ...DEFAULT_REGIME_WEIGHTS })
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(false)
  const [groupConfigs, setGroupConfigs] = useState<Record<string, CachedConfig>>({})

  useEffect(() => {
    fetchWatchlist()
    loadGroupConfig('_default', 'base')
  }, [])

  // ── 加载分组配置 ──

  const loadGroupConfig = async (group: string, mode: ModeType) => {
    setLoading(true)
    try {
      const { data } = await configApi.getGroupConfig(group, mode)
      // data = { group_name, config, presets }
      applyGroupConfig(data.config)
    } catch {
      // 后端不可用时回退到默认值
      applyGroupConfig(DEFAULT_GROUP_CONFIG)
    }
    setLoading(false)
  }

  const applyGroupConfig = (gc: any) => {
    setConfig({
      score_threshold: gc.score_threshold ?? DEFAULT_GROUP_CONFIG.score_threshold,
      score_ceiling: gc.score_ceiling ?? DEFAULT_GROUP_CONFIG.score_ceiling,
      cooldown_days: gc.cooldown_days ?? DEFAULT_GROUP_CONFIG.cooldown_days,
      consecutive_loss_suspend: gc.consecutive_loss_suspend ?? DEFAULT_GROUP_CONFIG.consecutive_loss_suspend,
      max_consecutive_losses: gc.max_consecutive_losses ?? DEFAULT_GROUP_CONFIG.max_consecutive_losses,
      vol_ratio_threshold: gc.vol_ratio_threshold ?? DEFAULT_GROUP_CONFIG.vol_ratio_threshold,
      atr_price_ratio_max: gc.atr_price_ratio_max ?? DEFAULT_GROUP_CONFIG.atr_price_ratio_max,
      require_macd_dif_above_zero: gc.require_macd_dif_above_zero ?? DEFAULT_GROUP_CONFIG.require_macd_dif_above_zero,
      price_ma20_max_deviation: gc.price_ma20_max_deviation ?? DEFAULT_GROUP_CONFIG.price_ma20_max_deviation,
      rsi_overbought: gc.rsi_overbought ?? DEFAULT_GROUP_CONFIG.rsi_overbought,
      atr_stop_mult: gc.atr_stop_mult ?? DEFAULT_GROUP_CONFIG.atr_stop_mult,
      max_per_stock_boost: gc.max_per_stock_boost ?? DEFAULT_GROUP_CONFIG.max_per_stock_boost,
    })

    setIndicatorParams(gc.indicator_params || {})
    setIndicatorWeights(gc.indicator_weights || { ...DEFAULT_INDICATOR_WEIGHTS })
    setRegimeWeights(gc.regime_weights || { ...DEFAULT_REGIME_WEIGHTS })
  }

  // ── 切换分组 / 模式 ──

  const cacheCurrent = () => {
    setGroupConfigs(prev => ({
      ...prev,
      [cacheKey(selectedGroup, selectedMode)]: {
        config, indicatorParams, indicatorWeights, regimeWeights,
      },
    }))
  }

  const handleGroupChange = (value: string) => {
    cacheCurrent()
    setSelectedGroup(value)

    const key = cacheKey(value, selectedMode)
    const cached = groupConfigs[key]
    if (cached) {
      setConfig(cached.config)
      setIndicatorParams(cached.indicatorParams)
      setIndicatorWeights(cached.indicatorWeights)
      setRegimeWeights(cached.regimeWeights)
    } else {
      loadGroupConfig(value, selectedMode)
    }
  }

  const handleModeChange = (value: ModeType) => {
    cacheCurrent()
    setSelectedMode(value)

    const key = cacheKey(selectedGroup, value)
    const cached = groupConfigs[key]
    if (cached) {
      setConfig(cached.config)
      setIndicatorParams(cached.indicatorParams)
      setIndicatorWeights(cached.indicatorWeights)
      setRegimeWeights(cached.regimeWeights)
    } else {
      loadGroupConfig(selectedGroup, value)
    }
  }

  // ── 保存 ──

  const handleSave = async () => {
    setSaving(true)
    try {
      const payload = {
        ...config,
        indicator_params: indicatorParams,
        indicator_weights: indicatorWeights,
        regime_weights: regimeWeights,
      }

      if (selectedMode === 'base') {
        // 基础模式：保存完整配置
        await configApi.saveGroupConfig(selectedGroup, payload)
      } else {
        // 趋势上涨 / 震荡模式：仅保存 preset 相关字段
        await configApi.saveGroupPreset(selectedGroup, selectedMode, payload)
      }

      // 保存后更新缓存
      setGroupConfigs(prev => ({
        ...prev,
        [cacheKey(selectedGroup, selectedMode)]: {
          config, indicatorParams, indicatorWeights, regimeWeights,
        },
      }))

      message.success(
        selectedMode === 'base'
          ? '基础配置已保存'
          : `「${MODE_OPTIONS.find(m => m.value === selectedMode)?.label}」预设已保存`,
      )
    } catch {
      message.error('保存失败, 请检查后端服务')
    }
    setSaving(false)
  }

  const handleReset = () => {
    setConfig({ ...DEFAULT_GROUP_CONFIG })
    setIndicatorParams({})
    setIndicatorWeights({ ...DEFAULT_INDICATOR_WEIGHTS })
    setRegimeWeights({ ...DEFAULT_REGIME_WEIGHTS })
    message.info('已恢复默认值（未保存，点「保存」后生效）')
  }

  // ── 表单更新 ──

  const updateIndicatorParam = (indicatorName: string, key: string, value: any) => {
    setIndicatorParams(prev => ({
      ...prev,
      [indicatorName]: {
        ...(prev[indicatorName] || {}),
        [key]: value,
      },
    }))
  }

  const updateIndicatorWeight = (name: string, value: number) => {
    setIndicatorWeights(prev => ({ ...prev, [name]: value }))
  }

  const updateRegimeWeight = (regime: string, category: string, value: number) => {
    setRegimeWeights(prev => ({
      ...prev,
      [regime]: {
        ...(prev[regime] || {}),
        [category]: value,
      },
    }))
  }

  const weightSum = Object.values(indicatorWeights).reduce((a, b) => a + (b || 0), 0)

  // 判断当前模式是否仅保存 preset（用于提示）
  const isPresetMode = selectedMode !== 'base'

  // ── 渲染参数输入 ──

  const renderParamInput = (indicatorName: string, param: any) => {
    const currentValue = indicatorParams[indicatorName]?.[param.key] ?? param.default

    if (param.type === 'number') {
      return (
        <div key={param.key} style={{ marginBottom: 8 }}>
          <Text style={{ color: '#94a3b8', fontSize: 12 }}>{param.label}</Text>
          <InputNumber
            size="small"
            value={currentValue}
            min={param.min}
            max={param.max}
            step={param.step || 1}
            onChange={v => updateIndicatorParam(indicatorName, param.key, v)}
            style={{ width: '100%', marginTop: 2 }}
          />
        </div>
      )
    }

    if (param.type === 'percent') {
      return (
        <div key={param.key} style={{ marginBottom: 8 }}>
          <Text style={{ color: '#94a3b8', fontSize: 12 }}>{param.label}</Text>
          <InputNumber
            size="small"
            value={currentValue}
            min={param.min}
            max={param.max}
            step={param.step || 0.001}
            onChange={v => updateIndicatorParam(indicatorName, param.key, v)}
            style={{ width: '100%', marginTop: 2 }}
            formatter={v => `${(Number(v) * 100).toFixed(1)}%`}
            parser={v => Number(v?.replace('%', '')) / 100}
          />
        </div>
      )
    }

    if (param.type === 'select') {
      return (
        <div key={param.key} style={{ marginBottom: 8 }}>
          <Text style={{ color: '#94a3b8', fontSize: 12 }}>{param.label}</Text>
          <Select
            size="small"
            value={currentValue}
            onChange={v => updateIndicatorParam(indicatorName, param.key, v)}
            options={param.options}
            style={{ width: '100%', marginTop: 2 }}
          />
        </div>
      )
    }

    if (param.type === 'range') {
      return (
        <div key={param.key} style={{ marginBottom: 8 }}>
          <Text style={{ color: '#94a3b8', fontSize: 12 }}>{param.label}</Text>
          <div style={{ display: 'flex', gap: 8, marginTop: 2 }}>
            <InputNumber
              size="small"
              value={currentValue?.[0] ?? param.default[0]}
              min={param.min}
              max={param.max}
              onChange={v => updateIndicatorParam(indicatorName, param.key, [v, currentValue?.[1] ?? param.default[1]])}
              style={{ width: '50%' }}
            />
            <InputNumber
              size="small"
              value={currentValue?.[1] ?? param.default[1]}
              min={param.min}
              max={param.max}
              onChange={v => updateIndicatorParam(indicatorName, param.key, [currentValue?.[0] ?? param.default[0], v])}
              style={{ width: '50%' }}
            />
          </div>
        </div>
      )
    }

    return null
  }

  return (
    <div style={{ maxWidth: 1200 }}>
      {/* 顶部 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>策略参数配置</Title>
          <Text style={{ color: '#64748b', fontSize: 13 }}>
            按分组自定义指标参数、权重和体制配置
            {isPresetMode && (
              <Tag color="orange" style={{ marginLeft: 8, fontSize: 11 }}>
                {MODE_OPTIONS.find(m => m.value === selectedMode)?.label}预设
              </Tag>
            )}
          </Text>
        </div>
        <Space>
          <Text style={{ color: '#94a3b8', fontSize: 12, marginRight: -4 }}>分组</Text>
          <Select
            value={selectedGroup}
            onChange={handleGroupChange}
            style={{ width: 160 }}
            suffixIcon={<GlobalOutlined />}
            options={[
              { value: '_default', label: '全局默认' },
              ...watchlistGroups.map(g => ({ value: g.name, label: g.name })),
            ]}
          />

          <Text style={{ color: '#94a3b8', fontSize: 12, marginLeft: 16, marginRight: -4 }}>模式</Text>
          <Radio.Group
            value={selectedMode}
            onChange={e => handleModeChange(e.target.value)}
            optionType="button"
            buttonStyle="solid"
            size="small"
          >
            {MODE_OPTIONS.map(m => (
              <Radio.Button key={m.value} value={m.value}>
                {m.icon} {m.label}
              </Radio.Button>
            ))}
          </Radio.Group>

          <Button icon={<UndoOutlined />} onClick={handleReset} style={{ marginLeft: 16 }}>恢复默认</Button>
          <Button type="primary" icon={<SaveOutlined />} loading={saving} onClick={handleSave}>
            保存配置
          </Button>
        </Space>
      </div>

      {/* 模式提示 */}
      {isPresetMode && (
        <div style={{
          background: 'rgba(250, 173, 20, 0.1)', border: '1px solid rgba(250, 173, 20, 0.25)',
          borderRadius: 6, padding: '8px 14px', marginBottom: 16,
        }}>
          <Text style={{ color: '#facc15', fontSize: 12 }}>
            当前为「{MODE_OPTIONS.find(m => m.value === selectedMode)?.label}」预设模式。
            保存时仅写入得分阈值、得分上限、冷却天数、ATR止损倍率、连亏相关和ATR波动率上限 7 个字段。
            切换回「基础」模式可编辑完整配置。
          </Text>
        </div>
      )}

      <Spin spinning={loading}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* 左列 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* 信号引擎参数 */}
            <Card
              title={<span style={{ color: '#e2e8f0' }}><SettingOutlined /> 信号引擎</span>}
              size="small"
              style={{ background: '#0f1729', borderColor: '#1e2d45' }}
              styles={{ header: { borderBottomColor: '#1e2d45' } }}
            >
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>得分阈值</Text>
                  <InputNumber
                    size="small" value={config.score_threshold} min={0} max={100}
                    onChange={v => setConfig(p => ({ ...p, score_threshold: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>得分上限 (0=不限)</Text>
                  <InputNumber
                    size="small" value={config.score_ceiling} min={0} max={100}
                    onChange={v => setConfig(p => ({ ...p, score_ceiling: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>冷却天数</Text>
                  <InputNumber
                    size="small" value={config.cooldown_days} min={0} max={30}
                    onChange={v => setConfig(p => ({ ...p, cooldown_days: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>量比阈值</Text>
                  <InputNumber
                    size="small" value={config.vol_ratio_threshold} min={0} max={2} step={0.1}
                    onChange={v => setConfig(p => ({ ...p, vol_ratio_threshold: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>ATR止损倍率</Text>
                  <InputNumber
                    size="small" value={config.atr_stop_mult} min={1} max={5} step={0.5}
                    onChange={v => setConfig(p => ({ ...p, atr_stop_mult: v || 2.5 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>仓位加成</Text>
                  <InputNumber
                    size="small" value={config.max_per_stock_boost} min={0.5} max={2} step={0.1}
                    onChange={v => setConfig(p => ({ ...p, max_per_stock_boost: v || 1 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>连亏暂停天数</Text>
                  <InputNumber
                    size="small" value={config.consecutive_loss_suspend} min={0} max={30}
                    onChange={v => setConfig(p => ({ ...p, consecutive_loss_suspend: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>连亏触发阈值</Text>
                  <InputNumber
                    size="small" value={config.max_consecutive_losses} min={0} max={10}
                    onChange={v => setConfig(p => ({ ...p, max_consecutive_losses: v || 0 }))}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
              <Divider style={{ borderColor: '#1e2d45', margin: '12px 0' }} />
              <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                <div>
                  <Text style={{ color: '#94a3b8', fontSize: 12 }}>MACD DIF零轴上方</Text>
                  <Switch
                    size="small"
                    checked={config.require_macd_dif_above_zero}
                    onChange={v => setConfig(p => ({ ...p, require_macd_dif_above_zero: v }))}
                  />
                </div>
              </div>
            </Card>

            {/* 指标权重 */}
            <Card
              title={
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                  <span style={{ color: '#e2e8f0' }}><BarChartOutlined /> 指标权重</span>
                  <Text style={{ color: weightSum === 1 ? '#22c55e' : '#ef4444', fontSize: 12, fontFamily: 'monospace' }}>
                    合计: {weightSum.toFixed(2)}
                  </Text>
                </div>
              }
              size="small"
              style={{ background: '#0f1729', borderColor: '#1e2d45' }}
              styles={{ header: { borderBottomColor: '#1e2d45' } }}
            >
              {Object.entries(indicatorWeights).map(([name, weight]) => (
                <div key={name} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <Text style={{ color: '#e2e8f0', fontSize: 12 }}>{name}</Text>
                    <InputNumber
                      size="small"
                      value={weight}
                      min={0}
                      max={1}
                      step={0.01}
                      onChange={v => updateIndicatorWeight(name, v || 0)}
                      style={{ width: 70 }}
                    />
                  </div>
                  <Slider
                    min={0}
                    max={0.5}
                    step={0.01}
                    value={weight}
                    onChange={v => updateIndicatorWeight(name, v)}
                    tooltip={{ formatter: v => `${((v || 0) * 100).toFixed(0)}%` }}
                  />
                </div>
              ))}
            </Card>
          </div>

          {/* 右列 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* 体制权重 */}
            <Card
              title={<span style={{ color: '#e2e8f0' }}><LineChartOutlined /> 体制权重</span>}
              size="small"
              style={{ background: '#0f1729', borderColor: '#1e2d45' }}
              styles={{ header: { borderBottomColor: '#1e2d45' } }}
            >
              <Collapse
                ghost
                items={REGIME_TYPES.map(regime => ({
                  key: regime,
                  label: <Text style={{ color: '#e2e8f0', fontWeight: 500 }}>{REGIME_LABELS[regime]}</Text>,
                  children: (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      {CATEGORY_KEYS.map(cat => (
                        <div key={cat}>
                          <Text style={{ color: '#94a3b8', fontSize: 11 }}>{CATEGORY_LABELS[cat]}</Text>
                          <InputNumber
                            size="small"
                            value={regimeWeights[regime]?.[cat] ?? 0}
                            min={0}
                            max={1}
                            step={0.01}
                            onChange={v => updateRegimeWeight(regime, cat, v || 0)}
                            style={{ width: '100%' }}
                          />
                        </div>
                      ))}
                    </div>
                  ),
                }))}
              />
            </Card>

            {/* 指标参数 */}
            <Card
              title={<span style={{ color: '#e2e8f0' }}><SettingOutlined /> 指标参数</span>}
              size="small"
              style={{ background: '#0f1729', borderColor: '#1e2d45' }}
              styles={{ header: { borderBottomColor: '#1e2d45' } }}
            >
              <Collapse
                ghost
                items={INDICATOR_CATEGORIES.map(cat => ({
                  key: cat.key,
                  label: (
                    <span style={{ color: '#e2e8f0' }}>
                      {cat.icon} {cat.label}
                    </span>
                  ),
                  children: (
                    <Collapse
                      ghost
                      items={cat.indicators.map(ind => ({
                        key: ind.name,
                        label: <Text style={{ color: '#94a3b8', fontSize: 13 }}>{ind.label}</Text>,
                        children: (
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                            {ind.params.map(p => renderParamInput(ind.name, p))}
                          </div>
                        ),
                      }))}
                    />
                  ),
                }))}
              />
            </Card>
          </div>
        </div>
      </Spin>
    </div>
  )
}