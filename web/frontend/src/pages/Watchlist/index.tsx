import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Card, Button, Table, Tag, Space, Modal, Input, Select, App,
  Popconfirm, Typography, Empty, Spin, Tooltip, AutoComplete,
} from 'antd'
import {
  PlusOutlined, DeleteOutlined, SwapOutlined,
  FolderAddOutlined, ReloadOutlined,
} from '@ant-design/icons'
import { useWatchlistStore } from '../../stores/watchlist'
import { stockApi } from '../../api'

const { Title, Text } = Typography

interface StockItem {
  code: string
  name: string
  market: string
}

export default function WatchlistPage() {
  const { message } = App.useApp()
  const { groups, ungrouped, loading, fetchWatchlist, addStock, removeStock, createGroup, deleteGroup } = useWatchlistStore()
  const [addModalOpen, setAddModalOpen] = useState(false)
  const [groupModalOpen, setGroupModalOpen] = useState(false)
  const [searchResults, setSearchResults] = useState<StockItem[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [newGroupDesc, setNewGroupDesc] = useState('')
  const [selectedGroup, setSelectedGroup] = useState('')
  const [moveModalOpen, setMoveModalOpen] = useState(false)
  const [moveStock, setMoveStock] = useState<StockItem | null>(null)
  const [moveTarget, setMoveTarget] = useState('')
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const inputRef = useRef<string>('')  // 用 ref 追踪输入值，避免受控重渲染失焦

  useEffect(() => {
    fetchWatchlist()
    return () => {
      // 组件卸载时清理防抖定时器
      if (debounceTimer.current) clearTimeout(debounceTimer.current)
    }
  }, [])

  // 联想搜索（300ms 防抖，不触发状态更新避免失焦）
  const handleSearch = useCallback((value: string) => {
    inputRef.current = value
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
      debounceTimer.current = null
    }
    if (!value || value.length < 1) {
      setSearchResults([])
      setSearchLoading(false)
      return
    }
    // loading 延迟到防抖后设置，避免每次按键都重渲染导致失焦
    debounceTimer.current = setTimeout(async () => {
      setSearchLoading(true)
      try {
        const { data } = await stockApi.search(value, 10)
        setSearchResults(data.results || [])
      } catch {
        setSearchResults([])
      }
      setSearchLoading(false)
    }, 300)
  }, [])

  // 添加股票
  const handleAdd = async (stock: StockItem, group?: string) => {
    try {
      await addStock(stock.code, stock.name, stock.market, group)
      message.success(`已添加 ${stock.name || stock.code}`)
      setAddModalOpen(false)
      inputRef.current = ''
      setSearchResults([])
    } catch (e: any) {
      const msg = e?.response?.data?.detail || '添加失败'
      message.error(msg)
    }
  }

  // 删除股票
  const handleRemove = async (code: string, name: string) => {
    try {
      await removeStock(code)
      message.success(`已移除 ${name || code}`)
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '删除失败')
    }
  }

  // 新建分组
  const handleCreateGroup = async () => {
    if (!newGroupName.trim()) {
      message.warning('请输入分组名称')
      return
    }
    try {
      await createGroup(newGroupName.trim(), newGroupDesc.trim())
      message.success(`已创建分组「${newGroupName}」`)
      setGroupModalOpen(false)
      setNewGroupName('')
      setNewGroupDesc('')
    } catch (e: any) {
      message.error(e?.response?.data?.detail || '创建失败')
    }
  }

  // 换组
  const handleMove = async () => {
    if (!moveStock || !moveTarget) return
    try {
      await useWatchlistStore.getState().updateStock(moveStock.code, { group: moveTarget })
      message.success(`已将 ${moveStock.name} 移至「${moveTarget || '未分组'}」`)
      setMoveModalOpen(false)
      setMoveStock(null)
    } catch {
      message.error('换组失败')
    }
  }

  // 市场标签颜色
  const marketColor = (m: string) => {
    if (m === 'sh') return '#ef4444'
    if (m === 'sz') return '#3b82f6'
    return '#f59e0b'
  }

  const marketLabel = (m: string) => {
    if (m === 'sh') return '沪'
    if (m === 'sz') return '深'
    if (m === 'bj') return '京'
    return m
  }

  const allGroupNames = groups.map(g => g.name)

  // 联想下拉选项（useMemo 稳定引用，防止输入时失焦）
  const suggestOptions = useMemo(() =>
    searchResults.map(s => ({
      value: s.code,
      label: (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 0' }}>
          <Text style={{ color: '#93c5fd', fontFamily: "'JetBrains Mono', monospace", fontSize: 13 }}>
            {s.code}
          </Text>
          <Text style={{ color: '#e2e8f0', fontSize: 13 }}>{s.name}</Text>
          <Tag color={marketColor(s.market)} style={{ marginLeft: 'auto', borderRadius: 3, fontSize: 10 }}>
            {marketLabel(s.market)}
          </Tag>
        </div>
      ),
    })), [searchResults])

  // 股票表格列
  const stockColumns = (_groupName: string) => [
    {
      title: '代码',
      dataIndex: 'code',
      width: 100,
      render: (code: string) => (
        <Text style={{ fontFamily: "'JetBrains Mono', monospace", color: '#93c5fd' }}>{code}</Text>
      ),
    },
    {
      title: '名称',
      dataIndex: 'name',
      width: 120,
      render: (name: string) => <Text style={{ color: '#e2e8f0' }}>{name}</Text>,
    },
    {
      title: '市场',
      dataIndex: 'market',
      width: 60,
      render: (m: string) => (
        <Tag
          color={marketColor(m)}
          style={{ borderRadius: 4, fontSize: 11, minWidth: 28, textAlign: 'center' }}
        >
          {marketLabel(m)}
        </Tag>
      ),
    },
    {
      title: '操作',
      width: 120,
      render: (_: any, record: StockItem) => (
        <Space size={4}>
          <Tooltip title="换组">
            <Button
              type="text"
              size="small"
              icon={<SwapOutlined />}
              onClick={() => {
                setMoveStock(record)
                setMoveTarget('')
                setMoveModalOpen(true)
              }}
            />
          </Tooltip>
          <Popconfirm
            title={`确认移除 ${record.name}?`}
            onConfirm={() => handleRemove(record.code, record.name)}
            okText="移除"
            cancelText="取消"
          >
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ maxWidth: 1200 }}>
      {/* 顶部操作栏 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>自选股管理</Title>
          <Text style={{ color: '#64748b', fontSize: 13 }}>
            管理分组与自选股，配置将同步至策略引擎
          </Text>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => fetchWatchlist()}>刷新</Button>
          <Button icon={<FolderAddOutlined />} onClick={() => setGroupModalOpen(true)}>新建分组</Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setAddModalOpen(true)}>
            添加股票
          </Button>
        </Space>
      </div>

      <Spin spinning={loading}>
        {groups.length === 0 && ungrouped.length === 0 ? (
          <Empty description="暂无自选股，点击「添加股票」开始" />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {groups.map(group => (
              <Card
                key={group.name}
                size="small"
                title={
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{group.name}</span>
                    <Tag style={{ borderRadius: 4, fontSize: 11 }}>{group.stocks.length}只</Tag>
                  </div>
                }
                extra={
                  <Popconfirm
                    title={`删除分组「${group.name}」? 股票将移至未分组`}
                    onConfirm={() => deleteGroup(group.name)}
                    okText="删除"
                    cancelText="取消"
                  >
                    <Button type="text" size="small" danger icon={<DeleteOutlined />}>
                      删除组
                    </Button>
                  </Popconfirm>
                }
                style={{ background: '#0f1729', borderColor: '#1e2d45' }}
                styles={{ header: { borderBottomColor: '#1e2d45' } }}
              >
                <Table
                  dataSource={group.stocks}
                  columns={stockColumns(group.name)}
                  rowKey="code"
                  size="small"
                  pagination={false}
                  style={{ background: 'transparent' }}
                />
              </Card>
            ))}

            {ungrouped.length > 0 && (
              <Card
                size="small"
                title={
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: '#94a3b8', fontWeight: 600 }}>未分组</span>
                    <Tag style={{ borderRadius: 4, fontSize: 11 }}>{ungrouped.length}只</Tag>
                  </div>
                }
                style={{ background: '#0f1729', borderColor: '#1e2d45' }}
                styles={{ header: { borderBottomColor: '#1e2d45' } }}
              >
                <Table
                  dataSource={ungrouped}
                  columns={stockColumns('')}
                  rowKey="code"
                  size="small"
                  pagination={false}
                />
              </Card>
            )}
          </div>
        )}
      </Spin>

      {/* 添加股票弹窗 */}
      <Modal
        title="添加自选股"
        open={addModalOpen}
        onCancel={() => {
          setAddModalOpen(false)
          inputRef.current = ''
          setSearchResults([])
          if (debounceTimer.current) clearTimeout(debounceTimer.current)
        }}
        footer={null}
        width={560}
      >
        <div style={{ marginBottom: 16 }}>
          <AutoComplete
            options={suggestOptions}
            onSearch={handleSearch}
            onSelect={(code) => {
              const stock = searchResults.find(s => s.code === code)
              if (stock) handleAdd(stock, selectedGroup || undefined)
            }}
            notFoundContent={
              searchLoading
                ? <Spin size="small" style={{ padding: 8 }} />
                : (inputRef.current ? <Text style={{ color: '#64748b', padding: 8 }}>未找到匹配股票，请检查输入</Text> : null)
            }
            style={{ width: '100%' }}
          >
            <Input
              size="large"
              placeholder="输入代码 / 名称 / 拼音，实时联想"
            />
          </AutoComplete>
        </div>

        {searchResults.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <Text style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, display: 'block' }}>
              搜索提示 ({searchResults.length} 条)
            </Text>
          </div>
        )}

        <div>
          <Text style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, display: 'block' }}>
            添加到分组 (可选)
          </Text>
          <Select
            placeholder="选择分组"
            value={selectedGroup || undefined}
            onChange={setSelectedGroup}
            allowClear
            style={{ width: '100%' }}
            options={allGroupNames.map(n => ({ label: n, value: n }))}
          />
        </div>
      </Modal>

      {/* 新建分组弹窗 */}
      <Modal
        title="新建分组"
        open={groupModalOpen}
        onOk={handleCreateGroup}
        onCancel={() => { setGroupModalOpen(false); setNewGroupName(''); setNewGroupDesc('') }}
        okText="创建"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>分组名称</Text>
            <Input
              placeholder="如: 科技成长型"
              value={newGroupName}
              onChange={e => setNewGroupName(e.target.value)}
            />
          </div>
          <div>
            <Text style={{ color: '#94a3b8', fontSize: 12, display: 'block', marginBottom: 4 }}>描述 (可选)</Text>
            <Input
              placeholder="如: 高Beta、题材驱动"
              value={newGroupDesc}
              onChange={e => setNewGroupDesc(e.target.value)}
            />
          </div>
        </div>
      </Modal>

      {/* 换组弹窗 */}
      <Modal
        title={`移动 ${moveStock?.name || ''} 到分组`}
        open={moveModalOpen}
        onOk={handleMove}
        onCancel={() => { setMoveModalOpen(false); setMoveStock(null) }}
        okText="确认移动"
      >
        <Select
          placeholder="选择目标分组"
          value={moveTarget || undefined}
          onChange={setMoveTarget}
          style={{ width: '100%' }}
          options={[
            { label: '移出分组 (未分组)', value: '' },
            ...allGroupNames.map(n => ({ label: n, value: n })),
          ]}
        />
      </Modal>
    </div>
  )
}
