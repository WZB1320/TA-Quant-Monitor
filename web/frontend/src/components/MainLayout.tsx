import { useEffect, useState } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Button, Tooltip } from 'antd'
import {
  StockOutlined,
  SettingOutlined,
  LineChartOutlined,
  FundOutlined,
  ThunderboltOutlined,
  AimOutlined,
  DashboardOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons'

const { Sider, Content, Header } = Layout

const menuItems = [
  { key: '/signals', icon: <ThunderboltOutlined />, label: '信号看板' },
  { key: '/watchlist', icon: <StockOutlined />, label: '自选股管理' },
  { key: '/config', icon: <SettingOutlined />, label: '策略配置' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测分析' },
  { key: '/stock-backtest', icon: <AimOutlined />, label: '个股回测' },
  { key: '/dashboard', icon: <DashboardOutlined />, label: '复盘大屏' },
]

/** 菜单栏收展状态：收起 = 仅图标 64px，展开 = 200px（全宽）。状态存本地，刷新后保持 */
const SIDER_COLLAPSED_KEY = 'qm.sider.collapsed'

function readStoredCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDER_COLLAPSED_KEY) === '1'
  } catch {
    return false // 隐私模式等 localStorage 不可用时按展开处理
  }
}

export default function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(readStoredCollapsed)

  useEffect(() => {
    try {
      localStorage.setItem(SIDER_COLLAPSED_KEY, collapsed ? '1' : '0')
    } catch {
      // 写不进去不影响功能，仅丢失记忆
    }
  }, [collapsed])

  const toggleLabel = collapsed ? '展开菜单' : '收起菜单'

  return (
    <Layout style={{ minHeight: '100vh', background: '#080e1a' }}>
      <Header
        style={{
          background: 'linear-gradient(90deg, #0a1628 0%, #0f1f3d 100%)',
          borderBottom: '1px solid #1e2d45',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          height: 56,
          gap: 12,
        }}
      >
        <Tooltip title={toggleLabel} placement="bottomLeft">
          <Button
            type="text"
            aria-label={toggleLabel}
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(v => !v)}
            style={{
              color: '#94a3b8',
              fontSize: 17,
              width: 34,
              height: 34,
              marginLeft: -10,
              flex: 'none',
            }}
          />
        </Tooltip>
        <FundOutlined style={{ fontSize: 22, color: '#3b82f6' }} />
        <span
          style={{
            fontSize: 17,
            fontWeight: 700,
            color: '#e2e8f0',
            letterSpacing: '0.5px',
          }}
        >
          QuantMonitor
        </span>
        <span
          style={{
            fontSize: 12,
            color: '#475569',
            marginLeft: 8,
            letterSpacing: '1px',
          }}
        >
          量化交易监控系统
        </span>
      </Header>

      <Layout>
        <Sider
          width={200}
          collapsedWidth={64}
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          trigger={null}
          style={{
            background: '#0a1628',
            borderRight: '1px solid #1e2d45',
          }}
        >
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
            style={{
              background: 'transparent',
              borderRight: 'none',
              marginTop: 8,
            }}
            theme="dark"
          />
        </Sider>

        <Content
          style={{
            padding: 24,
            background: '#080e1a',
            overflow: 'auto',
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
