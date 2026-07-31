import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  StockOutlined,
  SettingOutlined,
  LineChartOutlined,
  FundOutlined,
  ThunderboltOutlined,
  AimOutlined,
} from '@ant-design/icons'

const { Sider, Content, Header } = Layout

const menuItems = [
  { key: '/signals', icon: <ThunderboltOutlined />, label: '信号看板' },
  { key: '/watchlist', icon: <StockOutlined />, label: '自选股管理' },
  { key: '/config', icon: <SettingOutlined />, label: '策略配置' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测分析' },
  { key: '/stock-backtest', icon: <AimOutlined />, label: '个股回测' },
]

export default function MainLayout() {
  const navigate = useNavigate()
  const location = useLocation()

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
