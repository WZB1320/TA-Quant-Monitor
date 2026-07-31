import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ConfigProvider, theme, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import MainLayout from './components/MainLayout'
import WatchlistPage from './pages/Watchlist'
import SignalsPage from './pages/Signals'
import ConfigPage from './pages/Config'
import BacktestPage from './pages/Backtest'
import StockBacktestPage from './pages/StockBacktest'

function App() {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#3b82f6',
          colorBgContainer: '#0f1729',
          colorBgElevated: '#162032',
          colorBgLayout: '#080e1a',
          colorBorder: '#1e2d45',
          colorText: '#e2e8f0',
          colorTextSecondary: '#94a3b8',
          borderRadius: 8,
          fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas', monospace",
        },
        components: {
          Table: {
            headerBg: '#0d1829',
            rowHoverBg: '#162032',
            borderColor: '#1e2d45',
          },
          Card: {
            colorBgContainer: '#0f1729',
          },
          Input: {
            colorBgContainer: '#0d1829',
          },
          Select: {
            colorBgContainer: '#0d1829',
          },
          Modal: {
            contentBg: '#0f1729',
            headerBg: '#0f1729',
          },
        },
      }}
    >
      <AntApp>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<MainLayout />}>
              <Route index element={<Navigate to="/signals" replace />} />
              <Route path="signals" element={<SignalsPage />} />
              <Route path="watchlist" element={<WatchlistPage />} />
              <Route path="config" element={<ConfigPage />} />
              <Route path="backtest" element={<BacktestPage />} />
              <Route path="stock-backtest" element={<StockBacktestPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}

export default App
