import axios from 'axios'

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// ── 自选股 ──

export const watchlistApi = {
  getList: () => api.get('/watchlist'),

  addStock: (data: { code: string; name: string; market?: string; group?: string }) =>
    api.post('/watchlist', data),

  removeStock: (code: string) => api.delete(`/watchlist/${code}`),

  updateStock: (code: string, data: { name?: string; market?: string; group?: string }) =>
    api.put(`/watchlist/${code}`, data),

  createGroup: (data: { name: string; description?: string }) =>
    api.post('/watchlist/groups', data),

  deleteGroup: (name: string) => api.delete(`/watchlist/groups/${name}`),
}

// ── 股票知识库搜索 ──

export const stockApi = {
  search: (q: string, limit?: number) =>
    api.get('/stocks/search', { params: { q, limit: limit ?? 10 } }),
}

// ── 配置 ──

export const configApi = {
  get: () => api.get('/config'),
  update: (data: any) => api.put('/config', data),
  getGroups: () => api.get('/config/groups'),
  getGroupConfig: (name: string, mode?: string) =>
    api.get(`/config/groups/${name}`, { params: mode ? { mode } : {} }),
  saveGroupConfig: (name: string, data: any) =>
    api.put(`/config/groups/${name}`, data),
  saveGroupPreset: (name: string, mode: string, data: any) =>
    api.put(`/config/groups/${name}/presets/${mode}`, data),
  getIndicators: () => api.get('/config/indicators'),
}

// ── 回测 ──

export const backtestApi = {
  run: (data: {
    groups: string[]
    mode: string
    start_date?: string
    end_date?: string
    initial_capital?: number
    benchmark?: string
  }) => api.post('/backtest/run', data, { timeout: 300000 }),
  getStatus: () => api.get('/backtest/status'),
}

// ── 信号分析 ──

export const signalsApi = {
  run: (group?: string, userRegime?: string) =>
    api.post('/signals/run', { group: group || '', user_regime: userRegime || 'auto' }),
  getStatus: () => api.get('/signals/status'),
  getResult: () => api.get('/signals/result'),
  getGroups: () => api.get('/signals/groups'),
}

// ── 个股信号回测 ──

export const stockBacktestApi = {
  run: (data: {
    code: string
    mode: string
    start_date?: string
    end_date?: string
  }) => api.post('/stock-backtest/run', data, { timeout: 180000 }),
  getStatus: () => api.get('/stock-backtest/status'),
  getResult: () => api.get('/stock-backtest/result'),
  getStocks: () => api.get('/stock-backtest/stocks'),
}

export default api
