import { create } from 'zustand'

interface BacktestResult {
  total_return: number
  annual_return: number
  max_drawdown: number
  sharpe_ratio: number
  trade_count: number
  win_rate: number
  profit_factor: number
  total_pnl: number
  initial_capital: number
  final_value: number
}

interface Trade {
  symbol: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  pnl: number
  pnl_pct: number
  holding_days: number
  entry_signal: string
}

interface BacktestState {
  running: boolean
  progress: string
  result: BacktestResult | null
  trades: Trade[]
  equityCurve: { date: string; value: number; benchmark: number }[]
  runBacktest: (params: any) => Promise<void>
  fetchResult: () => Promise<void>
}

export const useBacktestStore = create<BacktestState>((set) => ({
  running: false,
  progress: '',
  result: null,
  trades: [],
  equityCurve: [],

  runBacktest: async (params) => {
    set({ running: true, progress: '启动回测...' })
    try {
      const { backtestApi } = await import('../api')
      const { data } = await backtestApi.run(params)
      const taskId = data.task_id

      // SSE 轮询进度
      let done = false
      while (!done) {
        await new Promise(r => setTimeout(r, 500))
        try {
          const { data: status } = await backtestApi.getStatus(taskId)
          set({ progress: status.message || '计算中...' })
          if (status.done) {
            done = true
          }
        } catch {
          done = true
        }
      }

      set({ running: false, progress: '' })
    } catch {
      set({ running: false, progress: '' })
    }
  },

  fetchResult: async () => {
    try {
      const { backtestApi } = await import('../api')
      const [resultRes, tradesRes] = await Promise.all([
        backtestApi.getResult(),
        backtestApi.getTrades(),
      ])
      set({
        result: resultRes.data,
        trades: tradesRes.data.trades || [],
      })
    } catch {
      // 后端尚未实现这些接口
    }
  },
}))
