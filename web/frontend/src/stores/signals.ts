import { create } from 'zustand'
import { signalsApi } from '../api'

interface SignalIndicators {
  ma60?: number
  ma60_dir?: string
  rsi?: number
  dif?: number
  dea?: number
  adx?: number
  volume_ratio?: number
  atr_pct?: number
}

interface SignalExecution {
  executable: boolean
  status: string
  reason: string
}

interface SignalResult {
  code: string
  name: string
  market: string
  group: string
  close: number
  date: string
  level: string
  label: string
  score: number
  confidence: number
  reason: string
  details: string
  actionable: boolean
  execution: SignalExecution
  initial_level: string
  demotion_chain: string[]
  hard_filter_blocked: boolean
  block_reason: string
  block_detail: string
  indicators: SignalIndicators
}

interface SignalSummary {
  bullish: number
  neutral: number
  bearish: number
  actionable: number
  total: number
}

interface SignalGroup {
  name: string
  count: number
}

interface SignalsState {
  results: SignalResult[]
  summary: SignalSummary
  analyzedAt: string | null
  analyzedGroup: string
  userRegime: string
  running: boolean
  progress: number
  total: number
  hasResult: boolean
  groups: SignalGroup[]
  runAnalysis: (group?: string, userRegime?: string) => Promise<void>
  pollStatus: () => Promise<void>
  fetchResult: () => Promise<void>
  fetchGroups: () => Promise<void>
}

export const useSignalsStore = create<SignalsState>((set, get) => ({
  results: [],
  summary: { bullish: 0, neutral: 0, bearish: 0, actionable: 0, total: 0 },
  analyzedAt: null,
  analyzedGroup: '',
  userRegime: 'auto',
  running: false,
  progress: 0,
  total: 0,
  hasResult: false,
  groups: [],

  runAnalysis: async (group?: string, userRegime?: string) => {
    try {
      await signalsApi.run(group, userRegime)
      set({ running: true, progress: 0, total: 0, analyzedGroup: group || '', userRegime: userRegime || 'auto' })
      const poll = async () => {
        await get().pollStatus()
        if (get().running) {
          setTimeout(poll, 1500)
        }
      }
      poll()
    } catch {
      set({ running: false })
    }
  },

  pollStatus: async () => {
    try {
      const { data } = await signalsApi.getStatus()
      set({
        running: data.running,
        progress: data.progress,
        total: data.total,
        hasResult: data.has_result,
      })
      if (!data.running && data.has_result) {
        await get().fetchResult()
      }
    } catch {
      // 忽略轮询错误
    }
  },

  fetchResult: async () => {
    try {
      const { data } = await signalsApi.getResult()
      set({
        results: data.results || [],
        summary: data.summary || { bullish: 0, neutral: 0, bearish: 0, actionable: 0, total: 0 },
        analyzedAt: data.analyzed_at,
        analyzedGroup: data.group || '',
        hasResult: data.status === 'done',
      })
    } catch {
      // 忽略
    }
  },

  fetchGroups: async () => {
    try {
      const res = await signalsApi.getGroups()
      console.log('[signals] fetchGroups response:', res.data)
      set({ groups: res.data?.groups || [] })
    } catch (e) {
      console.error('[signals] fetchGroups error:', e)
    }
  },
}))
