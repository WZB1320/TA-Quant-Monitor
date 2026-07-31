import { create } from 'zustand'

interface StockItem {
  code: string
  name: string
  market: string
}

interface WatchlistGroup {
  name: string
  stocks: StockItem[]
}

interface WatchlistState {
  groups: WatchlistGroup[]
  ungrouped: StockItem[]
  loading: boolean
  fetchWatchlist: () => Promise<void>
  addStock: (code: string, name: string, market?: string, group?: string) => Promise<void>
  removeStock: (code: string) => Promise<void>
  updateStock: (code: string, data: { name?: string; market?: string; group?: string }) => Promise<void>
  createGroup: (name: string, description?: string) => Promise<void>
  deleteGroup: (name: string) => Promise<void>
}

export const useWatchlistStore = create<WatchlistState>((set, get) => ({
  groups: [],
  ungrouped: [],
  loading: false,

  fetchWatchlist: async () => {
    set({ loading: true })
    try {
      const { data } = await import('../api').then(m => m.watchlistApi.getList())
      set({ groups: data.groups || [], ungrouped: data.ungrouped || [], loading: false })
    } catch {
      set({ loading: false })
    }
  },

  addStock: async (code, name, market, group) => {
    await import('../api').then(m => m.watchlistApi.addStock({ code, name, market, group }))
    await get().fetchWatchlist()
  },

  removeStock: async (code) => {
    await import('../api').then(m => m.watchlistApi.removeStock(code))
    await get().fetchWatchlist()
  },

  updateStock: async (code, data) => {
    await import('../api').then(m => m.watchlistApi.updateStock(code, data))
    await get().fetchWatchlist()
  },

  createGroup: async (name, description) => {
    await import('../api').then(m => m.watchlistApi.createGroup({ name, description }))
    await get().fetchWatchlist()
  },

  deleteGroup: async (name) => {
    await import('../api').then(m => m.watchlistApi.deleteGroup(name))
    await get().fetchWatchlist()
  },
}))
