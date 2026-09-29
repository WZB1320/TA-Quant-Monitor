/** 大屏等比缩放适配：把固定 1920×1200 的舞台缩放进容器（保持设计比例不失真） */
import { useCallback, useEffect, useRef } from 'react'

export const STAGE_W = 1920
export const STAGE_H = 1200

export function useFitScale() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    const stage = stageRef.current
    if (!wrap || !stage) return
    const w = wrap.clientWidth
    const h = wrap.clientHeight
    if (!w || !h) return
    const scale = Math.min(w / STAGE_W, h / STAGE_H)
    stage.style.transform = `scale(${scale})`
    // 缩放后舞台实际占位小于容器时，水平/垂直居中
    stage.style.marginLeft = `${Math.max(0, (w - STAGE_W * scale) / 2)}px`
    stage.style.marginTop = `${Math.max(0, (h - STAGE_H * scale) / 2)}px`
  }, [])

  useEffect(() => {
    fit()
    const ro = new ResizeObserver(fit)
    if (wrapRef.current) ro.observe(wrapRef.current)
    window.addEventListener('resize', fit)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', fit)
    }
  }, [fit])

  return { wrapRef, stageRef, fit }
}
