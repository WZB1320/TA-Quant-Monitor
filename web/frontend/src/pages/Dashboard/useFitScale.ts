/** 大屏缩放适配：以「宽度」为唯一基准
 *
 * 为什么不用 min(宽比, 高比)：等比缩放会被瓶颈维度锁死 —— 容器比 1.6 更扁时高度成为瓶颈，
 * 左侧菜单栏释放的宽度 100% 被浪费（实测 1920×1049 下收起菜单视觉尺寸变化 +0.0%）。
 * 改为纯宽度基准后：横向永远铺满，高度交给内容 + 纵向滚动 —— 这样「字号更大 + 行数可增长」才成立。
 * 副作用（正向）：收起菜单会让容器变宽 → 缩放比变大 → 整屏等比放大（实测 +8.5% 字号）。
 *
 * 为什么需要 fitBox 占位：transform 不改变布局尺寸，舞台上未缩放的 1920px 宽仍会撑出横向溢出
 * （scrollWidth > clientWidth），聚焦元素时浏览器可能把内容横向带偏。占位容器用
 * 「设计高 × 缩放比」撑出精确的滚动高度，宽度交给 100%，从而彻底消除横向溢出。
 */
import { useCallback, useEffect, useRef } from 'react'

export const STAGE_W = 1920

export function useFitScale() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const fitBoxRef = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)

  const fit = useCallback(() => {
    const wrap = wrapRef.current
    const fitBox = fitBoxRef.current
    const stage = stageRef.current
    if (!wrap || !fitBox || !stage) return
    const w = wrap.clientWidth
    if (!w) return
    // clientWidth 已排除纵向滚动条（且 .dash-root 预留了 scrollbar-gutter），故缩放后正好铺满
    const scale = w / STAGE_W
    stage.style.transform = `scale(${scale})`
    // offsetHeight 是未缩放的设计高度；行数变化时舞台会变高，占位高度随之更新
    fitBox.style.height = `${stage.offsetHeight * scale}px`
  }, [])

  useEffect(() => {
    fit()
    const ro = new ResizeObserver(fit)
    if (wrapRef.current) ro.observe(wrapRef.current)   // 宽度变化（窗口/菜单栏收起）
    if (stageRef.current) ro.observe(stageRef.current) // 设计高度变化（行数增减）
    window.addEventListener('resize', fit)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', fit)
    }
  }, [fit])

  return { wrapRef, fitBoxRef, stageRef, fit }
}
