"""L2 教训注入周报 prompt — 单元测试

覆盖:
  - _dominant_regime: by_regime 优先, 排除 unknown, 回退 signal_by_regime
  - _build_lesson_context: 按主导体制检索, 注入 root_cause/rule/do_not_repeat
  - invalidated 教训被过滤
  - 体制不匹配时不注入
  - 无教训 / 空库 返回空串
  - max_items 限制
"""
import pytest

from src.ai.report_generator import WeeklyReportGenerator
from src.memory.lesson_library import LessonLibrary


@pytest.fixture
def lib(tmp_path):
    return LessonLibrary(lessons_dir=str(tmp_path / "lessons"))


@pytest.fixture
def gen(lib):
    # llm 用占位对象: 本测试只验证 prompt 构建, 不触发真实调用
    return WeeklyReportGenerator(llm=object(), lesson_library=lib)


def _make_lesson(lib, *, regime=None, level="param", root="参数问题",
                 rule="调参", dnr=None):
    return lib.create_lesson(
        linked_cases=["fc_a", "fc_b"],
        scope={"group": None, "regime": regime, "strategy_mode": None},
        failure_level=level,
        root_cause=root,
        rule=rule,
        do_not_repeat=dnr or [],
    )


# ─────────────────────────────────────────────────────────────
# _dominant_regime
# ─────────────────────────────────────────────────────────────

def test_dominant_regime_from_by_regime(gen):
    stats = {"by_regime": [
        {"name": "ranging", "count": 10},
        {"name": "trending", "count": 3},
        {"name": "unknown", "count": 20},
    ]}
    assert gen._dominant_regime(stats) == "ranging"


def test_dominant_regime_fallback_to_signals(gen):
    stats = {"signal_by_regime": {"ranging": 5, "trending": 9}}
    assert gen._dominant_regime(stats) == "trending"


def test_dominant_regime_none(gen):
    assert gen._dominant_regime({}) is None
    assert gen._dominant_regime(
        {"by_regime": [{"name": "unknown", "count": 5}]}) is None


# ─────────────────────────────────────────────────────────────
# _build_lesson_context
# ─────────────────────────────────────────────────────────────

def test_lesson_context_hits_regime_and_wildcard(gen, lib):
    _make_lesson(lib, regime="ranging", level="regime_mismatch",
                 root="震荡市均值回归失效", rule="ranging 降低仓位",
                 dnr=["ranging 满仓均值回归"])
    _make_lesson(lib, regime=None, level="param",
                 root="score_threshold 偏低", rule="阈值 30 → 38",
                 dnr=["score_threshold <= 30"])

    stats = {"by_regime": [{"name": "ranging", "count": 10},
                           {"name": "trending", "count": 2}]}
    ctx = gen._build_lesson_context(stats)

    assert "历史失效教训" in ctx
    assert "不得与这些方向冲突" in ctx
    assert "震荡市均值回归失效" in ctx          # 精确匹配 ranging
    assert "ranging 降低仓位" in ctx
    assert "ranging 满仓均值回归" in ctx        # do_not_repeat 注入
    assert "score_threshold <= 30" in ctx      # 通配教训也命中
    assert ctx.count("根因:") == 2


def test_lesson_context_excludes_invalidated(gen, lib):
    lesson = _make_lesson(lib, regime=None, root="已被证伪的教训",
                          dnr=["旧方向"])
    lib.mark_applied(lesson["lesson_id"], "run_a")
    lib.mark_invalidated(lesson["lesson_id"], "run_b", notes="反而恶化")

    stats = {"by_regime": [{"name": "ranging", "count": 5}]}
    ctx = gen._build_lesson_context(stats)
    assert ctx == ""


def test_lesson_context_skips_wrong_regime(gen, lib):
    # lesson 绑定 trending, 当前主导 ranging → 不匹配
    _make_lesson(lib, regime="trending", root="趋势策略教训")
    stats = {"by_regime": [{"name": "ranging", "count": 8}]}
    ctx = gen._build_lesson_context(stats)
    assert ctx == ""


def test_lesson_context_empty_library(gen):
    assert gen._build_lesson_context({"by_regime": [{"name": "ranging", "count": 5}]}) == ""


def test_lesson_context_no_dominant_regime_returns_all(gen, lib):
    # 无体制信息时 scope={}, 不约束, 全部教训命中
    _make_lesson(lib, regime="ranging", root="教训A")
    _make_lesson(lib, regime=None, root="教训B")
    ctx = gen._build_lesson_context({})
    assert "教训A" in ctx and "教训B" in ctx


def test_lesson_context_max_items(gen, lib):
    for i in range(6):
        _make_lesson(lib, regime=None, root=f"教训{i}", rule=f"规则{i}")
    ctx = gen._build_lesson_context({"by_regime": [{"name": "ranging", "count": 5}]})
    assert ctx.count("根因:") == 5  # 默认上限 5


def test_lesson_context_ignores_invalid_lessons(gen):
    class _BrokenLib:
        def find_relevant_lessons(self, scope):
            raise RuntimeError("boom")

    broken = WeeklyReportGenerator(llm=object(), lesson_library=_BrokenLib())
    assert broken._build_lesson_context({"by_regime": [{"name": "x", "count": 1}]}) == ""
