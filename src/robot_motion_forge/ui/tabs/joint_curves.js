/* 根据时间轴时刻更新侧栏图表的标线，缩放时沿用 uPlot 的坐标变换。 */
(() => {
    const key = Symbol.for("robot-motion-forge.joint-curves");
    if (window[key]) return;
    window[key] = true;

    const lines = new WeakMap();
    const expanded = new WeakMap();
    let time = NaN;
    let pending = false;

    function position(chart, line) {
        const { min, max } = chart.scales.x;
        line.hidden = !Number.isFinite(time) || min == null || max == null || time < min || time > max;
        if (!line.hidden) line.style.left = `${chart.valToPos(time, "x")}px`;
    }

    function expandedChart(container) {
        const root = container.querySelector(".uplot");
        if (!root) return;
        const cached = expanded.get(root);
        if (cached) return cached;
        // Viser 只注册侧栏实例，弹窗的实例由它自己的 PlotComponent 持有。
        const reactKey = Object.keys(container).find(name => name.startsWith("__reactFiber"));
        for (let fiber = container[reactKey]; fiber; fiber = fiber.return) {
            for (const node of [fiber, fiber.alternate]) {
                for (let hook = node?.memoizedState; hook; hook = hook.next) {
                    const chart = hook.memoizedState;
                    if (chart?.root === root) {
                        expanded.set(root, chart);
                        return chart;
                    }
                }
            }
        }
    }

    function refresh() {
        pending = false;
        const value = document.querySelector("[data-rmf-curve-time]")?.textContent;
        time = value ? Number(value) : NaN;
        const charts = Object.values(window.__viserTestpoints?.uplots ?? {}).map(({ chart }) => chart);
        for (const container of document.querySelectorAll('[role="dialog"] .uplot-container')) {
            const chart = expandedChart(container);
            if (chart) charts.push(chart);
        }
        for (const chart of charts) {
            if (chart.series[1]?.class !== "rmf-joint-curve") continue;
            let line = lines.get(chart);
            if (!line) {
                line = document.createElement("div");
                line.className = "rmf-curve-cursor";
                chart.over.appendChild(line);
                lines.set(chart, line);
                (chart.hooks.draw ??= []).push(() => position(chart, line));
            }
            position(chart, line);
        }
    }

    new MutationObserver(() => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(refresh);
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
    refresh();
})();
