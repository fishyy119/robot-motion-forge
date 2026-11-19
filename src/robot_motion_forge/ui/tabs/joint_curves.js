/* 根据时间轴时刻更新侧栏图表的标线，缩放时沿用 uPlot 的坐标变换。 */
(() => {
    const lines = new WeakMap();
    let time = NaN;
    let pending = false;

    function position(chart, line) {
        const { min, max } = chart.scales.x;
        line.hidden = !Number.isFinite(time) || min == null || max == null || time < min || time > max;
        if (!line.hidden) line.style.left = `${chart.valToPos(time, "x")}px`;
    }

    function refresh() {
        pending = false;
        const value = document.querySelector("[data-rmf-curve-time]")?.textContent;
        time = value ? Number(value) : NaN;
        for (const chart of rmfViser.visiblePlots()) {
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
