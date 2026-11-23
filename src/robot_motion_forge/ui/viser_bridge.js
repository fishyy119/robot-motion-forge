/* 集中适配 Viser 浏览器内部接口；各功能独立初始化。 */
const extensions = Symbol.for("robot-motion-forge.browser-extensions");
if (window[extensions]) return;
window[extensions] = true;
document.title = "Robot Motion Forge";

// 样式随页面初始化，侧栏收起后仍然生效。
const stylesheet = document.createElement("style");
stylesheet.textContent = css;
document.head.append(stylesheet);

const rmfViser = (() => {
    function* reactFibers(element) {
        // HTML 控件中的自定义节点无 React 句柄，从最近的宿主元素开始。
        for (let host = element; host; host = host.parentElement) {
            const key = Object.keys(host).find(name => name.startsWith("__reactFiber"));
            if (!key) continue;
            for (let fiber = host[key]; fiber; fiber = fiber.return) yield fiber;
            return;
        }
    }

    let viewer;
    function whenViewer(ready) {
        if (!viewer) {
            for (const selector of ["[data-rmf-timeline]", "[data-dock-root]", "canvas"]) {
                for (const fiber of reactFibers(document.querySelector(selector))) {
                    const value = fiber.memoizedProps?.value;
                    if (value?.useGuiConfig && value?.mutable?.current) {
                        viewer = value;
                        break;
                    }
                }
                if (viewer) break;
            }
        }
        if (viewer) ready(viewer);
        else requestAnimationFrame(() => whenViewer(ready));
    }

    function whenScene(viewer, ready) {
        const current = viewer.mutable.current;
        const ref = Object.values(current.nodeRefFromName).find(ref => (ref?.current ?? ref)?.__r3f?.root);
        const store = (ref?.current ?? ref)?.__r3f?.root;
        if (current.canvas && store) ready(viewer, current, current.canvas, store);
        else requestAnimationFrame(() => whenScene(viewer, ready));
    }

    const connected = viewer => viewer.useGui.get().websocketState === "connected";
    const guiConfig = (viewer, uuid) => viewer.useGuiConfig.get(uuid);
    function setGuiValue(viewer, uuid, value, preview = false) {
        if (!connected(viewer)) return;
        if (preview) viewer.guiActions.updateGuiProps(uuid, { value });
        viewer.mutable.current.sendMessage({ type: "GuiUpdateMessage", uuid, updates: { value } });
    }

    const expandedPlots = new WeakMap();
    function expandedPlot(container) {
        const root = container.querySelector(".uplot");
        if (!root) return;
        const cached = expandedPlots.get(root);
        if (cached) return cached;
        // Viser 只注册侧栏实例，弹窗的实例由它自己的 PlotComponent 持有。
        for (const fiber of reactFibers(container)) {
            for (const node of [fiber, fiber.alternate]) {
                for (let hook = node?.memoizedState; hook; hook = hook.next) {
                    const chart = hook.memoizedState;
                    if (chart?.root === root) {
                        expandedPlots.set(root, chart);
                        return chart;
                    }
                }
            }
        }
    }

    function visiblePlots() {
        const charts = Object.values(window.__viserTestpoints?.uplots ?? {}).map(({ chart }) => chart);
        for (const container of document.querySelectorAll('[role="dialog"] .uplot-container')) {
            const chart = expandedPlot(container);
            if (chart) charts.push(chart);
        }
        return charts;
    }

    const timelineLabels = slider => slider.querySelectorAll(".mantine-Slider-markLabel");
    return { whenViewer, whenScene, connected, guiConfig, setGuiValue, visiblePlots, timelineLabels };
})();
