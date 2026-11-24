/* 集中适配 Viser 浏览器内部接口；各功能独立初始化。 */
/** @param {string} css */
function initializeBrowserExtensions(css) {
    const extensions = Symbol.for("robot-motion-forge.browser-extensions");
    const browser = /** @type {Window & { [extensions]?: true }} */ (window);
    if (browser[extensions]) return false;
    browser[extensions] = true;
    document.title = "Robot Motion Forge";

    // 样式随页面初始化，侧栏收起后仍然生效。
    const stylesheet = document.createElement("style");
    stylesheet.textContent = css;
    document.head.append(stylesheet);
    return true;
}

const rmfViser = (() => {
    /**
     * @typedef {{ memoizedState: unknown, next: ReactHook | null }} ReactHook
     * @typedef {{
     *     return: ReactFiber | null,
     *     alternate: ReactFiber | null,
     *     memoizedProps?: { value?: unknown } | null,
     *     memoizedState?: ReactHook | null,
     * }} ReactFiber
     */

    /** @param {Element | null} element */
    function* reactFibers(element) {
        // HTML 控件中的自定义节点无 React 句柄，从最近的宿主元素开始。
        for (let host = element; host; host = host.parentElement) {
            const key = Object.keys(host).find(name => name.startsWith("__reactFiber"));
            if (!key) continue;
            const fiber = /** @type {ReactFiber | null} */ (Reflect.get(host, key));
            for (let node = fiber; node; node = node.return) yield node;
            return;
        }
    }

    /** @type {ViserBrowser.Viewer | undefined} */
    let viewer;
    /** @param {(viewer: ViserBrowser.Viewer) => void} ready */
    function whenViewer(ready) {
        if (!viewer) {
            for (const selector of ["[data-rmf-timeline]", "[data-dock-root]", "canvas"]) {
                for (const fiber of reactFibers(document.querySelector(selector))) {
                    const value = /** @type {ViserBrowser.Viewer | undefined} */ (fiber.memoizedProps?.value);
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

    /** @param {ViserBrowser.SceneNodeRef | undefined} ref */
    function sceneObject(ref) {
        return ref && "current" in ref ? ref.current : ref;
    }

    /**
     * @param {ViserBrowser.Viewer} viewer
     * @param {ViserBrowser.SceneReady} ready
     */
    function whenScene(viewer, ready) {
        const current = viewer.mutable.current;
        const ref = Object.values(current.nodeRefFromName).find(ref => sceneObject(ref)?.__r3f?.root);
        const store = sceneObject(ref)?.__r3f?.root;
        if (current.canvas && store) ready(viewer, current, current.canvas, store);
        else requestAnimationFrame(() => whenScene(viewer, ready));
    }

    /** @param {ViserBrowser.Viewer} viewer */
    const connected = viewer => viewer.useGui.get().websocketState === "connected";

    /**
     * @param {ViserBrowser.Viewer} viewer
     * @param {string} uuid
     */
    function guiConfig(viewer, uuid) {
        const config = /** @type {ViserBrowser.GuiConfig | undefined} */ (viewer.useGuiConfig.get(uuid));
        if (config?.type === "GuiCheckboxMessage" || config?.type === "GuiDropdownMessage") return config;
    }

    /**
     * @param {ViserBrowser.Viewer} viewer
     * @param {string} uuid
     * @param {ViserBrowser.GuiValue} value
     */
    function setGuiValue(viewer, uuid, value, preview = false) {
        if (!connected(viewer)) return;
        if (preview) viewer.guiActions.updateGuiProps(uuid, { value });
        viewer.mutable.current.sendMessage({ type: "GuiUpdateMessage", uuid, updates: { value } });
    }

    /** @type {WeakMap<Element, ViserBrowser.Plot>} */
    const expandedPlots = new WeakMap();
    /** @param {Element} container */
    function expandedPlot(container) {
        const root = container.querySelector(".uplot");
        if (!root) return;
        const cached = expandedPlots.get(root);
        if (cached) return cached;
        // Viser 只注册侧栏实例，弹窗的实例由它自己的 PlotComponent 持有。
        for (const fiber of reactFibers(container)) {
            for (const node of [fiber, fiber.alternate]) {
                for (let hook = node?.memoizedState; hook; hook = hook.next) {
                    const chart = /** @type {ViserBrowser.Plot | undefined} */ (hook.memoizedState);
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

    /**
     * @param {HTMLElement} slider
     * @returns {NodeListOf<HTMLElement>}
     */
    const timelineLabels = slider => slider.querySelectorAll(".mantine-Slider-markLabel");
    return { whenViewer, whenScene, sceneObject, connected, guiConfig, setGuiValue, visiblePlots, timelineLabels };
})();
