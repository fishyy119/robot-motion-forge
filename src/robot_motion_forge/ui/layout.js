/* 浮动时间轴与 3D 区域的两侧对齐。 */
(() => {
    /** @type {HTMLElement | null} */
    let scene = null;
    const sceneSize = new ResizeObserver(syncBounds);
    const scenePosition = new MutationObserver(syncBounds);

    function syncBounds() {
        const dock = scene?.parentElement;
        if (!scene?.isConnected || !dock) return;
        const width = scene.clientWidth;
        const inset = Math.min(16, Math.floor(width / 2));
        dock.style.setProperty("--rmf-timeline-left", `${scene.offsetLeft + inset}px`);
        dock.style.setProperty("--rmf-timeline-width", `${width - 2 * inset}px`);
        dock.dataset.rmfTimelineBounds = "";
    }

    function bindLayout() {
        // DockManager 的第一个子元素是场景容器，随侧栏同步改变尺寸。
        const child = document.querySelector("[data-dock-root] > :first-child");
        const viewport = child instanceof HTMLElement ? child : null;
        if (viewport === scene) return;
        sceneSize.disconnect();
        scenePosition.disconnect();
        scene?.removeEventListener("transitionend", syncBounds);
        scene = viewport;
        if (!scene) return;
        sceneSize.observe(scene);
        scenePosition.observe(scene, { attributes: true, attributeFilter: ["style"] });
        scene.addEventListener("transitionend", syncBounds);
        syncBounds();
    }

    new MutationObserver(bindLayout).observe(document.body, { childList: true, subtree: true });
    bindLayout();
})();
