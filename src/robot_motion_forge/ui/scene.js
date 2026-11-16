/* 补足 Viser Python API 未公开的场景显示选项；模型与拖拽仍由原生组件管理。 */
(() => {
    document.title = "Robot Motion Forge";
    const key = Symbol.for("robot-motion-forge.scene");
    const settings = window[key] ??= { installed: new WeakSet(), labels: {} };

    function install() {
        const element = document.querySelector("canvas");
        const fiberKey = element && Object.keys(element).find(key => key.startsWith("__reactFiber"));
        let viewer;
        for (let fiber = fiberKey && element[fiberKey]; fiber; fiber = fiber.return) {
            const value = fiber.memoizedProps?.value;
            if (value?.mutable?.current?.nodeRefFromName) viewer = value;
        }
        const current = viewer?.mutable.current;
        const ref = current && Object.values(current.nodeRefFromName).find(ref => (ref?.current ?? ref)?.__r3f?.root);
        const store = (ref?.current ?? ref)?.__r3f?.root;
        if (!store) {
            requestAnimationFrame(install);
            return;
        }
        if (settings.installed.has(store)) return;
        settings.installed.add(store);
        const canvas = current.canvas;
        const state = store.getState();
        const roots = new WeakSet();
        const isMarker = object => {
            for (let node = object; node; node = node.parent) {
                if (roots.has(node)) return true;
            }
            return false;
        };
        const previousFilter = state.events.filter;
        state.setEvents({ filter: (hits, state) => {
            const filtered = previousFilter ? previousFilter(hits, state) : hits;
            return filtered.sort((a, b) => Number(isMarker(b.object)) - Number(isMarker(a.object)));
        } });

        // 提示独立读取射线交点，不给机器人网格注册会抢占拖拽控件的点击回调。
        const tooltip = document.createElement("div");
        tooltip.className = "rmf-scene-tooltip";
        tooltip.setAttribute("role", "tooltip");
        tooltip.hidden = true;
        document.body.appendChild(tooltip);
        const raycaster = new state.raycaster.constructor();
        const pointerNdc = state.pointer.clone();
        let labelsText;
        let pointer = null;
        const hide = () => {
            pointer = null;
            tooltip.hidden = true;
        };
        const move = event => {
            if (event.buttons) {
                hide();
                return;
            }
            pointer = { x: event.clientX, y: event.clientY };
            viewer.mutable.current.requestRender();
        };
        canvas.addEventListener("pointermove", move, { passive: true });
        canvas.addEventListener("pointerup", move, { passive: true });
        for (const event of ["pointerleave", "pointerdown", "pointercancel"]) {
            canvas.addEventListener(event, hide, { passive: true });
        }
        window.addEventListener("blur", hide);
        window.addEventListener("resize", hide);

        const visible = (object, control) => {
            // 原生坐标控件用不可见的扩大拾取面接收指针，提示沿用同一拾取范围。
            for (let node = control ? object.parent : object; node; node = node.parent) {
                if (!node.visible) return false;
            }
            if (control) return true;
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            return materials.some(material => material && material.visible && material.opacity > 0);
        };

        state.internal.subscribe({ current: () => {
            const m = viewer.mutable.current;
            const text = document.querySelector("[data-rmf-scene-labels]")?.textContent;
            if (text !== labelsText) {
                labelsText = text;
                settings.labels = text ? JSON.parse(text) : {};
            }
            // 原生组件会在切换主题时重建材质，因此在绘制前补齐置顶属性。
            for (const [name, ref] of Object.entries(m.nodeRefFromName)) {
                if (!name.startsWith("/end_effectors/")) continue;
                const root = ref?.current ?? ref;
                if (!root?.traverse) continue;
                roots.add(root);
                root.traverse(object => {
                    if (!object.isMesh) return;
                    object.renderOrder = 10000;
                    const materials = Array.isArray(object.material) ? object.material : [object.material];
                    for (const material of materials) {
                        if (!material.depthTest && !material.depthWrite && material.transparent) continue;
                        material.depthTest = material.depthWrite = false;
                        material.transparent = true;
                        material.needsUpdate = true;
                    }
                });
            }

            if (!pointer || state.internal.capturedMap.size) {
                tooltip.hidden = true;
                return;
            }
            const bounds = canvas.getBoundingClientRect();
            pointerNdc.set(
                (pointer.x - bounds.x) / bounds.width * 2 - 1,
                1 - (pointer.y - bounds.y) / bounds.height * 2,
            );
            raycaster.setFromCamera(pointerNdc, m.camera);
            const targets = new Map();
            for (const [name, label] of Object.entries(settings.labels)) {
                const ref = m.nodeRefFromName[name];
                const root = ref?.current ?? ref;
                if (label && root?.traverse) {
                    targets.set(root, { label, priority: name.startsWith("/end_effectors/") ? 0 : name === "/target" ? 1 : 2 });
                }
            }
            let selected = null;
            for (const hit of raycaster.intersectObjects([...targets.keys()], true)) {
                let target;
                for (let node = hit.object; node && !target; node = node.parent) target = targets.get(node);
                if (!target || !visible(hit.object, target.priority === 1)) continue;
                if (!selected || target.priority < selected.priority) selected = target;
                if (selected?.priority === 0) break;
            }
            if (!selected) {
                tooltip.hidden = true;
                return;
            }
            tooltip.textContent = selected.label;
            tooltip.hidden = false;
            const rect = tooltip.getBoundingClientRect();
            tooltip.style.left = `${Math.max(8, Math.min(pointer.x + 14, window.innerWidth - rect.width - 8))}px`;
            tooltip.style.top = `${Math.max(8, Math.min(pointer.y + 14, window.innerHeight - rect.height - 8))}px`;
        } }, 0, store);
        current.requestRender();
    }
    install();
})();
