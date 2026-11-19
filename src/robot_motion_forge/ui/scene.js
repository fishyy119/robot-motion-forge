/* 补足控制点置顶、拾取优先级、悬停提示和右键菜单。 */
(() => {
    function install(viewer, current, canvas, store) {
        const state = store.getState();
        const markers = new Map();
        function targetName(object, targets) {
            for (let node = object; node; node = node.parent) {
                const name = targets.get(node);
                if (name) return name;
            }
        }

        function visible(object) {
            for (let node = object; node; node = node.parent) {
                if (!node.visible) return false;
            }
            return true;
        }

        const isMarker = object => visible(object) && Boolean(targetName(object, markers));
        const previousFilter = state.events.filter;
        state.setEvents({ filter: (hits, state) => {
            const filtered = previousFilter ? previousFilter(hits, state) : hits;
            return filtered.sort((a, b) => Number(isMarker(b.object)) - Number(isMarker(a.object)));
        } });

        const raycaster = new state.raycaster.constructor();
        const pointerNdc = state.pointer.clone();
        function pickTarget(targets, x, y) {
            const bounds = canvas.getBoundingClientRect();
            if (x < bounds.left || x >= bounds.right || y < bounds.top || y >= bounds.bottom) return;
            pointerNdc.set((x - bounds.left) / bounds.width * 2 - 1, 1 - (y - bounds.top) / bounds.height * 2);
            raycaster.setFromCamera(pointerNdc, viewer.mutable.current.camera);
            let nearest;
            for (const { object } of raycaster.intersectObjects([...targets.keys()], true)) {
                const name = targetName(object, targets);
                const control = name === "/target";
                // 原生坐标控件用不可见的扩大拾取面接收指针，提示沿用同一拾取范围。
                if (!name || !visible(control ? object.parent : object)) continue;
                const materials = Array.isArray(object.material) ? object.material : [object.material];
                if (!control && !materials.some(material => material && material.visible && material.opacity > 0)) continue;
                if (name.startsWith("/frames/")) return name;
                if (control || !nearest) nearest = name;
            }
            return nearest;
        }

        const menu = document.createElement("div");
        menu.className = "rmf-frame-menu";
        menu.hidden = true;
        const tooltip = document.createElement("div");
        tooltip.className = "rmf-scene-tooltip";
        tooltip.hidden = true;
        document.body.append(menu, tooltip);

        function placePopup(element, x, y) {
            element.hidden = false;
            const rect = element.getBoundingClientRect();
            element.style.left = `${Math.max(8, Math.min(x, window.innerWidth - rect.width - 8))}px`;
            element.style.top = `${Math.max(8, Math.min(y, window.innerHeight - rect.height - 8))}px`;
        }

        let hoverPosition;
        const hideTooltip = () => {
            hoverPosition = undefined;
            tooltip.hidden = true;
        };
        const hover = event => {
            if (event.buttons || !menu.hidden) {
                hideTooltip();
                return;
            }
            hoverPosition = { x: event.clientX, y: event.clientY };
            viewer.mutable.current.requestRender();
        };
        canvas.addEventListener("pointermove", hover, { passive: true });
        canvas.addEventListener("pointerup", hover, { passive: true });
        for (const event of ["pointerleave", "pointerdown", "pointercancel"]) {
            canvas.addEventListener(event, hideTooltip, { passive: true });
        }

        function openMenu(name, x, y) {
            if (!rmfViser.connected(viewer)) return;
            const data = document.querySelector("[data-rmf-frame-dropdowns]")?.textContent;
            const dropdowns = data && JSON.parse(data)[name];
            if (!dropdowns) return;

            const title = document.createElement("p");
            title.className = "rmf-frame-menu-title";
            title.textContent = name.slice("/frames/".length);
            const groups = document.createElement("div");
            groups.className = "rmf-frame-menu-groups";
            for (const [label, uuid] of Object.entries(dropdowns)) {
                const dropdown = rmfViser.guiConfig(viewer, uuid);
                if (dropdown?.type !== "GuiDropdownMessage") continue;
                const group = document.createElement("div");
                const heading = document.createElement("p");
                heading.className = "rmf-frame-menu-title";
                heading.textContent = label;
                group.appendChild(heading);
                for (const value of dropdown.props.options) {
                    const button = document.createElement("button");
                    button.type = "button";
                    button.disabled = dropdown.props.disabled;
                    button.classList.toggle("selected", value === dropdown.value);
                    button.textContent = value;
                    button.addEventListener("click", () => {
                        menu.hidden = true;
                        if (!rmfViser.connected(viewer)) return;
                        const latest = rmfViser.guiConfig(viewer, uuid);
                        if (!latest || latest.props.disabled || latest.value === value) return;
                        rmfViser.setGuiValue(viewer, uuid, value, true);
                    });
                    group.appendChild(button);
                }
                groups.appendChild(group);
            }
            if (!groups.childElementCount) return;
            menu.replaceChildren(title, groups);
            hideTooltip();
            placePopup(menu, x, y);
        }

        document.addEventListener("pointerdown", event => {
            if (!menu.contains(event.target)) menu.hidden = true;
        }, true);
        canvas.addEventListener("contextmenu", event => {
            const name = pickTarget(markers, event.clientX, event.clientY);
            if (!name) return;
            event.preventDefault();
            openMenu(name, event.clientX, event.clientY);
        }, true);

        let labelsText;
        let labels = {};
        state.internal.subscribe({ current: () => {
            const refs = viewer.mutable.current.nodeRefFromName;
            markers.clear();
            // 原生组件会在切换主题时重建材质，因此在绘制前补齐置顶属性。
            for (const [name, ref] of Object.entries(refs)) {
                if (!name.startsWith("/frames/")) continue;
                const root = ref?.current ?? ref;
                if (!root?.traverse) continue;
                markers.set(root, name);
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

            if (!hoverPosition || !menu.hidden || state.internal.capturedMap.size) {
                tooltip.hidden = true;
                return;
            }
            const text = document.querySelector("[data-rmf-scene-labels]")?.textContent;
            if (text !== labelsText) {
                labelsText = text;
                labels = text ? JSON.parse(text) : {};
            }
            // 提示独立读取射线交点，不给机器人网格注册会抢占拖拽控件的点击回调。
            const targets = new Map();
            for (const [name, label] of Object.entries(labels)) {
                const ref = refs[name];
                const root = ref?.current ?? ref;
                if (label && root?.traverse) targets.set(root, name);
            }
            const { x, y } = hoverPosition;
            const name = pickTarget(targets, x, y);
            if (!name) {
                tooltip.hidden = true;
                return;
            }
            tooltip.textContent = labels[name];
            placePopup(tooltip, x + 14, y + 14);
        } }, 0, store);
        current.requestRender();
    }
    rmfViser.whenViewer(viewer => rmfViser.whenScene(viewer, install));
})();
