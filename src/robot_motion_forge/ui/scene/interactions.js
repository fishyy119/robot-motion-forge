/* 补足控制点置顶、拾取优先级、悬停高亮与提示和右键菜单。 */
(() => {
    /**
     * @typedef {{
     *     frames: Record<string, { name: string, dropdowns: Record<string, string> } | undefined>,
     *     joints: Record<string, string | undefined>,
     *     bodies: Record<string, string[] | undefined>,
     *     gizmo: string,
     * }} SceneMenuConfig
     * @typedef {Map<ViserBrowser.SceneObject, string>} SceneTargets
     */

    /** @type {ViserBrowser.SceneReady} */
    const install = (viewer, current, canvas, store) => {
        const state = store.getState();
        /** @type {string | null | undefined} */
        let menuText;
        /** @type {SceneMenuConfig | undefined} */
        let menuConfig;
        function readMenuConfig() {
            const text = document.querySelector("[data-rmf-scene-menu]")?.textContent;
            if (text !== menuText) {
                menuText = text;
                menuConfig = text ? JSON.parse(text) : undefined;
            }
            return menuConfig;
        }

        /**
         * @param {ViserBrowser.SceneObject | null} object
         * @param {SceneTargets} targets
         */
        function targetName(object, targets) {
            for (let node = object; node; node = node.parent) {
                const name = targets.get(node);
                if (name) return name;
            }
        }

        /** @param {ViserBrowser.SceneObject | null} object */
        function visible(object) {
            for (let node = object; node; node = node.parent) {
                if (!node.visible) return false;
            }
            return true;
        }

        const raycaster = new state.raycaster.constructor();
        const pointerNdc = state.pointer.clone();
        /** @param {string[]} names */
        function sceneTargets(names) {
            const refs = viewer.mutable.current.nodeRefFromName;
            /** @type {SceneTargets} */
            const targets = new Map();
            for (const name of names) {
                const root = rmfViser.sceneObject(refs[name]);
                if (root?.traverse) targets.set(root, name);
            }
            return targets;
        }

        /**
         * @param {SceneTargets} targets
         * @param {number} x
         * @param {number} y
         * @param {SceneMenuConfig} config
         */
        function pickTarget(targets, x, y, config) {
            const bounds = canvas.getBoundingClientRect();
            if (x < bounds.left || x >= bounds.right || y < bounds.top || y >= bounds.bottom) return;
            const camera = viewer.mutable.current.camera;
            if (!camera) return;
            pointerNdc.set((x - bounds.left) / bounds.width * 2 - 1, 1 - (y - bounds.top) / bounds.height * 2);
            raycaster.setFromCamera(pointerNdc, camera);
            /** @type {{ name: string, object: ViserBrowser.SceneObject } | undefined} */
            let nearest;
            for (const { object } of raycaster.intersectObjects([...targets.keys()], true)) {
                const name = targetName(object, targets);
                const control = name === config.gizmo;
                // 原生坐标控件用不可见的扩大拾取面接收指针，提示沿用同一拾取范围。
                if (!name || !visible(control ? object.parent : object)) continue;
                const materials = Array.isArray(object.material) ? object.material : [object.material];
                if (!control && !materials.some(material => material && material.visible && material.opacity > 0)) continue;
                if (Object.hasOwn(config.frames, name)) return { name, object };
                if (control || !nearest) nearest = { name, object };
            }
            return nearest;
        }

        /** @type {ViserBrowser.SceneMesh | undefined} */
        let outline;
        /** @param {ViserBrowser.SceneObject} [marker] */
        function highlightMarker(marker) {
            if (outline?.parent === marker) return;
            outline?.removeFromParent();
            if (!marker?.isMesh || Array.isArray(marker.material)) return;
            if (!outline) {
                // 先绘制扩大的轮廓，再由控制点自身覆盖中间区域，保留固定状态与主题颜色。
                const material = marker.material.clone();
                material.color?.setHex(0x000000);
                material.emissive?.setHex(0xfbff00);
                material.toneMapped = false;
                material.depthTest = material.depthWrite = false;
                material.transparent = true;
                material.opacity = 0.8;
                outline = new marker.constructor(marker.geometry, material);
                outline.renderOrder = 9999;
                outline.scale.setScalar(1.35);
                outline.raycast = () => {};
            }
            outline.geometry = marker.geometry;
            marker.add(outline);
        }

        const menu = document.createElement("div");
        menu.className = "rmf-scene-menu";
        menu.hidden = true;
        const tooltip = document.createElement("div");
        tooltip.className = "rmf-scene-tooltip";
        tooltip.hidden = true;
        document.body.append(menu, tooltip);

        /**
         * @param {HTMLElement} element
         * @param {number} x
         * @param {number} y
         */
        function placePopup(element, x, y) {
            element.hidden = false;
            const rect = element.getBoundingClientRect();
            element.style.left = `${Math.max(8, Math.min(x, window.innerWidth - rect.width - 8))}px`;
            element.style.top = `${Math.max(8, Math.min(y, window.innerHeight - rect.height - 8))}px`;
        }

        /** @type {{ x: number, y: number } | undefined} */
        let hoverPosition;
        const clearHover = () => {
            hoverPosition = undefined;
            tooltip.hidden = true;
            if (outline?.parent) {
                highlightMarker();
                current.requestRender();
            }
        };
        /** @param {PointerEvent} event */
        const hover = event => {
            if (event.buttons || !menu.hidden) {
                clearHover();
                return;
            }
            hoverPosition = { x: event.clientX, y: event.clientY };
            viewer.mutable.current.requestRender();
        };
        canvas.addEventListener("pointermove", hover, { passive: true });
        canvas.addEventListener("pointerup", hover, { passive: true });
        for (const event of ["pointerleave", "pointerdown", "pointercancel"]) {
            canvas.addEventListener(event, clearHover, { passive: true });
        }
        window.addEventListener("blur", clearHover);

        /**
         * @param {ViserBrowser.GuiConfig} config
         * @param {string} text
         */
        function menuButton(config, text) {
            const toggle = config.type === "GuiCheckboxMessage";
            const selected = toggle ? config.value : config.value === text;
            const button = document.createElement("button");
            button.type = "button";
            button.disabled = config.props.disabled;
            button.classList.toggle("selected", selected);
            button.textContent = text;
            if (toggle) button.setAttribute("aria-pressed", String(selected));
            button.addEventListener("click", () => {
                menu.hidden = true;
                if (!rmfViser.connected(viewer)) return;
                const latest = rmfViser.guiConfig(viewer, config.uuid);
                if (!latest || latest.props.disabled) return;
                const value = toggle ? !latest.value : text;
                if (value !== latest.value) rmfViser.setGuiValue(viewer, config.uuid, value, true);
            });
            return button;
        }

        /** @param {Record<string, string>} dropdowns */
        function frameMenu(dropdowns) {
            const groups = document.createElement("div");
            groups.className = "rmf-frame-menu-groups";
            for (const [label, uuid] of Object.entries(dropdowns)) {
                const dropdown = rmfViser.guiConfig(viewer, uuid);
                if (dropdown?.type !== "GuiDropdownMessage") continue;
                const group = document.createElement("div");
                const heading = document.createElement("p");
                heading.className = "rmf-scene-menu-title";
                heading.textContent = label;
                group.appendChild(heading);
                for (const value of dropdown.props.options) {
                    group.appendChild(menuButton(dropdown, value));
                }
                groups.appendChild(group);
            }
            return groups;
        }

        /**
         * @param {string[]} names
         * @param {SceneMenuConfig["joints"]} checkboxes
         */
        function jointMenu(names, checkboxes) {
            const items = document.createElement("div");
            items.className = "rmf-joint-menu";
            for (const name of names) {
                const uuid = checkboxes[name];
                if (uuid === undefined) continue;
                const config = rmfViser.guiConfig(viewer, uuid);
                if (config?.type !== "GuiCheckboxMessage") continue;
                items.appendChild(menuButton(config, names.length === 1 ? "锁定关节" : name));
            }
            return items;
        }

        document.addEventListener("pointerdown", event => {
            if (!(event.target instanceof Node) || !menu.contains(event.target)) menu.hidden = true;
        }, true);
        canvas.addEventListener("contextmenu", event => {
            if (!rmfViser.connected(viewer)) return;
            const config = readMenuConfig();
            if (!config) return;
            const { frames, joints, bodies } = config;
            const targets = sceneTargets([...Object.keys(frames), ...Object.keys(bodies)]);
            const hit = pickTarget(targets, event.clientX, event.clientY, config);
            if (!hit) return;
            const { name } = hit;
            const frame = frames[name];
            const jointNames = bodies[name] ?? [];
            const items = frame ? frameMenu(frame.dropdowns) : jointMenu(jointNames, joints);
            if (!items.childElementCount) return;

            event.preventDefault();
            const title = document.createElement("p");
            title.className = "rmf-scene-menu-title";
            title.textContent = frame ? frame.name : jointNames.length === 1
                ? jointNames[0] : "锁定关节";
            menu.replaceChildren(title, items);
            clearHover();
            placePopup(menu, event.clientX, event.clientY);
        }, true);

        /** @type {string | null | undefined} */
        let labelsText;
        /** @type {Record<string, string | undefined>} */
        let labels = {};
        state.internal.subscribe({ current: () => {
            const refs = viewer.mutable.current.nodeRefFromName;
            const config = readMenuConfig();
            // 原生组件会在切换主题时重建材质，因此在绘制前补齐置顶属性。
            for (const name of Object.keys(config?.frames ?? {})) {
                const root = rmfViser.sceneObject(refs[name]);
                if (!root?.traverse) continue;
                root.traverse(object => {
                    if (!object.isMesh || object === outline) return;
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

            if (!config || !hoverPosition || !menu.hidden || state.internal.capturedMap.size) {
                tooltip.hidden = true;
                highlightMarker();
                return;
            }
            const text = document.querySelector("[data-rmf-scene-labels]")?.textContent;
            if (text !== labelsText) {
                labelsText = text;
                labels = text ? JSON.parse(text) : {};
            }
            // 提示独立读取射线交点，不给机器人网格注册会抢占拖拽控件的点击回调。
            const targets = sceneTargets(Object.keys(labels).filter(name => labels[name]));
            const { x, y } = hoverPosition;
            const hit = pickTarget(targets, x, y, config);
            highlightMarker(hit && Object.hasOwn(config.frames, hit.name) ? hit.object : undefined);
            if (!hit) {
                tooltip.hidden = true;
                return;
            }
            tooltip.textContent = labels[hit.name] ?? "";
            placePopup(tooltip, x + 14, y + 14);
        } }, 0, store);
        current.requestRender();
    };
    rmfViser.whenViewer(viewer => rmfViser.whenScene(viewer, install));
})();
