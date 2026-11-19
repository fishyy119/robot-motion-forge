/* 将时间轴刻度对应到工程关键帧。 */
rmfViser.whenViewer(viewer => {
    let boundSlider;
    function refresh() {
        const config = document.querySelector("[data-rmf-timeline]");
        if (!config) return;
        const slider = document.getElementById(config.dataset.rmfTimeline);
        if (!slider || slider === boundSlider) return;
        const keyIds = JSON.parse(config.textContent);
        const labels = rmfViser.timelineLabels(slider);
        if (labels.length !== keyIds.length) return;
        boundSlider = slider;
        const uuid = config.dataset.markClick;

        for (const [index, label] of labels.entries()) {
            label.dataset.rmfTimelineMark = "";
            label.setAttribute("role", "button");
            label.tabIndex = 0;
            label.title = label.textContent;
            // Mantine 在轨道上监听鼠标和触摸起点；标签将手势留给自身的点击。
            for (const type of ["mousedown", "touchstart"]) {
                label.addEventListener(type, event => event.stopPropagation());
            }
            label.addEventListener("click", () => {
                rmfViser.setGuiValue(viewer, uuid, keyIds[index]);
            });
            label.addEventListener("keydown", event => {
                if (event.key !== "Enter" && event.key !== " ") return;
                event.preventDefault();
                event.stopPropagation();
                label.click();
            });
        }
    }

    new MutationObserver(refresh).observe(document.body, { childList: true, subtree: true });
    refresh();
});
