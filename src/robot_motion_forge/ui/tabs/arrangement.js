/* 将编排表操作转发给 Viser，并在表格刷新后保留键盘焦点。 */
rmfViser.whenViewer(viewer => {
    let focused;
    const controlSelector = "button[data-rmf-key-action]";
    const tableSelector = "[data-rmf-arrangement]";

    document.addEventListener("click", event => {
        const control = event.target instanceof Element ? event.target.closest(controlSelector) : null;
        const table = control?.closest(tableSelector);
        if (table) rmfViser.setGuiValue(viewer, table.dataset.rmfArrangement, control.dataset.rmfKeyAction);
    });

    document.addEventListener("focusin", event => {
        const control = event.target;
        const table = control instanceof Element && control.matches(controlSelector) ? control.closest(tableSelector) : null;
        focused = table ? { control, uuid: table.dataset.rmfArrangement, action: control.dataset.rmfKeyAction } : null;
    });
    new MutationObserver(() => {
        if (!focused || focused.control.isConnected) return;
        const { uuid, action } = focused;
        focused = null;
        if (document.activeElement !== document.body) return;
        const table = document.querySelector(`[data-rmf-arrangement="${CSS.escape(uuid)}"]`);
        table?.querySelector(`button[data-rmf-key-action="${CSS.escape(action)}"]`)?.focus({ preventScroll: true });
    }).observe(document.body, { childList: true, subtree: true });
});
