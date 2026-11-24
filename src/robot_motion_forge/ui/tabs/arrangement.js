/* 将编排表操作转发给 Viser，并在表格刷新后保留键盘焦点。 */
rmfViser.whenViewer(viewer => {
    const controlSelector = "button[data-rmf-key-action]";
    const tableSelector = "[data-rmf-arrangement]";
    /** @type {ReturnType<typeof tableAction>} */
    let focused;

    /** @param {Element | null} control */
    function tableAction(control) {
        const table = control?.closest(tableSelector);
        if (!(control instanceof HTMLButtonElement) || !(table instanceof HTMLElement)) return;
        const uuid = table.dataset.rmfArrangement;
        const action = control.dataset.rmfKeyAction;
        if (uuid !== undefined && action !== undefined) return { control, uuid, action };
    }

    document.addEventListener("click", event => {
        const control = event.target instanceof Element ? event.target.closest(controlSelector) : null;
        const target = tableAction(control);
        if (target) rmfViser.setGuiValue(viewer, target.uuid, target.action);
    });

    document.addEventListener("focusin", event => {
        const control = event.target;
        focused = control instanceof Element && control.matches(controlSelector) ? tableAction(control) : undefined;
    });
    new MutationObserver(() => {
        if (!focused || focused.control.isConnected) return;
        const { uuid, action } = focused;
        focused = undefined;
        if (document.activeElement !== document.body) return;
        const table = document.querySelector(`[data-rmf-arrangement="${CSS.escape(uuid)}"]`);
        const button = table?.querySelector(`button[data-rmf-key-action="${CSS.escape(action)}"]`);
        if (button instanceof HTMLButtonElement) button.focus({ preventScroll: true });
    }).observe(document.body, { childList: true, subtree: true });
});
