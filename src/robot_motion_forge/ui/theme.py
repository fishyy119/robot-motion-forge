"""原生 Viser 主题，以及 Python API 未提供配置入口的排版与文字对比度。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from viser import GuiApi


def accent_color(dark_mode: bool) -> tuple[int, int, int]:
    return (45, 212, 191) if dark_mode else (15, 118, 110)


def warning_color(dark_mode: bool) -> tuple[int, int, int]:
    return (255, 183, 77) if dark_mode else (180, 83, 9)


def robot_color(dark_mode: bool) -> tuple[int, int, int]:
    return (138, 145, 155) if dark_mode else (160, 164, 171)


def locked_color(dark_mode: bool) -> tuple[int, int, int]:
    return (96, 165, 250) if dark_mode else (30, 58, 138)


def apply_theme(gui: GuiApi, *, dark_mode: bool) -> None:
    gui.configure_theme(
        dark_mode=dark_mode,
        control_width="large",
        show_logo=False,
        show_share_button=False,
        brand_color=accent_color(dark_mode),
    )


THEME_STYLE = """
<style>
:root {
    /* 抵消 Viser 的 92.5% 缩放，保留浏览器默认根字号与用户的字号偏好。 */
    font-size: 100%;
    --rmf-font-family: "Inter", "Noto Sans SC", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    --rmf-font-control: 0.875rem;
    --rmf-font-title: 1rem;
    --rmf-font-caption: 0.75rem;
    --rmf-space-control: 0.75rem;
    --rmf-space-group: 1.25rem;
    --mantine-font-family: var(--rmf-font-family);
    --mantine-font-family-headings: var(--rmf-font-family);
    --mantine-spacing-xs: 0.5rem;
}
/* 原生标签使用 dimmed；仅修正文字对比度，不重定义控件的交互状态和背景。 */
:root[data-mantine-color-scheme="light"] {
    --mantine-color-dimmed: #475569;
}
:root[data-mantine-color-scheme="dark"] {
    --mantine-color-dimmed: #b8c1cc;
}
body {
    font-family: var(--rmf-font-family);
    font-size: var(--rmf-font-control);
}
/* 原生内联样式包含字号和负字距；仅在这些排版属性上覆盖。 */
.mantine-Text-root:has(> label),
.mantine-Input-input,
.mantine-Combobox-option,
.mantine-Button-root,
.mantine-Button-label,
[role="tablist"],
[role="tab"],
[data-dock-unmergeable-header] {
    font-size: var(--rmf-font-control) !important;
    letter-spacing: normal !important;
}
.mantine-Paper-root > .mantine-Paper-root {
    font-size: var(--rmf-font-title);
}
.mantine-Slider-markLabel {
    font-size: var(--rmf-font-caption) !important;
    color: var(--mantine-color-dimmed);
}
/* 统一紧凑控件的高度；按钮组仍由 Viser 自行换行和分配宽度。 */
.mantine-Input-input,
.mantine-Button-root {
    min-height: 1.75rem !important;
}
/* 控件包装层统一行距，按钮组内部的排列仍由 Viser 管理。 */
div:has(> .mantine-Flex-root > div > .mantine-Text-root > label),
div:not(.mantine-Flex-root):has(> .mantine-Button-root[id]) {
    padding-bottom: var(--rmf-space-control) !important;
}
/* 固定标签栏宽度，不随分组层级缩窄；长关节名优先保留在同一行。 */
.mantine-Flex-root:has(> div > .mantine-Text-root > label) {
    min-height: 1.75rem;
}
.mantine-Flex-root > div:has(> .mantine-Text-root > label) {
    width: 7.5rem !important;
}
.mantine-Flex-root > div:has(> .mantine-Text-root > label) + div {
    min-width: 0;
}
/* 侧栏变窄时，将滑块或三轴输入整体换到下一行，保留名称与数值。 */
.mantine-Flex-root:has(> div > .mantine-Text-root > label):has(
    .mantine-Slider-root, .mantine-Flex-root > .mantine-NumberInput-root:nth-child(3)
) {
    flex-wrap: wrap;
    row-gap: 0.25rem;
}
.mantine-Flex-root:has(.mantine-Slider-root) > div:has(> .mantine-Text-root > label) {
    width: 10.5rem !important;
}
.mantine-Flex-root:has(> div > .mantine-Text-root > label):has(.mantine-Slider-root) {
    min-height: 2rem;
}
/* 滑块附带的数值框留足小数与负号的宽度，避免被滑条挤压。 */
.mantine-Slider-root + .mantine-NumberInput-root {
    width: 6rem !important;
    flex-shrink: 0;
}
.mantine-Slider-root:has(.mantine-Slider-markLabel) {
    /* 刻度文字为绝对定位，须单独为其留出高度；换行后也不能侵入下一控件。 */
    padding-bottom: 1.125rem !important;
}
.mantine-Flex-root:has(> div > .mantine-Text-root > label):has(.mantine-Slider-root) > div:last-child {
    flex-basis: 10rem;
}
.mantine-Flex-root > div:has(> .mantine-Flex-root > .mantine-NumberInput-root:nth-child(3)) {
    flex-basis: 15rem;
}
/* 分组标题浮在上边框，外侧间距和内侧留白分别为它预留空间。 */
.mantine-Paper-root:has(> .mantine-Paper-root + [aria-hidden]) {
    margin-top: var(--rmf-space-group) !important;
    margin-bottom: var(--rmf-space-group) !important;
    padding-bottom: 0.5rem;
}
.mantine-Paper-root:has(> .mantine-Paper-root + [aria-hidden]):last-child {
    margin-bottom: 0 !important;
}
/* 只调整内容层内边距，保留 Viser 对折叠高度、动画与显隐的管理。 */
.mantine-Paper-root > .mantine-Paper-root + [aria-hidden] > div {
    padding-top: 1rem !important;
}
/* 固定工程目录仅展示文本，长路径换行并允许选中复制。 */
.rmf-project-directory {
    margin: 0;
    padding: 0.5rem 0.75rem var(--rmf-space-control);
    font-size: var(--rmf-font-control);
    line-height: 1.5;
}
.rmf-project-directory dt {
    margin-bottom: 0.25rem;
    color: var(--mantine-color-dimmed);
}
.rmf-project-directory dd {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    user-select: text;
}
/* 场景提示只展示对象名称；不覆盖指针事件，也不固定占用场景空间。 */
.rmf-scene-tooltip {
    position: fixed;
    z-index: 2147483647;
    pointer-events: none;
    max-width: min(28rem, calc(100vw - 1rem));
    padding: 0.375rem 0.5rem;
    border: 1px solid var(--mantine-color-default-border);
    border-radius: 0.25rem;
    background: var(--mantine-color-body);
    color: var(--mantine-color-text);
    box-shadow: 0 2px 8px #0003;
    font-size: var(--rmf-font-caption);
    white-space: pre-line;
    overflow-wrap: anywhere;
}
</style>
"""
