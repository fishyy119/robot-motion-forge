/* 浏览器扩展使用的 Viser 私有接口，按项目访问的字段声明。 */
declare namespace ViserBrowser {
    type GuiValue = string | boolean;

    type GuiConfig =
        | { type: "GuiCheckboxMessage"; uuid: string; value: boolean; props: { disabled: boolean } }
        | { type: "GuiDropdownMessage"; uuid: string; value: string; props: { disabled: boolean; options: string[] } };

    interface Viewer {
        mutable: { current: ViewerState };
        useGui: { get(): { websocketState: string } };
        useGuiConfig: { get(uuid: string): unknown };
        guiActions: { updateGuiProps(uuid: string, updates: { value: GuiValue }): void };
    }

    interface ViewerState {
        canvas: HTMLCanvasElement | null;
        camera: object | null;
        nodeRefFromName: Record<string, SceneNodeRef | undefined>;
        requestRender(): void;
        sendMessage(message: { type: "GuiUpdateMessage"; uuid: string; updates: { value: GuiValue } }): void;
    }

    type SceneNodeRef = SceneObject | { current: SceneObject | null };

    interface SceneNode {
        parent: SceneObject | null;
        visible: boolean;
        __r3f?: { root: SceneStore };
        traverse(visit: (object: SceneObject) => void): void;
    }

    type SceneObject = SceneMesh | (SceneNode & { isMesh?: false; material?: SceneMaterial | SceneMaterial[] });

    interface SceneMesh extends SceneNode {
        isMesh: true;
        constructor: new (geometry: object, material: SceneMaterial) => SceneMesh;
        material: SceneMaterial | SceneMaterial[];
        geometry: object;
        renderOrder: number;
        scale: { setScalar(value: number): void };
        add(object: SceneObject): void;
        removeFromParent(): void;
        raycast(): void;
    }

    interface SceneMaterial {
        visible: boolean;
        opacity: number;
        depthTest: boolean;
        depthWrite: boolean;
        transparent: boolean;
        needsUpdate: boolean;
        toneMapped: boolean;
        color?: { setHex(value: number): void };
        emissive?: { setHex(value: number): void };
        clone(): this;
    }

    interface ScenePointer {
        clone(): ScenePointer;
        set(x: number, y: number): void;
    }

    interface SceneRaycaster {
        constructor: new () => SceneRaycaster;
        setFromCamera(pointer: ScenePointer, camera: object): void;
        intersectObjects(objects: SceneObject[], recursive: boolean): { object: SceneObject }[];
    }

    interface SceneStore {
        getState(): {
            pointer: ScenePointer;
            raycaster: SceneRaycaster;
            internal: {
                capturedMap: { size: number };
                subscribe(callback: { current: () => void }, priority: number, store: SceneStore): () => void;
            };
        };
    }

    type SceneReady = (viewer: Viewer, current: ViewerState, canvas: HTMLCanvasElement, store: SceneStore) => void;

    interface Plot {
        root: HTMLElement;
        over: HTMLElement;
        scales: { x: { min?: number | null; max?: number | null } };
        series: { class?: string }[];
        hooks: { draw?: ((chart: Plot) => void)[] };
        valToPos(value: number, scale: string): number;
    }
}

interface Window {
    __viserTestpoints?: { uplots?: Record<string, { chart: ViserBrowser.Plot }> };
}
