"""操作通知与客户端归属；通知更新统一由主线程执行。"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from viser import ClientHandle, NotificationHandle, ViserServer

type NotificationColor = Literal["green", "orange", "red"]


class OperationNotification:
    def __init__(self, handles: tuple[NotificationHandle, ...], *, loading: bool) -> None:
        self._handles = handles
        self._finished = not loading

    def progress(self, body: str) -> None:
        # 求解结束后，队列中剩余的进度消息不得覆盖结果。
        if not self._finished:
            for handle in self._handles:
                handle.body = body

    def finish(self, title: str, body: str = "", *, color: NotificationColor = "green") -> None:
        if self._finished:
            return
        self._finished = True
        for handle in self._handles:
            handle.title = title
            handle.body = body
            handle.color = color
            handle.loading = False
            handle.with_close_button = True
            handle.auto_close_seconds = 3.0 if color == "green" else 8.0


class Notifications:
    def __init__(self, server: ViserServer) -> None:
        self._server = server
        self._client: ClientHandle | None = None

    @property
    def client(self) -> ClientHandle | None:
        return self._client

    @contextmanager
    def for_client(self, client: ClientHandle | None) -> Generator[None]:
        previous, self._client = self._client, client
        try:
            yield
        finally:
            self._client = previous

    def show(
        self,
        title: str,
        body: str = "",
        *,
        color: NotificationColor = "green",
        client: ClientHandle | None = None,
        loading: bool = False,
    ) -> OperationNotification:
        recipient = client if client is not None else self._client
        clients = self._server.get_clients().values() if recipient is None else (recipient,)
        handles = tuple(
            target.add_notification(
                title=title,
                body=body,
                color=None if loading else color,
                loading=loading,
                with_close_button=not loading,
                auto_close_seconds=None if loading else (3.0 if color == "green" else 8.0),
            )
            for target in clients
        )
        return OperationNotification(handles, loading=loading)
