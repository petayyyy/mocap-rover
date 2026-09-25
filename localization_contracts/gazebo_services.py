"""Persistent isolated service client avoids transport13 request/callback GIL deadlock."""

import importlib
import multiprocessing
import time


def _serve(connection):
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean

    node = Node()
    time.sleep(0.5)
    connection.send(True)
    while True:
        command = connection.recv()
        if command is None:
            break
        service, module, name, data = command
        cls = getattr(importlib.import_module(module), name)
        msg = cls()
        msg.ParseFromString(data)
        ok, response = node.request(service, msg, cls, Boolean, 5000)
        connection.send((ok, bool(response.data)))
    connection.close()


class ServiceClient:
    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child,), daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(10):
            raise RuntimeError("service worker startup timed out")
        self.connection.recv()

    def request(self, service, msg):
        self.connection.send(
            (
                "/world/mocap_arena/" + service,
                type(msg).__module__,
                type(msg).__name__,
                msg.SerializeToString(),
            )
        )
        if not self.connection.poll(10):
            raise RuntimeError("service worker request timed out: " + service)
        ok, value = self.connection.recv()
        if not ok:
            raise TimeoutError(service)
        return value

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
            self.process.join(3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.connection.close()
