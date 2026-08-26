from __future__ import annotations

import logging
import hashlib
import os
from pathlib import Path
import socket
import threading
import time

import pytest

import enfold.daemon as daemon_module
from enfold.daemon import (
    DaemonAlreadyRunning,
    DaemonConfig,
    RequestError,
    SocketPathError,
    UnixJsonDaemon,
)
from enfold.protocol import (
    CAPABILITY_HEALTH,
    CAPABILITY_SEARCH,
    CAPABILITY_WRITE,
    ClientContext,
    Handshake,
    HandshakeResponse,
    ProtocolVersion,
    Request,
    Response,
    decode_frame,
    encode_frame,
)
from enfold.service import ServiceRequestError


def _context(**changes) -> ClientContext:
    values = {
        "client_id": "client-a-install-1",
        "surface": "client-a",
        "agent_id": "client-a",
        "session_id": "thread-123",
        "access_scopes": ("private",),
    }
    values.update(changes)
    return ClientContext(**values)


def _receive(client: socket.socket):
    data = bytearray()
    while b"\n" not in data:
        data.extend(client.recv(1))
    return decode_frame(bytes(data))


def _connect(
    path: Path,
    *,
    context: ClientContext | None = None,
    capabilities=(CAPABILITY_HEALTH, CAPABILITY_WRITE, CAPABILITY_SEARCH),
    protocol=ProtocolVersion(),
    schema_version=1,
    credential: str | None = None,
) -> tuple[socket.socket, HandshakeResponse]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(os.fspath(path))
    client.sendall(
        encode_frame(
            Handshake(
                context or _context(),
                capabilities=capabilities,
                protocol=protocol,
                schema_version=schema_version,
                credential=credential,
            )
        )
    )
    response = _receive(client)
    assert isinstance(response, HandshakeResponse)
    return client, response


def test_client_credentials_bind_claimed_identity_even_for_same_uid(tmp_path):
    path = tmp_path / "enfold.sock"
    token = "correct-horse-battery-staple-with-enough-entropy"
    digest = "sha256:" + hashlib.sha256(token.encode()).hexdigest()
    daemon = UnixJsonDaemon(
        _config(path, client_credentials={"privileged-install": digest}),
        lambda context, request: {},
    )
    daemon.start()
    try:
        forged, refusal = _connect(
            path,
            context=_context(client_id="privileged-install"),
            credential="attacker-controlled-token",
        )
        try:
            assert refusal.accepted is False
            assert refusal.error is not None
            assert refusal.error.code == "invalid_client_credentials"
        finally:
            forged.close()

        legitimate, accepted = _connect(
            path,
            context=_context(client_id="privileged-install"),
            credential=token,
        )
        try:
            assert accepted.accepted is True
        finally:
            legitimate.close()
    finally:
        daemon.shutdown()


def test_client_credentials_reject_one_bearer_token_shared_by_multiple_ids(tmp_path):
    digest = "sha256:" + hashlib.sha256(b"shared-token").hexdigest()

    with pytest.raises(ValueError, match="unique per client"):
        _config(
            tmp_path / "enfold.sock",
            client_credentials={"low-privilege": digest, "high-privilege": digest},
        )


def _request(path: Path, request: Request) -> Response:
    client, handshake = _connect(path)
    assert handshake.accepted
    try:
        client.sendall(encode_frame(request))
        response = _receive(client)
        assert isinstance(response, Response)
        return response
    finally:
        client.close()


def _config(path: Path, **changes) -> DaemonConfig:
    values = {
        "socket_path": path,
        "server_version": "1.0-test",
        "client_timeout": 0.5,
        "shutdown_timeout": 1.0,
    }
    values.update(changes)
    return DaemonConfig(**values)


def _overlong_socket_path(root: Path) -> Path:
    parent = root
    while len(os.fsencode(os.fspath(parent / "enfold.sock"))) <= 107:
        parent = parent / ("pad-" + "x" * 40)
    parent.mkdir(parents=True, exist_ok=True)
    return parent / "enfold.sock"


def test_bind_rejects_unix_socket_path_that_exceeds_af_unix_limit(tmp_path):
    path = _overlong_socket_path(tmp_path)
    encoded = len(os.fsencode(os.fspath(path)))
    with pytest.raises(SocketPathError, match="AF_UNIX") as caught:
        UnixJsonDaemon(_config(path), lambda context, request: {}).start()
    message = str(caught.value)
    assert str(encoded) in message
    assert "107" in message
    assert "--socket-path" in message


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_daemon_config_rejects_non_finite_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeouts must be positive"):
        _config(tmp_path / "enfold.sock", client_timeout=timeout)


def test_handshake_negotiates_capabilities_and_health_uses_trusted_context(tmp_path):
    path = tmp_path / "enfold.sock"
    seen = []
    daemon = UnixJsonDaemon(
        _config(path, server_capabilities=(CAPABILITY_HEALTH, CAPABILITY_SEARCH)),
        lambda context, request: {"unexpected": True},
        health_hook=lambda context: seen.append(context) or {"storage": "fixture"},
    )
    daemon.start()
    try:
        assert path.stat().st_mode & 0o777 == 0o600
        client, hello = _connect(
            path,
            capabilities=(CAPABILITY_WRITE, CAPABILITY_HEALTH, CAPABILITY_SEARCH),
        )
        try:
            assert hello.accepted
            assert hello.capabilities == (CAPABILITY_HEALTH, CAPABILITY_SEARCH)
            client.sendall(encode_frame(Request("health-1", "health")))
            response = _receive(client)
            assert isinstance(response, Response) and response.ok
            assert response.result == {
                "status": "ok",
                "service_version": "1.0-test",
                "protocol": {"major": 1, "minor": 0},
                "schema_version": 1,
                "storage": "fixture",
            }
            assert seen == [_context()]
        finally:
            client.close()
    finally:
        daemon.shutdown()
    assert not path.exists()


def test_missing_peer_credentials_are_refused_when_so_peercred_exists(
    tmp_path, monkeypatch
):
    if not hasattr(socket, "SO_PEERCRED"):
        pytest.skip("SO_PEERCRED is Linux-specific")
    monkeypatch.setattr(daemon_module, "inspect_peer_credentials", lambda client: None)
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect(os.fspath(path))
        assert client.recv(1) == b""
    finally:
        client.close()
        daemon.shutdown()


def test_accept_path_caps_concurrent_clients(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(
        _config(path, backlog=1, client_timeout=2.0),
        lambda context, request: {},
    )
    daemon.start()
    try:
        first, hello = _connect(path)
        try:
            assert hello.accepted
            second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            second.settimeout(1)
            try:
                second.connect(os.fspath(path))
                assert second.recv(1) == b""
            finally:
                second.close()
        finally:
            first.close()
    finally:
        daemon.shutdown()


def test_linux_peer_credentials_are_kernel_supplied_and_auditable(tmp_path):
    if not hasattr(socket, "SO_PEERCRED"):
        pytest.skip("SO_PEERCRED is Linux-specific")
    path = tmp_path / "enfold.sock"
    seen = []
    daemon = UnixJsonDaemon(
        _config(path), lambda context, request: {}, peer_hook=seen.append
    )
    daemon.start()
    try:
        client, hello = _connect(path)
        assert hello.accepted
        client.close()
        deadline = time.monotonic() + 1
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen and seen[0].pid == os.getpid()
        assert seen[0].uid == os.getuid()
        assert seen[0].gid == os.getgid()
    finally:
        daemon.shutdown()


def test_request_shutdown_is_nonblocking_signal_safe_notification(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    thread = threading.Thread(target=daemon.serve_forever)
    thread.start()
    deadline = time.monotonic() + 1
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    daemon.request_shutdown()
    thread.join(2)
    assert not thread.is_alive()
    assert not path.exists()


def test_first_frame_must_be_handshake(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    try:
        client.connect(os.fspath(path))
        client.sendall(encode_frame(Request("req-1", "health")))
        refusal = _receive(client)
        assert isinstance(refusal, HandshakeResponse)
        assert refusal.accepted is False
        assert refusal.error.code == "handshake_required"
        assert client.recv(1) == b""
    finally:
        client.close()
        daemon.shutdown()


def test_handler_receives_immutable_context_and_typed_request(tmp_path):
    path = tmp_path / "enfold.sock"
    received = []

    def handler(context, request):
        received.append((context, request))
        return {"actor": context.agent_id, "query": request.params["query"]}

    daemon = UnixJsonDaemon(_config(path), handler)
    daemon.start()
    try:
        response = _request(
            path, Request("req-1", "memory.search", {"query": "current model"})
        )
        assert response.ok
        assert response.result == {"actor": "client-a", "query": "current model"}
        assert received == [
            (
                _context(),
                Request("req-1", "memory.search", {"query": "current model"}),
            )
        ]
    finally:
        daemon.shutdown()


def test_unnegotiated_capability_is_typed_error_and_handler_is_not_called(tmp_path):
    path = tmp_path / "enfold.sock"
    called = False

    def handler(context, request):
        nonlocal called
        called = True

    daemon = UnixJsonDaemon(_config(path), handler)
    daemon.start()
    try:
        client, hello = _connect(path, capabilities=(CAPABILITY_HEALTH,))
        assert hello.accepted
        try:
            client.sendall(encode_frame(Request("req-1", "memory.write")))
            response = _receive(client)
            assert isinstance(response, Response) and not response.ok
            assert response.error.code == "capability_not_negotiated"
            assert called is False
            # Capability failure is request-local, not connection-fatal.
            client.sendall(encode_frame(Request("req-2", "health")))
            assert _receive(client).ok is True
        finally:
            client.close()
    finally:
        daemon.shutdown()


@pytest.mark.parametrize(
    ("protocol", "schema_version", "error_code"),
    [
        (ProtocolVersion(2, 0), 1, "incompatible_protocol_major"),
        (ProtocolVersion(1, 1), 1, "incompatible_protocol_minor"),
        (ProtocolVersion(), 2, "incompatible_schema"),
    ],
)
def test_incompatible_handshakes_are_explicitly_refused(
    tmp_path, protocol, schema_version, error_code
):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    try:
        client, response = _connect(
            path, protocol=protocol, schema_version=schema_version
        )
        try:
            assert response.accepted is False
            assert response.error.code == error_code
        finally:
            client.close()
    finally:
        daemon.shutdown()


def test_request_cannot_change_connection_version_or_schema(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    try:
        client, response = _connect(path)
        assert response.accepted
        try:
            changed = Request(
                "req-1", "health", protocol=ProtocolVersion(1, 1)
            )
            client.sendall(encode_frame(changed))
            error = _receive(client)
            assert isinstance(error, Response) and not error.ok
            assert error.error.code == "connection_protocol_changed"
            assert client.recv(1) == b""
        finally:
            client.close()
    finally:
        daemon.shutdown()


def test_public_and_internal_handler_errors_are_typed(tmp_path, caplog):
    path = tmp_path / "enfold.sock"

    def handler(context, request):
        if request.params.get("public"):
            raise RequestError("not_allowed", "operation is not allowed")
        raise RuntimeError("secret implementation detail")

    daemon = UnixJsonDaemon(_config(path), handler)
    daemon.start()
    try:
        public = _request(
            path, Request("req-public", "memory.write", {"public": True})
        )
        assert public.error.code == "not_allowed"
        assert public.error.message == "operation is not allowed"
        with caplog.at_level(logging.ERROR, logger="enfold.daemon"):
            internal = _request(path, Request("req-internal", "memory.write"))
        assert internal.error.code == "internal_error"
        assert "secret" not in internal.error.message
        records = [
            record for record in caplog.records
            if record.name == "enfold.daemon" and record.levelno == logging.ERROR
        ]
        assert len(records) == 1
        assert records[0].getMessage() == "request handler failed"
        assert records[0].exc_info[0] is RuntimeError
    finally:
        daemon.shutdown()


def test_internal_handler_error_is_logged_with_traceback_without_socket(caplog):
    class RecordingClient:
        def __init__(self):
            self.frames = []

        def sendall(self, frame):
            self.frames.append(frame)

    def handler(_context, _request):
        raise RuntimeError("secret implementation detail")

    daemon = UnixJsonDaemon(_config(Path("/tmp/enfold-unused.sock")), handler)
    client = RecordingClient()
    keep_open, connection = daemon._handle_frame(
        client,
        encode_frame(Handshake(_context())),
        None,
    )
    assert keep_open is True

    with caplog.at_level(logging.ERROR, logger="enfold.daemon"):
        daemon._handle_frame(
            client,
            encode_frame(Request("req-internal", "memory.write")),
            connection,
        )

    response = decode_frame(client.frames[-1])
    assert isinstance(response, Response)
    assert response.error.code == "internal_error"
    record = caplog.records[-1]
    assert record.getMessage() == "request handler failed"
    assert record.exc_info[0] is RuntimeError


def test_transport_neutral_service_errors_are_serialized(tmp_path):
    path = tmp_path / "enfold.sock"

    def handler(context, request):
        raise ServiceRequestError("invalid_params", "query syntax is invalid")

    daemon = UnixJsonDaemon(_config(path), handler)
    daemon.start()
    try:
        response = _request(path, Request("req-service", "memory.search"))
        assert response.ok is False
        assert response.error.code == "invalid_params"
        assert response.error.message == "query syntax is invalid"
    finally:
        daemon.shutdown()


def test_requests_are_concurrent_but_handler_execution_is_serialized(tmp_path):
    path = tmp_path / "enfold.sock"
    lock = threading.Lock()
    active = 0
    maximum = 0

    def handler(context, request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return request.params

    daemon = UnixJsonDaemon(_config(path), handler)
    daemon.start()
    try:
        results = []
        threads = [
            threading.Thread(
                target=lambda value=i: results.append(
                    _request(
                        path,
                        Request(f"req-{value}", "memory.write", {"n": value}),
                    )
                )
            )
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 4
        assert maximum == 1
    finally:
        daemon.shutdown()


def test_connection_can_carry_multiple_typed_requests(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(
        _config(path), lambda context, request: {"seen": request.params["n"]}
    )
    daemon.start()
    try:
        client, hello = _connect(path)
        assert hello.accepted
        try:
            client.sendall(
                encode_frame(Request("req-1", "memory.write", {"n": 1}))
                + encode_frame(Request("req-2", "memory.write", {"n": 2}))
            )
            assert _receive(client).result["seen"] == 1
            assert _receive(client).result["seen"] == 2
        finally:
            client.close()
    finally:
        daemon.shutdown()


def test_oversize_pre_handshake_frame_gets_typed_refusal(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(
        _config(path, max_frame_bytes=512), lambda context, request: None
    )
    daemon.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2)
    try:
        client.connect(os.fspath(path))
        client.sendall(b"{" + b"x" * 600 + b"\n")
        response = _receive(client)
        assert isinstance(response, HandshakeResponse)
        assert response.error.code == "frame_too_large"
    finally:
        client.close()
        daemon.shutdown()


def test_stale_socket_requires_explicit_cleanup(tmp_path):
    path = tmp_path / "enfold.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(os.fspath(path))
    stale.close()

    with pytest.raises(SocketPathError, match="explicit cleanup"):
        UnixJsonDaemon(_config(path), lambda context, request: {}).start()
    assert path.exists()

    daemon = UnixJsonDaemon(
        _config(path, cleanup_stale_socket=True), lambda context, request: {}
    )
    daemon.start()
    daemon.shutdown()
    assert not path.exists()


def test_live_socket_and_regular_file_are_never_removed(tmp_path):
    path = tmp_path / "live.sock"
    first = UnixJsonDaemon(_config(path), lambda context, request: {})
    first.start()
    try:
        with pytest.raises(DaemonAlreadyRunning):
            UnixJsonDaemon(
                _config(path, cleanup_stale_socket=True),
                lambda context, request: {},
            ).start()
        assert path.exists()
    finally:
        first.shutdown()

    regular = tmp_path / "regular.sock"
    regular.write_text("do not delete")
    with pytest.raises(SocketPathError, match="not a Unix socket"):
        UnixJsonDaemon(
            _config(regular, cleanup_stale_socket=True),
            lambda context, request: {},
        ).start()
    assert regular.read_text() == "do not delete"


def test_handler_deadline_fails_the_request_instead_of_running_past_budget(tmp_path):
    path = tmp_path / "enfold.sock"
    started = threading.Event()

    def handler(context, request):
        started.set()
        time.sleep(0.4)
        return {"late": True}

    daemon = UnixJsonDaemon(
        _config(path, client_timeout=2.0, handler_timeout=0.05),
        handler,
    )
    daemon.start()
    try:
        response = _request(path, Request("slow-1", "memory.write", {}))
        assert response.ok is False
        assert response.error is not None
        assert response.error.code == "request_timeout"
        assert started.wait(1.0)
    finally:
        daemon.shutdown()


def test_idle_client_timeout_is_typed_and_shutdown_is_graceful(tmp_path):
    path = tmp_path / "enfold.sock"
    daemon = UnixJsonDaemon(
        _config(path, client_timeout=0.05), lambda context, request: {}
    )
    daemon.start()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect(os.fspath(path))
        response = _receive(client)
        assert isinstance(response, HandshakeResponse)
        assert response.error.code == "request_timeout"
    finally:
        client.close()
        daemon.shutdown()
    assert not path.exists()


def test_socket_liveness_reports_absent_live_and_stale(tmp_path):
    path = tmp_path / "enfold.sock"
    assert daemon_module.socket_liveness(path) == "absent"

    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    try:
        assert daemon_module.socket_liveness(path) == "live"
    finally:
        daemon.shutdown()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(path))
    listener.close()
    assert path.exists()
    assert daemon_module.socket_liveness(path) == "stale"
    assert daemon_module.remove_stale_socket(path) is True
    assert not path.exists()
    assert daemon_module.socket_liveness(path) == "absent"


def test_socket_liveness_reports_absent_live_and_stale(tmp_path):
    path = tmp_path / "enfold.sock"
    assert daemon_module.socket_liveness(path) == "absent"

    daemon = UnixJsonDaemon(_config(path), lambda context, request: {})
    daemon.start()
    try:
        assert daemon_module.socket_liveness(path) == "live"
    finally:
        daemon.shutdown()

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(path))
    listener.close()
    assert path.exists()
    assert daemon_module.socket_liveness(path) == "stale"
    assert daemon_module.remove_stale_socket(path) is True
    assert not path.exists()
    assert daemon_module.socket_liveness(path) == "absent"
