"""The viewer's back end: processing session, layer images, map image and state JSON."""
import argparse
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tools import make_sample_data
from tools.run_data import add_data_arguments
from tools.viewer import LAYERS, Session, _state_json, make_handler

pytestmark = pytest.mark.slow  # renders sample data


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    data = tmp_path_factory.mktemp("viewer")
    make_sample_data.main(["--out", str(data), "--frames", "3"])
    ap = argparse.ArgumentParser()
    add_data_arguments(ap)
    ap.add_argument("--play-fps", type=float, default=0.0)
    ap.add_argument("--keep", type=int, default=2)
    return Session(ap.parse_args([str(data)]))


def test_steps_render_every_layer_and_keep_a_bounded_history(session):
    for _ in range(3):
        session.control("step")
    state = json.loads(_state_json(session))
    assert state["frame"] == 2 and state["processed"] == 3
    assert state["cached"] == [1, 2]          # --keep 2
    assert abs(state["record"]["altitude_m"] - 6.0) < 0.1
    for name in LAYERS:
        assert session.layer(2, name)[:2] == b"\xff\xd8"  # JPEG
    assert state["has_map"] and session.map_image("height")[:4] == b"\x89PNG"
    session.control("back")
    assert json.loads(_state_json(session))["frame"] == 1
    session.control("step")                   # past the end of the data
    assert json.loads(_state_json(session))["done"]


def test_http_api(session):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Perception viewer" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        frame = state["frame"]
        assert urllib.request.urlopen(f"{base}/api/frame/{frame}/height.jpg").read()[:2] == b"\xff\xd8"
        req = urllib.request.Request(base + "/api/control?action=pause", method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["playing"] is False
    finally:
        server.shutdown()
