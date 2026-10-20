from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(seq=None, temp=None, ph=None):
    at = AppTest.from_file(APP, default_timeout=60).run()
    if seq is not None:
        at.sidebar.text_input[0].set_value(seq)
    if temp is not None:
        at.sidebar.slider[0].set_value(temp)
    if ph is not None:
        at.sidebar.slider[1].set_value(ph)
    at.sidebar.button[0].click().run()
    return at


def _diagnoses(at):
    return len(at.error) + len(at.warning) + len(at.success) + len(at.info)


def test_defaults_render():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.sidebar.text_input[0].value == "HKHPHDHPKHHDH"
    assert [s.value for s in at.sidebar.slider] == [1.2, 7.0, 300]
    assert at.sidebar.button[0].label == "Run 3D Simulation"


def test_run_shows_diagnosis_and_charge():
    at = _run()
    assert not at.exception
    assert _diagnoses(at) == 1
    assert "Net charge at pH 7.0" in at.caption[0].value


@pytest.mark.parametrize("ph", [0.0, 14.0])
def test_extreme_ph_runs(ph):
    at = _run(ph=ph)
    assert not at.exception
    assert _diagnoses(at) == 1


def test_invalid_sequence_shows_error():
    at = _run(seq="HHXP")
    assert not at.exception
    assert "only H, P, K and D" in at.sidebar.error[0].value


def test_no_core_sequence_shows_info():
    at = _run(seq="PPPPP")
    assert not at.exception
    assert len(at.info) == 1
