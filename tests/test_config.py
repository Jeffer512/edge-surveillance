import pytest

from edge_surveillance.config import AppConfig, MotionConfig, load_config


def test_defaults_match_no_file():
    assert AppConfig() == load_config(None)


def test_example_yaml_loads_with_defaults():
    cfg = load_config("config/config.example.yaml")
    assert cfg == AppConfig()
    assert cfg.motion.min_area_percent == pytest.approx(1.0)
    assert (cfg.motion.probe_width, cfg.motion.probe_height) == (320, 240)
    assert cfg.detector.input_size == (320, 320)


def test_partial_dict_fills_defaults():
    cfg = AppConfig.from_dict({"motion": {"min_change_percent": 2.0}})
    assert cfg.motion.min_change_percent == pytest.approx(2.0)
    cfg.motion.min_change_percent = 1.5
    assert cfg == AppConfig()


def test_unknown_keys_raise_type_error():
    with pytest.raises(TypeError):
        AppConfig.from_dict({"motion": {"threshhold": 1}})
    with pytest.raises(TypeError):
        AppConfig.from_dict({"bogus_section": {}})


def test_invalid_values_raise():
    with pytest.raises(ValueError):
        MotionConfig(min_change_percent=500)
    with pytest.raises(ValueError):
        MotionConfig(threshold=300)
    with pytest.raises(ValueError):
        MotionConfig(min_area_percent=0)
    with pytest.raises(ValueError):
        MotionConfig(probe_width=0, probe_height=240)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml")
