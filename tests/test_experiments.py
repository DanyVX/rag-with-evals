from ragx.experiments import ExperimentRunner, config_hash


def test_config_hash_is_key_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_runner_caches_completed_result(tmp_path) -> None:
    runner = ExperimentRunner(tmp_path)
    calls = {"n": 0}

    def run(config):
        calls["n"] += 1
        return {"metric": config["x"]}

    first = runner.run({"x": 1}, run)
    second = runner.run({"x": 1}, run)
    assert not first.cached
    assert second.cached
    assert calls["n"] == 1
