"""Characterization tests for load_balance_url round-robin + error paths.

This is the exact function whose module-level counters will be made
thread-safe in the optimization phase, so we lock in its observable
behavior (selection sequence + raised errors) first.

Note: load_balance_url stores its counters on the function object
(load_balance_url.counters). We reset that in setUp so each test starts
from a known counter state.
"""

import unittest
from unittest.mock import patch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import proxy_server
from proxy_server import (
    load_balance_url,
    ProxyConfig,
    SubAccountConfig,
    ServiceKey,
)


def _make_config():
    """Two subaccounts share 'shared-model'; sub-b also has a 2-URL model."""
    sub_a = SubAccountConfig(
        name="A",
        resource_group="rg-a",
        service_key_json="unused.json",
        deployment_models={"shared-model": ["https://a/deploy1"]},
    )
    sub_a.service_key = ServiceKey("id", "secret", "https://auth", "zone")
    sub_a.normalized_models = {"shared-model": ["https://a/deploy1"]}

    sub_b = SubAccountConfig(
        name="B",
        resource_group="rg-b",
        service_key_json="unused.json",
        deployment_models={
            "shared-model": ["https://b/deploy1"],
            "multi-url": ["https://b/u1", "https://b/u2"],
        },
    )
    sub_b.service_key = ServiceKey("id", "secret", "https://auth", "zone")
    sub_b.normalized_models = {
        "shared-model": ["https://b/deploy1"],
        "multi-url": ["https://b/u1", "https://b/u2"],
    }

    cfg = ProxyConfig(
        subaccounts={"A": sub_a, "B": sub_b},
        secret_authentication_tokens=["tok"],
    )
    cfg.build_model_mapping()
    return cfg


class TestLoadBalanceUrl(unittest.TestCase):
    def setUp(self):
        # Fresh counters for every test (function attribute persists globally).
        load_balance_url.counters = {}
        self._cfg = _make_config()
        self._patch = patch.object(proxy_server, "proxy_config", self._cfg)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        load_balance_url.counters = {}

    def test_round_robin_across_subaccounts(self):
        # 'shared-model' is on A and B; consecutive calls alternate subaccounts.
        seen = [load_balance_url("shared-model")[1] for _ in range(4)]
        # model mapping order is insertion order: A then B
        self.assertEqual(seen, ["A", "B", "A", "B"])

    def test_returns_expected_tuple_shape(self):
        url, sub, rg, model = load_balance_url("shared-model")
        self.assertEqual(model, "shared-model")
        self.assertIn(sub, {"A", "B"})
        self.assertTrue(url.startswith("https://"))
        self.assertIn(rg, {"rg-a", "rg-b"})

    def test_round_robin_across_urls_within_subaccount(self):
        # 'multi-url' lives only on B with two URLs -> cycles between them.
        urls = [load_balance_url("multi-url")[0] for _ in range(4)]
        self.assertEqual(
            urls,
            ["https://b/u1", "https://b/u2", "https://b/u1", "https://b/u2"],
        )

    def test_unknown_model_raises_valueerror(self):
        with self.assertRaises(ValueError):
            load_balance_url("does-not-exist")

    def test_model_with_empty_url_list_raises(self):
        # Model registered in mapping but subaccount has empty URL list.
        self._cfg.subaccounts["A"].normalized_models["shared-model"] = []
        # Force selection of subaccount A (counter 0 -> index 0 -> A).
        load_balance_url.counters = {}
        with self.assertRaises(ValueError):
            load_balance_url("shared-model")

    def test_concurrent_calls_do_not_lose_increments(self):
        # Regression guard for the thread-safety fix: N concurrent calls must
        # advance the model counter by exactly N (no lost read-modify-writes).
        import threading

        n = 200
        barrier = threading.Barrier(n)

        def worker():
            barrier.wait()  # maximize contention
            load_balance_url("shared-model")

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(load_balance_url.counters["shared-model"], n)


if __name__ == "__main__":
    unittest.main()
