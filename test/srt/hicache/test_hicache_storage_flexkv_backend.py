"""
E2E tests for HiCache Storage with FlexKV backend.

Reuses SGLang's official HiCacheStorageBaseMixin, only overriding the
backend and page-size configuration for FlexKV.

Usage:
    # With local model:
    SGLANG_TEST_MODEL_PATH=/path/to/model \
        python3 -m pytest test/srt/hicache/test_hicache_storage_flexkv_backend.py -v

    # With HuggingFace model (default: meta-llama/Llama-3.1-8B-Instruct):
    python3 -m pytest test/srt/hicache/test_hicache_storage_flexkv_backend.py -v

Requirements:
    - FlexKV installed: FLEXKV_DEBUG=1 pip install -e /path/to/FlexKV
    - SGLang cache_controller.py patch (flexkv in zero-copy whitelist)
"""

import json
import os
import unittest

from sglang.test.test_utils import CustomTestCase, is_in_ci

from test_hicache_storage_file_backend import (
    HiCacheStorageBaseMixin,
    run_eval_accuracy_test,
)


class HiCacheStorageFlexKVBackendBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin that switches the storage backend to FlexKV.

    Model parameters (num_layers, num_kv_heads, head_size) are auto-detected
    by the FlexKV adapter from SGLang's mem_pool_host at runtime.

    Set SGLANG_TEST_MODEL_PATH to use a local model instead of downloading.
    """

    @classmethod
    def _get_model_name(cls):
        return os.environ.get("SGLANG_TEST_MODEL_PATH", super()._get_model_name())

    @classmethod
    def _get_additional_server_args_and_env(cls):
        flexkv_config = {
            "enable_cpu": True,
            "enable_ssd": False,
            "num_cpu_blocks": 10000,
        }
        server_args = {
            "--hicache-storage-backend": "flexkv",
            "--hicache-storage-backend-extra-config": json.dumps(flexkv_config),
            "--page-size": 16,  # FlexKV C++ radix tree requires >= 2
        }
        return server_args, {}


@unittest.skipIf(is_in_ci(), "Requires FlexKV installed.")
class TestFlexKVBackendLayerFirst(
    HiCacheStorageFlexKVBackendBaseMixin, CustomTestCase
):
    """Basic backup and prefetch test with FlexKV backend (layer_first)."""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "layer_first"
        return server_args, env_vars


@unittest.skipIf(is_in_ci(), "Requires FlexKV installed.")
class TestFlexKVBackendAccuracy(
    HiCacheStorageFlexKVBackendBaseMixin, CustomTestCase
):
    """Accuracy test: GSM8K evaluation with cache persistence via FlexKV."""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-ratio"] = 1.5
        return server_args, env_vars

    def test_eval_accuracy(self):
        run_eval_accuracy_test(self)


if __name__ == "__main__":
    unittest.main(verbosity=2)
