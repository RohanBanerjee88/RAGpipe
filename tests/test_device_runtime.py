"""Regression tests for dependency profiles and GPU device selection."""

import unittest
import warnings
from unittest.mock import patch

import device_runtime
from device_runtime import GPUUnavailableError, cuda_architecture_supported
from scripts import check_environment
from scripts.check_environment import PROJECT_ROOT, load_expected_versions


class ArchitectureCompatibilityTests(unittest.TestCase):
    def test_cuda_126_architectures_cover_cluster_gpu_generations(self):
        cuda_126_arches = ("sm_70", "sm_80", "sm_86", "sm_90", "compute_90")

        self.assertTrue(cuda_architecture_supported((7, 0), cuda_126_arches))  # V100
        self.assertTrue(cuda_architecture_supported((8, 0), cuda_126_arches))  # A100
        self.assertTrue(cuda_architecture_supported((9, 0), cuda_126_arches))  # H100

    def test_cuda_130_style_architectures_reject_v100(self):
        cuda_130_arches = ("sm_75", "sm_80", "sm_86", "sm_90", "compute_90")

        self.assertFalse(cuda_architecture_supported((7, 0), cuda_130_arches))
        self.assertTrue(cuda_architecture_supported((8, 0), cuda_130_arches))
        self.assertTrue(cuda_architecture_supported((9, 0), cuda_130_arches))

    def test_ptx_target_can_support_a_newer_device(self):
        self.assertTrue(cuda_architecture_supported((10, 0), ("compute_90",)))


class DeviceSelectionTests(unittest.TestCase):
    def tearDown(self):
        device_runtime.clear_device_selection_cache()

    @patch.object(device_runtime, "_probe_cuda_subprocess", return_value=(True, "passed"))
    @patch.object(
        device_runtime.torch.cuda, "get_arch_list", return_value=["sm_70", "sm_80", "sm_90"]
    )
    @patch.object(device_runtime.torch.cuda, "get_device_capability", return_value=(7, 0))
    @patch.object(device_runtime.torch.cuda, "get_device_name", return_value="Tesla V100")
    @patch.object(device_runtime.torch.cuda, "is_available", return_value=True)
    def test_supported_gpu_is_selected(self, *_mocks):
        selection = device_runtime.select_runtime_device(request="cuda")

        self.assertEqual(selection.device, "cuda")
        self.assertEqual(selection.capability, (7, 0))
        self.assertEqual(selection.gpu_name, "Tesla V100")

    @patch.object(
        device_runtime.torch.cuda, "get_arch_list", return_value=["sm_75", "sm_80", "sm_90"]
    )
    @patch.object(device_runtime.torch.cuda, "get_device_capability", return_value=(7, 0))
    @patch.object(device_runtime.torch.cuda, "get_device_name", return_value="Tesla V100")
    @patch.object(device_runtime.torch.cuda, "is_available", return_value=True)
    def test_auto_mode_falls_back_when_wheel_omits_gpu_architecture(self, *_mocks):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selection = device_runtime.select_runtime_device(request="auto")

        self.assertEqual(selection.device, "cpu")
        self.assertIn("requires sm_70", selection.reason)
        self.assertTrue(any("requirements-cuda126.txt" in str(item.message) for item in caught))

    @patch.object(
        device_runtime.torch.cuda, "get_arch_list", return_value=["sm_75", "sm_80", "sm_90"]
    )
    @patch.object(device_runtime.torch.cuda, "get_device_capability", return_value=(7, 0))
    @patch.object(device_runtime.torch.cuda, "get_device_name", return_value="Tesla V100")
    @patch.object(device_runtime.torch.cuda, "is_available", return_value=True)
    def test_forced_cuda_fails_fast_when_architecture_is_missing(self, *_mocks):
        with self.assertRaisesRegex(GPUUnavailableError, "requires sm_70"):
            device_runtime.select_runtime_device(request="cuda")

    @patch.object(
        device_runtime,
        "_probe_cuda_subprocess",
        return_value=(False, "CUDA smoke test terminated by SIGILL"),
    )
    @patch.object(device_runtime.torch.cuda, "get_arch_list", return_value=["sm_80"])
    @patch.object(device_runtime.torch.cuda, "get_device_capability", return_value=(8, 0))
    @patch.object(device_runtime.torch.cuda, "get_device_name", return_value="NVIDIA A100")
    @patch.object(device_runtime.torch.cuda, "is_available", return_value=True)
    def test_fatal_probe_failure_is_contained_and_reported(self, *_mocks):
        with self.assertRaisesRegex(GPUUnavailableError, "SIGILL"):
            device_runtime.select_runtime_device(request="cuda")

    def test_cpu_override_never_initializes_cuda(self):
        with patch.object(device_runtime.torch.cuda, "is_available") as is_available:
            selection = device_runtime.select_runtime_device(request="cpu")

        self.assertEqual(selection.device, "cpu")
        is_available.assert_not_called()


class RequirementProfileTests(unittest.TestCase):
    LINUX_MARKERS = {"sys_platform": "linux"}

    def test_cpu_profile_pins_cpu_only_linux_torch(self):
        expected = load_expected_versions(
            PROJECT_ROOT / "requirements.txt", self.LINUX_MARKERS
        )

        self.assertEqual(expected["torch"], "2.13.0+cpu")
        self.assertEqual(expected["sentence-transformers"], "5.7.0")

    def test_cluster_profile_pins_cuda_126_torch(self):
        expected = load_expected_versions(
            PROJECT_ROOT / "requirements-cuda126.txt", self.LINUX_MARKERS
        )

        self.assertEqual(expected["torch"], "2.13.0+cu126")
        self.assertEqual(expected["sentence-transformers"], "5.7.0")

    @patch.object(check_environment.importlib.metadata, "version", return_value="2.13.0+cu126")
    def test_auto_profile_detects_cuda_126(self, _version):
        self.assertEqual(check_environment.detect_profile(), "cuda126")

    @patch.object(check_environment.importlib.metadata, "version", return_value="2.13.0+cu130")
    def test_auto_profile_rejects_cuda_130(self, _version):
        with self.assertRaisesRegex(ValueError, "Unsupported automatic profile"):
            check_environment.detect_profile()


if __name__ == "__main__":
    unittest.main()
