from pathlib import Path
import tomllib


def test_rocm_torch_family_uses_amd_official_wheels():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sources = project["tool"]["uv"]["sources"]
    overrides = project["tool"]["uv"]["override-dependencies"]

    expected_urls = {
        "torch": "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.10.0%2Brocm7.2.2.lw.git23d69b29-cp312-cp312-linux_x86_64.whl",
        "torchaudio": "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchaudio-2.10.0%2Brocm7.2.2.git5047768f-cp312-cp312-linux_x86_64.whl",
        "torchvision": "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchvision-0.25.0%2Brocm7.2.2.git82df5f59-cp312-cp312-linux_x86_64.whl",
        "triton": "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/triton-3.6.0%2Brocm7.2.2.git4ed88892-cp312-cp312-linux_x86_64.whl",
    }

    for package_name, url in expected_urls.items():
        [source] = sources[package_name]
        assert source["url"] == url
        assert source["marker"] == "sys_platform == 'linux'"
        assert (
            f"{package_name} @ {url} ; sys_platform == 'linux'"
            in overrides
        )


def test_api_group_uses_platform_specific_llm_backends():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    api_dependencies = project["dependency-groups"]["api"]
    sources = project["tool"]["uv"]["sources"]

    assert "mlx-lm>=0.20; sys_platform == 'darwin'" in api_dependencies
    assert "vllm>=0.14; sys_platform == 'linux'" in api_dependencies
    assert sources["vllm"] == [
        {"index": "vllm-rocm", "marker": "sys_platform == 'linux'"}
    ]
