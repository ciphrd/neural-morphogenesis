"""Direct Python port of mls-mpm/src/gpu/shaderTemplate.ts's own
templateShader() — WGSL has no preprocessor, so compile-time constants
(grid resolution, dt, ...) get baked in as plain __NAME__ string
substitution before device.create_shader_module(), once at setup, not
per step. Kept byte-for-byte equivalent to the TS version rather than
"improved" — same tradeoff as core/'s own shaders: a small, obviously-
correct duplicate beats a shared build step neither language can use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

CORE_DIR = Path(__file__).parent.parent / "core"


def template_shader(source: str, template_vars: Mapping[str, object]) -> str:
    result = source.replace("__DOMAIN_FUNCTIONS__", (CORE_DIR / "materialDomain.wgsl").read_text())
    for key, value in template_vars.items():
        result = result.replace(f"__{key}__", str(value))
    return result


def load_core_shader(filename: str, template_vars: Mapping[str, object]) -> str:
    """Reads a .wgsl file straight out of ../core/ and applies
    template_shader() — the one place trainer/ touches core/'s files, so
    every consumer (mpm_core.py) goes through the same path."""
    source = (CORE_DIR / filename).read_text()
    return template_shader(source, template_vars)
