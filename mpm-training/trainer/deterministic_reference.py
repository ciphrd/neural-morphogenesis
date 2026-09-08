"""Slow, ordered native-GPU reference with unchanged floating-point formulas.

Production float atomics remain parallel by default. Serializing their callers
is useful for reproducibility and validating a future fixed-order parallel
reduction, without selecting a lossy fixed-point scale. This controls ordering
on one backend, not rounding/transcendental behavior across GPU vendors.
"""
import re
import shader_template

REFERENCE_STAGES = ('p2g', 'agentStep', 'splatChemicalState')
AUDIT_STAGES = REFERENCE_STAGES + ('indexRefinementEdges', 'propagateRefinement', 'reserveRefinement')
_original_template = shader_template.template_shader


def ordered_shader_source(source, names):
    for name in names:
        if name not in AUDIT_STAGES:
            raise ValueError(f'unsupported ordered stage: {name}')
        pattern = r'@compute @workgroup_size\(64\)\s*fn '+name+r'\(@builtin\(global_invocation_id\) gid: vec3<u32>\)'
        source, count = re.subn(pattern, 'fn ordered_'+name+'(gid: vec3<u32>)', source)
        if count:
            source += f'''
@compute @workgroup_size(64)
fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x != 0u) {{ return; }}
  for (var pi=0u; pi<activeCount; pi++) {{ ordered_{name}(vec3<u32>(pi,0u,0u)); }}
}}
'''
    return source


def install_ordered_stages(names=REFERENCE_STAGES):
    """Configure shader compilation before constructing GPU systems in a worker."""
    names=tuple(names)
    if any(n not in AUDIT_STAGES for n in names) or len(set(names)) != len(names):
        raise ValueError('ordered stages must be unique supported entry points')
    def transform(source, variables):
        return ordered_shader_source(_original_template(source, variables), names)
    shader_template.template_shader = transform if names else _original_template
