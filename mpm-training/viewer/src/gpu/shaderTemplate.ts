import domainFunctions from "../../../core/materialDomain.wgsl?raw";
// WGSL has no preprocessor — compile-time constants get baked in as
// plain __NAME__ string substitution before device.createShaderModule(),
// same convention trainer/shader_template.py's template_shader() (Python
// side) and mls-mpm/src/gpu/shaderTemplate.ts use.

export function templateShader(source: string, vars: Record<string, string | number>): string {
  let result = source.split("__DOMAIN_FUNCTIONS__").join(domainFunctions);
  for (const [key, value] of Object.entries(vars)) {
    result = result.split(`__${key}__`).join(String(value));
  }
  return result;
}
