/** WGSL has no preprocessor, but several shaders here need compile-time
 * constants (grid resolution, particle count, ...) baked in as `const`
 * declarations — this does plain string substitution of `__NAME__`
 * tokens before `device.createShaderModule()`, once at setup, not per
 * frame. */
export function templateShader(source: string, vars: Record<string, string | number>): string {
  let result = source;
  for (const [key, value] of Object.entries(vars)) {
    result = result.split(`__${key}__`).join(String(value));
  }
  return result;
}
