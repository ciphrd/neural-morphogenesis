/** WGSL has no preprocessor, but several shaders here need compile-time
 * constants (grid size, channel count, agent count, ...) baked in as
 * `const` declarations so function-scope scratch arrays can be sized —
 * this does plain string substitution of `__NAME__` tokens before
 * `device.createShaderModule()`, once per `reset(config)` (see
 * gpu/simulation.ts), not per frame. */
export function templateShader(source: string, vars: Record<string, string | number>): string {
  let result = source;
  for (const [key, value] of Object.entries(vars)) {
    result = result.split(`__${key}__`).join(String(value));
  }
  return result;
}
