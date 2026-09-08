struct ParticleRest {
  growthF: vec4<f32>, jp: f32, growthVectorX: f32, growthVectorY: f32,
  verticesAB: vec4<f32>, vertexC: vec2<f32>, originalArea: f32, quadratureWeight: f32,
}
struct Blade { start: vec2<f32>, end: vec2<f32>, radius: f32, padding0: f32, padding1: f32, padding2: f32 }
struct Counts { particles: u32, strokes: u32 }
@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> material: array<ParticleRest>;
@group(0) @binding(2) var<storage, read> blades: array<Blade>;
@group(0) @binding(3) var<storage, read_write> victims: array<u32>;
@group(0) @binding(4) var<uniform> counts: Counts;

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x*b.y-a.y*b.x; }
fn pointDistanceSquared(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let edge=b-a;
  let t=clamp(dot(p-a,edge)/max(dot(edge,edge),1e-20),0.0,1.0);
  let delta=p-(a+t*edge);
  return dot(delta,delta);
}
fn segmentDistanceSquared(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, d: vec2<f32>) -> f32 {
  let ab=b-a; let cd=d-c; let determinant=cross2(ab,cd);
  if (abs(determinant)>1e-12) {
    let t=cross2(c-a,cd)/determinant; let u=cross2(c-a,ab)/determinant;
    if (t>=0.0 && t<=1.0 && u>=0.0 && u<=1.0) { return 0.0; }
  }
  return min(min(pointDistanceSquared(a,c,d),pointDistanceSquared(b,c,d)),
    min(pointDistanceSquared(c,a,b),pointDistanceSquared(d,a,b)));
}
fn insideTriangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
  if (abs(cross2(b-a,c-a))<1e-12) { return false; }
  let signs=vec3<f32>(cross2(b-a,p-a),cross2(c-b,p-b),cross2(a-c,p-c));
  return all(signs>=vec3<f32>(0.0)) || all(signs<=vec3<f32>(0.0));
}

@compute @workgroup_size(64)
fn classifyCut(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pi=gid.x; if (pi>=counts.particles) { return; }
  let rest=material[pi];
  var hit=false;
  for (var i=0u; i<counts.strokes; i++) {
    let blade=blades[i]; let radiusSquared=blade.radius*blade.radius;
    if (rest.originalArea<=0.0) {
      hit=pointDistanceSquared(positions[pi],blade.start,blade.end)<=radiusSquared;
    } else {
      // Unwrap the material footprint, then test each periodic image against
      // the visible stroke. Never turn a long stroke into a shortcut over a seam.
      let a=rest.verticesAB.xy;
      let b=a+(rest.verticesAB.zw-a)-floor(rest.verticesAB.zw-a+vec2<f32>(0.5));
      let c=a+(rest.vertexC-a)-floor(rest.vertexC-a+vec2<f32>(0.5));
      for (var x=-1; x<=1; x++) {
        for (var y=-1; y<=1; y++) {
          let shift=vec2<f32>(f32(x),f32(y));
          let p=a+shift; let q=b+shift; let v=c+shift;
          let low=min(p,min(q,v))-vec2<f32>(blade.radius);
          let high=max(p,max(q,v))+vec2<f32>(blade.radius);
          if (any(max(blade.start,blade.end)<low) || any(min(blade.start,blade.end)>high)) { continue; }
          hit=insideTriangle(blade.start,p,q,v) || insideTriangle(blade.end,p,q,v)
            || segmentDistanceSquared(blade.start,blade.end,p,q)<=radiusSquared
            || segmentDistanceSquared(blade.start,blade.end,q,v)<=radiusSquared
            || segmentDistanceSquared(blade.start,blade.end,v,p)<=radiusSquared;
          if (hit) { break; }
        }
        if (hit) { break; }
      }
    }
    if (hit) { break; }
  }
  victims[pi]=select(0u,1u,hit);
}
