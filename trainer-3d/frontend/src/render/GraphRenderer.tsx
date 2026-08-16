import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GraphNode, Triangle } from "../net/socket";

interface GraphRendererProps {
  nodes: GraphNode[];
  edges: [number, number][];
  triangles: Triangle[];
  targetPoints: [number, number, number][] | null;
  onGrowTriangle: (triangleId: number) => void;
}

const NODE_RADIUS = 0.08;

function buildTriangleGeometry(
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3
): THREE.BufferGeometry {
  const verts = new Float32Array([a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export function GraphRenderer({
  nodes,
  edges,
  triangles,
  targetPoints,
  onGrowTriangle,
}: GraphRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const nodeGroupRef = useRef<THREE.Group | null>(null);
  const edgeLinesRef = useRef<THREE.LineSegments | null>(null);
  const triangleGroupRef = useRef<THREE.Group | null>(null);
  const targetPointsObjectRef = useRef<THREE.Points | null>(null);
  const nodeGeometryRef = useRef<THREE.SphereGeometry | null>(null);
  const nodeMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const activeMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const grownMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const onGrowRef = useRef(onGrowTriangle);
  onGrowRef.current = onGrowTriangle;

  // Scene, camera, renderer, controls, and hover/click handling are set up
  // once. Live graph data is read directly off each triangle mesh's
  // userData at hover/click time, so this effect never depends on it.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      50,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(3, 2.5, 4);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(8, 12, 6);
    scene.add(dirLight);

    const grid = new THREE.GridHelper(20, 20, 0x2a2d35, 0x2a2d35);
    scene.add(grid);

    const nodeGroup = new THREE.Group();
    scene.add(nodeGroup);
    nodeGroupRef.current = nodeGroup;

    const edgeLines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x6a7280 })
    );
    scene.add(edgeLines);
    edgeLinesRef.current = edgeLines;

    const triangleGroup = new THREE.Group();
    scene.add(triangleGroup);
    triangleGroupRef.current = triangleGroup;

    // The goal shape, overlaid as a faint point cloud — not raycast
    // against, purely a visual reference for how far the structure has
    // grown toward it.
    const targetPointsMaterial = new THREE.PointsMaterial({
      color: 0xc9c9d9,
      size: 0.05,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    });
    const targetPointsObject = new THREE.Points(new THREE.BufferGeometry(), targetPointsMaterial);
    scene.add(targetPointsObject);
    targetPointsObjectRef.current = targetPointsObject;

    const nodeGeometry = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
    const nodeMaterial = new THREE.MeshStandardMaterial({
      color: 0x4f8cff,
      roughness: 0.5,
      metalness: 0.1,
    });
    nodeGeometryRef.current = nodeGeometry;
    nodeMaterialRef.current = nodeMaterial;

    // Growable faces: interactive, brighter. Grown faces: inert, dimmer —
    // they're now internal to the structure, just a record of where it's
    // already grown.
    const activeMaterial = new THREE.MeshStandardMaterial({
      color: 0x4f8cff,
      transparent: true,
      opacity: 0.16,
      roughness: 0.6,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    activeMaterialRef.current = activeMaterial;

    const grownMaterial = new THREE.MeshStandardMaterial({
      color: 0x4a4d57,
      transparent: true,
      opacity: 0.05,
      roughness: 0.8,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    grownMaterialRef.current = grownMaterial;

    // Ghost node + spokes shown on hover, before the growth is committed.
    const previewGroup = new THREE.Group();
    previewGroup.visible = false;
    scene.add(previewGroup);

    const previewNodeGeometry = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
    const previewNodeMaterial = new THREE.MeshStandardMaterial({
      color: 0xffb84f,
      transparent: true,
      opacity: 0.6,
      roughness: 0.4,
    });
    const previewNode = new THREE.Mesh(previewNodeGeometry, previewNodeMaterial);
    previewGroup.add(previewNode);

    const previewLines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xffb84f, transparent: true, opacity: 0.7 })
    );
    previewGroup.add(previewLines);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const setPointerFromEvent = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
    };
    const growableMeshes = () =>
      triangleGroup.children.filter((child) => !child.userData.grown);

    const handlePointerMove = (event: PointerEvent) => {
      setPointerFromEvent(event);
      const hits = raycaster.intersectObjects(growableMeshes());
      if (hits.length === 0) {
        previewGroup.visible = false;
        return;
      }

      const vertexPositions = hits[0].object.userData.vertexPositions as THREE.Vector3[];
      const apex = hits[0].object.userData.apexPreview as THREE.Vector3;

      previewNode.position.copy(apex);

      const linePositions = new Float32Array(vertexPositions.length * 2 * 3);
      let o = 0;
      for (const p of vertexPositions) {
        linePositions[o++] = apex.x;
        linePositions[o++] = apex.y;
        linePositions[o++] = apex.z;
        linePositions[o++] = p.x;
        linePositions[o++] = p.y;
        linePositions[o++] = p.z;
      }
      previewLines.geometry.dispose();
      previewLines.geometry = new THREE.BufferGeometry();
      previewLines.geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));

      previewGroup.visible = true;
    };
    renderer.domElement.addEventListener("pointermove", handlePointerMove);

    const handleClick = (event: MouseEvent) => {
      setPointerFromEvent(event);
      const hits = raycaster.intersectObjects(growableMeshes());
      if (hits.length === 0) {
        console.debug("[trainer] click missed every growable triangle");
        return;
      }
      const triangleId = hits[0].object.userData.triangleId as number;
      console.debug("[trainer] growing triangle", triangleId);
      onGrowRef.current(triangleId);
      previewGroup.visible = false;
    };
    renderer.domElement.addEventListener("click", handleClick);

    let frameId: number;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("click", handleClick);
      controls.dispose();
      nodeGeometry.dispose();
      nodeMaterial.dispose();
      activeMaterial.dispose();
      grownMaterial.dispose();
      targetPointsObject.geometry.dispose();
      targetPointsMaterial.dispose();
      previewNodeGeometry.dispose();
      previewNodeMaterial.dispose();
      previewLines.geometry.dispose();
      (previewLines.material as THREE.Material).dispose();
      edgeLines.geometry.dispose();
      (edgeLines.material as THREE.Material).dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Rebuild node spheres, edge lines, and triangle meshes whenever the
  // graph state changes.
  useEffect(() => {
    const nodeGroup = nodeGroupRef.current;
    const edgeLines = edgeLinesRef.current;
    const triangleGroup = triangleGroupRef.current;
    const nodeGeometry = nodeGeometryRef.current;
    const nodeMaterial = nodeMaterialRef.current;
    const activeMaterial = activeMaterialRef.current;
    const grownMaterial = grownMaterialRef.current;
    if (
      !nodeGroup ||
      !edgeLines ||
      !triangleGroup ||
      !nodeGeometry ||
      !nodeMaterial ||
      !activeMaterial ||
      !grownMaterial
    ) {
      return;
    }

    for (const child of [...nodeGroup.children]) nodeGroup.remove(child);
    for (const node of nodes) {
      const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
      mesh.position.set(node.position[0], node.position[1], node.position[2]);
      nodeGroup.add(mesh);
    }

    const positionById = new Map(nodes.map((n) => [n.id, n.position]));

    const linePositions = new Float32Array(edges.length * 2 * 3);
    let o = 0;
    for (const [a, b] of edges) {
      const pa = positionById.get(a);
      const pb = positionById.get(b);
      if (!pa || !pb) continue;
      linePositions[o++] = pa[0];
      linePositions[o++] = pa[1];
      linePositions[o++] = pa[2];
      linePositions[o++] = pb[0];
      linePositions[o++] = pb[1];
      linePositions[o++] = pb[2];
    }
    edgeLines.geometry.dispose();
    edgeLines.geometry = new THREE.BufferGeometry();
    edgeLines.geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));

    for (const child of [...triangleGroup.children]) {
      triangleGroup.remove(child);
      (child as THREE.Mesh).geometry.dispose();
    }
    for (const tri of triangles) {
      const [pa, pb, pc] = tri.vertices.map((id) => {
        const p = positionById.get(id)!;
        return new THREE.Vector3(p[0], p[1], p[2]);
      });
      const geometry = buildTriangleGeometry(pa, pb, pc);
      const mesh = new THREE.Mesh(geometry, tri.grown ? grownMaterial : activeMaterial);
      mesh.userData.triangleId = tri.id;
      mesh.userData.grown = tri.grown;
      mesh.userData.vertexPositions = [pa, pb, pc];
      if (tri.apexPreview) {
        mesh.userData.apexPreview = new THREE.Vector3(...tri.apexPreview);
      }
      triangleGroup.add(mesh);
    }
  }, [nodes, edges, triangles]);

  // Rebuild the target overlay whenever the selected target changes.
  useEffect(() => {
    const targetPointsObject = targetPointsObjectRef.current;
    if (!targetPointsObject) return;

    targetPointsObject.geometry.dispose();

    if (!targetPoints || targetPoints.length === 0) {
      targetPointsObject.geometry = new THREE.BufferGeometry();
      targetPointsObject.visible = false;
      return;
    }

    const positions = new Float32Array(targetPoints.length * 3);
    let o = 0;
    for (const [x, y, z] of targetPoints) {
      positions[o++] = x;
      positions[o++] = y;
      positions[o++] = z;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    targetPointsObject.geometry = geometry;
    targetPointsObject.visible = true;
  }, [targetPoints]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
