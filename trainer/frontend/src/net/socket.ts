import { useEffect, useRef, useState } from "react";

export interface GraphNode {
  id: number;
  position: [number, number];
  // Only populated by the Training tab's client-side sim (sim/runner.ts) —
  // the interactive websocket's `state` message doesn't include these per
  // node, only for whichever one is selected (net/nodes.ts).
  chemicals?: number[];
  idVector?: number[];
  // Unlike chemicals/idVector, these *are* broadcast per-node by the
  // interactive websocket's `state` message (see main.py's
  // serialize_state) — small enough to send for every node so the
  // "direction" color mode and the always-on energy ring work live
  // without a per-node fetch.
  spawnDirection?: number[];
  splitProb?: number;
  energy?: number;
  // Raw [vx, vy] — velocity is persistent, integrated motion state;
  // accel is the fresh world-frame acceleration the network's last
  // invocation produced (added onto velocity, not itself accumulated).
  // Broadcast for every node (see main.py's serialize_state /
  // sim/runner.ts's toGraphNodes) since GraphRenderer always draws both
  // as direction ticks regardless of colorMode, not just in a specific
  // data-driven mode.
  velocity?: number[];
  accel?: number[];
}

export interface GraphState {
  nodes: GraphNode[];
  radius: number;
  // Denominator for the energy ring (GraphRenderer draws energy/maxEnergy
  // as an arc fraction) — sent alongside radius since it's the other
  // per-run constant the renderer needs but doesn't own itself.
  maxEnergy: number;
}

const EMPTY_STATE: GraphState = { nodes: [], radius: 0.5, maxEnergy: 100 };

export function useGraphSocket(url: string) {
  const [state, setState] = useState<GraphState>(EMPTY_STATE);
  const [playing, setPlaying] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  // read inside the message handler below, which closes over it once at
  // connect time — a ref keeps it live without needing to reconnect the
  // socket whenever play/pause toggles
  const playingRef = useRef(false);
  playingRef.current = playing;

  useEffect(() => {
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "state") {
        setState({ nodes: message.nodes, radius: message.radius, maxEnergy: message.maxEnergy });
        // Self-pacing autoplay: only send the next step once this one's
        // result has arrived, rather than firing on a fixed timer — a
        // single step can take well over a second once the graph is
        // large (O(n^2) physics), and a fixed interval would just pile
        // up a backlog of queued steps the server processes one at a
        // time anyway.
        if (playingRef.current) {
          socket.send(JSON.stringify({ type: "step" }));
        }
      }
    };

    // React 18 StrictMode mounts effects twice in dev (mount, cleanup,
    // mount) to surface non-idempotent effects. Closing a socket that's
    // still mid-handshake is what the "closed before the connection is
    // established" warning is about — harmless, but avoid it by deferring
    // the close until the handshake finishes instead of firing it blind.
    return () => {
      if (socket.readyState === WebSocket.CONNECTING) {
        socket.addEventListener("open", () => socket.close());
      } else {
        socket.close();
      }
    };
  }, [url]);

  const splitNode = (nodeId: number) => {
    socketRef.current?.send(JSON.stringify({ type: "split_node", nodeId }));
  };

  // Runs one autonomous simulation step: every node senses, decides, and
  // acts via the learned update rule, then physics relaxes the result.
  const step = () => {
    socketRef.current?.send(JSON.stringify({ type: "step" }));
  };

  const togglePlay = () => {
    setPlaying((wasPlaying) => {
      if (!wasPlaying) step(); // kick off the self-pacing chain
      return !wasPlaying;
    });
  };

  return { state, splitNode, step, playing, togglePlay };
}
