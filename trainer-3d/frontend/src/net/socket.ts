import { useEffect, useRef, useState } from "react";

export interface GraphNode {
  id: number;
  position: [number, number, number];
}

export interface Triangle {
  id: number;
  vertices: [number, number, number];
  grown: boolean;
  apexPreview: [number, number, number] | null;
}

export interface GraphState {
  nodes: GraphNode[];
  edges: [number, number][];
  triangles: Triangle[];
}

const EMPTY_STATE: GraphState = { nodes: [], edges: [], triangles: [] };

export function useGraphSocket(url: string) {
  const [state, setState] = useState<GraphState>(EMPTY_STATE);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "state") {
        setState({
          nodes: message.nodes,
          edges: message.edges,
          triangles: message.triangles,
        });
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

  const growTriangle = (triangleId: number) => {
    socketRef.current?.send(JSON.stringify({ type: "grow_triangle", triangleId }));
  };

  return { state, growTriangle };
}
