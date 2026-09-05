# Continuous tissue growth

```mermaid
flowchart LR
  subgraph POLICY["1 · Tissue proposes"]
    P1((•)) -->|"↗ 0.8"| G
    P2((•)) -->|"→ 0.4"| G
    P3((•)) -->|"↗ 0.6"| G
  end

  G["2 · Smooth MPM growth field\nvolume-weighted average"]
  G --> T["3 · Continuous rest-volume growth\nFg ← exp(g Δt) Fg"]
  T --> S["4 · Surface moves through MPM"]
  S --> Q{"enough samples\nto resolve it?"}
  Q -->|yes| K["keep the same samples"]
  Q -->|no| N(("add a numerical sample"))

  style G fill:#173c2a,stroke:#55d98b,color:#fff
  style T fill:#173c2a,stroke:#55d98b,color:#fff
  style S fill:#173c2a,stroke:#55d98b,color:#fff
  style N fill:#245c3d,stroke:#8bf0ae,color:#fff
```

```mermaid
flowchart TB
  subgraph OLD["Division-based abstraction"]
    direction LR
    A((cell)) --> B((cell))
    A --> C((cell))
    D["new cell = new material"]
  end

  subgraph NEW["Volume-growth abstraction"]
    direction LR
    M1["material"] --> M2["more material"]
    M2 -. "adaptive resolution" .-> R(("extra sample"))
    E["growth is physical\nsampling is numerical"]
  end

  style OLD fill:#3d2225,stroke:#d47777,color:#fff
  style NEW fill:#173c2a,stroke:#55d98b,color:#fff
```

The neural network emits only a local 2-D vector. Its magnitude is growth rate;
its direction shapes the growth tensor. Nearby vectors are smoothly integrated
on the same grid as MPM, so overlapping samples do not multiply growth. New
points conserve existing represented volume and are added only to restore
integration resolution.
