# Vector-first growth experiment

Same saved generation-4 checkpoint and rollout seed; no retraining. Compared with the earlier tensor-first diagnostic replay. Snapshots are captured before physics at the labeled macro step.

| At step 1000 | Tensor first | Vector first |
|---|---:|---:|
| Material area | 0.07062609 | 0.01658868 |
| Active samples | 52546 | 9340 |
| Mean NN vector magnitude | 0.19692 | 0.18782 |

Material area is 76.5% lower. Vector-first reaches 23.5% of the material budget by this snapshot.

This is an existing-policy replay, not evidence of improved trained fitness. Feedback changes the trajectory and subsequent neural outputs. GPU acceptance checks verify opposite-vector cancellation, unchanged aligned growth, perpendicular diagonal growth, and unequal opposition; the complete continuous-growth suite and viewer production build pass.
