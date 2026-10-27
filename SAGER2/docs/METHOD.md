# Main inference method

At each selected decoder-block output, use only the last input-token hidden
state. Normalize it to unit L2 norm in FP64. The layerwise ReLU MLP is
`f(z)=v^T ReLU(W_eff z+b)+c`, where
`W_eff=sqrt(hidden_size)*fc1.weight`. Training labels are benign=1 and toxic=0;
the score is a raw benign-oriented logit, not a calibrated safety probability.
The gate is the mean score of benign TRAIN examples for that model and layer.

When the current score is below that mean, construct anchors from Top2 absolute
gradient and Top2 absolute activation dimensions, remove duplicates, and form
unordered coordinate pairs with at least one anchor. Rank the eligible pairs
by `abs(z_i*g_j-z_j*g_i)` and retain100. Each actual rotation solves the finite
±5° ReLU piecewise angle problem for every pair in the current window and applies
the pair/angle with the largest positive exact gain. Repeated coordinates are
allowed; at most800 distinct coordinates may be used. Gain must exceed1e-9.

All six model configurations recompute gradient, anchors and candidate
IDs every20 slots. Exact finite-angle gains and preactivations are updated after
each rotation; rotations are sequential, not100 simultaneous updates. All use a
500-slot budget and do not stop merely on reaching the benign mean threshold.
All selected layers skip editing when the current input's final token ID is13.

Restore the original hidden norm and write back in the model's FP16 dtype.
Orthogonality preserves norm in real arithmetic; FP16 writeback introduces
rounding. Neither norm preservation, a5° per-pair angle, nor sparse coordinate
coverage guarantees small cumulative drift or semantic equivalence.

## Implementations

- `backends/window_solver.py`, `window_ops.py`, `angle_body.py`: main refresh-20
  implementation with100 parallel angle problems and sequential rotations.
- `backends/probe.py`: shared FP64 analytic probe parameters and scoring.
- `editor.py`: main batch-one prepared-state reuse and non-triggered fast path.

Unused alternative experimental modes are removed. Import namespaces and path
handling were reorganized. Core angle/search arithmetic was copied from the
actual main implementation and is checked against the source AST/parameters in
the packaging audit. GPU compiler/hardware arithmetic can still differ; CPU and
packaging checks alone are not an all-input, bitwise reproduction guarantee.
