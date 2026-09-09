********************************************************************************
Three-Block RBE Disturbance-Uncertainty Force Range
********************************************************************************

This example keeps the three-block geometry from example 18. Block 0 is fixed,
the visible safe load remains the ``Fx``-``Fz`` load on block 2, and the block 2
load is realized through hidden point forces at the four vertices of its
rightmost face.

The uncertain load is separate from the visible safe load. It acts on block 1
through a two-dimensional disturbance box:

``Fx_b1, Fz_b1 in [-r W1, r W1]``

where ``W1`` is block 1's gravity load magnitude and ``r`` is ``5%``, ``10%``,
``15%``, or ``20%``. A ``0%`` case is included as the baseline. The plot shows
how the block 2 safe-load region changes as the block 1 disturbance bound
increases.

The hidden hand-force bound is still the large ``1e6`` placeholder used in
example 18. It is a numerical stand-in, not a physical actuator or contact
limit.

.. literalinclude:: 18_2_rbe_uncertainty_disturb_3block.py
    :language: python
