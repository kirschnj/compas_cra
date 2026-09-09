Three-Block RBE With Geometry Uncertainty
=========================================

This example uses the same three-block geometry as example 18. Block 0 is
fixed, and the visible safe load remains the ``(Fx, Fz)`` load on block 2. The
load is applied through four candidate points on the rightmost side of block 2,
with the same large ``1e6`` placeholder hand-force bound used by the robust
examples.

The geometry-uncertainty solver enumerates finite scenarios and intersects the
corresponding safe-load sets. This example saves three focused comparison
figures instead of overlaying every uncertainty type on one diagram:

* ``20_rbe_uncertainty_geometry_3block_region_shrink.svg`` compares global
  interface scale ratios. One ratio is applied to every detected interface in
  the scenario, so the two interfaces shrink by the same amount.
* ``20_rbe_uncertainty_geometry_3block_point_offsets.svg`` compares bounded
  in-plane contact-point offset samples. ``point_offset_bounds=[du, dv]``
  samples every interface point independently with local ``dw=0``, so the
  normal direction is not changed by the point-offset uncertainty.
* ``20_rbe_uncertainty_geometry_3block_normal_tilts.svg`` compares bounded
  normal-frame rotations. ``normal_tilt_bounds=[theta_u, theta_v]`` samples
  every interface frame independently. This example varies only the local ``y``
  rotation, but the solver still accepts two-component bounds.

The plotted safe-load regions are robust intersections over the generated
finite scenarios. Example 20 uses ``N=60`` deterministic scenarios, including
the nominal geometry, for each sampled bounded uncertainty curve. These are
finite scenario checks, not a continuous nonlinear geometry uncertainty model.

.. literalinclude:: 20_rbe_uncertainty_geometry_3block.py
    :language: python
