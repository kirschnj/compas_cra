********************************************************************************
Full-Arch Discretization Comparison With Fixed Foundation
********************************************************************************

This example is a corrected discretization comparison for the complete
example-19 arch. In ``19_5_rbe_discretization_arch.py``, arch block ``0`` is
fixed directly. Because arch block ``0`` changes shape when the block count
changes, the support geometry also changes with discretization.

This example instead adds a constant foundation block below the left springing
block and fixes only that foundation. Arch block ``0`` remains a free block and
is part of the discretized arch. The foundation geometry is the same for every
block count, so the fixed boundary condition is no longer a discretization
artifact.

The block count is swept from ``15`` through ``30`` while keeping span, height,
thickness, depth, friction, density, and hidden load-application radial-face
model fixed. The visible ``Fx``-``Fz`` load is applied to the last arch block.
The Matplotlib region plots use automatic view limits by default because the
fixed-foundation support reactions move the safe-load region away from the
viewport used by ``19_5``.

The script saves three SVG files:

* ``19_5_1_rbe_discretization_arch_fixed_foundation.svg`` overlays all solved
  safe-load regions.
* ``19_5_1_rbe_discretization_arch_fixed_foundation_overlay.svg`` is a clean
  visual-only overlay of all solved safe-load regions in one graph.
* ``19_5_1_rbe_discretization_arch_fixed_foundation_metrics.svg`` plots polygon
  area and load bounds versus arch block count.

When run with an interactive Matplotlib backend, the script also opens a
``compas_view2`` grid view of the fixed-foundation arch discretizations. The red
arrow marks positive ``Fx`` and the blue arrow marks positive ``Fz`` at each
last-block load face. This viewer is disabled under ``MPLBACKEND=Agg`` so SVG
generation and checks remain headless.

.. literalinclude:: 19_5_1_rbe_discretization_arch_fixed_foundation.py
    :language: python
