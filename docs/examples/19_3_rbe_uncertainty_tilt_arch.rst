********************************************************************************
Four-Block Arch RBE Foundation-Tilt Force Range
********************************************************************************

This example uses the same fixed four-block arch geometry as example 19_2:
``height=2``, ``span=4``, blocks ordered left to right, and block 0 fixed. The
visible safe load is the ``Fx``-``Fz`` load on block 3.

Foundation tilt is modeled as a change in gravity direction in the structure
coordinate frame. Contact geometry, contact normals, friction cones, support
conditions, the structural matrices, and the block 3 load-application points
remain fixed. The plotted safe-load region is reported back in the original
world ``Fx``-``Fz`` axes.

The plot compares the certainty case with symmetric tilt intervals
``±2.5°``, ``±5°``, ``±7.5°``, and ``±10°``. Each nonzero interval checks the two
endpoint tilt scenarios. The block 3 visible load is realized through hidden
point forces at the four vertices of block 3's right exposed radial face, with the same
large ``1e6`` placeholder force bound used by the robust examples.

Use ``tilt_load_frame="structure"`` only when intentionally reproducing the
older local-frame diagnostic convention.

The script also saves a deterministic single-angle diagnostic plot for exact
tilt angles from ``-20`` to ``+20`` degrees in 5 degree increments. These
single-angle cases use the solver's ``tilt_angles`` argument and are reported
in the same world ``Fx``-``Fz`` axes.

When run with an interactive Matplotlib backend, the script also opens a
``compas_view2`` assembly view. The red arrow marks positive ``Fx`` and the
blue arrow marks positive ``Fz`` at the block-3 load face. This viewer is
disabled under ``MPLBACKEND=Agg`` so automated SVG generation remains headless.

.. literalinclude:: 19_3_rbe_uncertainty_tilt_arch.py
    :language: python
