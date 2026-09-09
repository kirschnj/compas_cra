********************************************************************************
Three-Block Tilt-Uncertainty RBE Force Range
********************************************************************************

Compare block-2 safe ``Fx``-``Fz`` load regions for the example-18 three-block
assembly under symmetric foundation tilt uncertainty.

Foundation tilt is modeled as a change in gravity direction in the structure
``XZ`` plane only. Contact geometry, interface normals, friction cones, support
conditions, and the structural equilibrium matrix remain fixed. The plotted
safe-load region is reported back in the original world ``Fx``-``Fz`` axes.
The example specifies tilt angles in degrees and converts them to radians for
the solver.

The visible block-2 load is realized by hidden point forces at the four vertices
of block 2's rightmost face. Their summed ``x`` and ``z`` components define the
visible ``Fx`` and ``Fz`` loads, while the generated moment comes from
``r x q``. Each hidden force component uses a large ``1e6`` placeholder bound;
this is a numerical stand-in until actuator, contact, or material limits are
available.

The first plot compares the certainty case with symmetric tilt intervals
``+/-10``, ``+/-20``, ``+/-30``, ``+/-40``, ``+/-50``, and ``+/-60`` degrees.
Each nonzero interval checks the two endpoint tilt scenarios.

The script also saves a single-angle diagnostic plot for deterministic tilt
angles from ``-60`` to ``+60`` degrees in 10 degree increments. These cases use
one exact tilt angle at a time through the solver's ``tilt_angles`` argument.
Use ``tilt_load_frame="structure"`` only when intentionally reproducing the
older local-frame diagnostic convention.

.. literalinclude:: 18_3_rbe_uncertainty_tilt_3block.py
    :language: python
