********************************************************************************
Three-Block Robust RBE Force Range
********************************************************************************

Compare radial sampling, primal support, and dual support approximations of the
safe ``Fx``-``Fz`` load set on block 2. Block 0 is fixed, and gravity is the
only baseline load.

The visible load is realized by hidden point forces at the four vertices of
block 2's rightmost face. Their summed ``x`` and ``z`` components define the
visible ``Fx`` and ``Fz`` loads, while the generated moment comes from
``r x q``. Each hidden force component uses a large ``1e6`` placeholder bound;
this is a numerical stand-in until actuator, contact, or material limits are
available.

The plot uses fixed viewport limits for comparison. These viewport limits are
only for visualization and are not physical load constraints.

When run with an interactive Matplotlib backend, the script also opens a
``compas_view2`` window showing the three analyzed assemblies side by side. The
red arrow marks the positive ``Fx`` direction and the blue arrow marks the
positive ``Fz`` direction at the block-2 load application face. These arrows show
the coordinate convention only, not a solved load magnitude. The viewer is not
opened when running with ``MPLBACKEND=Agg``.

The dual support result also reports the governing visible boundary equations
``a Fx + b Fz <= c`` and labels the equations that intersect the displayed
viewport.

.. literalinclude:: 18_rbe_robust_3block.py
    :language: python
