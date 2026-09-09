********************************************************************************
Full-Arch Boundary Failure-Mode Diagnostics
********************************************************************************

This example analyzes the complete example-19 arch, not the construction
sequence. The arch uses block ``0`` as the only support and applies the visible
``Fx``-``Fz`` load to the last block. The visible load is realized through
hidden point forces at the four vertices of the last block's right exposed radial face.

The script enforces a 2D-extruded diagnostic model by tying every front/back
contact pair with matching ``x`` and ``z`` coordinates. The tied variables have
the same normal and friction values and directions. The same tie is applied to
the hidden point forces on the last block's right exposed radial face, so the end load is
also applied as a 2D pair model.

The script computes the robust visible safe-load polygon from this tied
compression-only matrix problem. Each plotted polygon edge is treated as one
visible boundary line ``a Fx + b Fz <= c``. For every boundary, the script takes
the edge midpoint, offsets it outward along the halfspace normal, and checks
that outside load with two LPs:

* a compression-only feasibility LP;
* a penalty LP that minimizes total tensile normal force ``fn_minus``.

Contacts with nonzero ``fn_minus`` identify where the structure would need
tension if that outside load were imposed. The script also reports open
contacts where both ``fn_plus`` and ``fn_minus`` are zero within tolerance, so
contacts that carry no normal force are visible in the console diagnostic. The
SVG marks boundary labels, edge midpoints, and outward test points so the
console report can be mapped back to the safe-load plot.

When run with an interactive Matplotlib backend, the script also opens a
``compas_view2`` full-arch assembly view. The red arrow marks positive ``Fx``
and the blue arrow marks positive ``Fz`` at the last-block load face. This
viewer is disabled under ``MPLBACKEND=Agg`` so automated diagnostics remain
headless.

.. literalinclude:: 19_4_rbe_boundary_failure_modes_arch.py
    :language: python
