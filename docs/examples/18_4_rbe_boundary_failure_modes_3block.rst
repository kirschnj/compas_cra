********************************************************************************
Three-Block Boundary Failure-Mode Diagnostics
********************************************************************************

This example analyzes the full example-18 three-block structure, not the
two-block or rigid-compound comparison cases. Block ``0`` is fixed and the
visible load is the block-2 ``Fx``-``Fz`` load. The visible load is realized
through hidden point forces at the four vertices of block 2's rightmost face.

The script enforces a 2D-extruded diagnostic model by tying every front/back
contact pair with matching ``x`` and ``z`` coordinates. The tied variables have
the same normal and friction values and directions. The same tie is applied to
the hidden point forces on block 2's rightmost face, so the end load is also
applied as a 2D pair model.

The script computes the robust visible safe-load polygon from this tied
compression-only matrix problem. The failure-mode checks are restricted to the
example-18 viewport ``Fx=(-1.0, 0.1)`` and ``Fz=(-0.5, 1.0)``. Only support
boundaries with a nonzero active segment inside that viewport are diagnosed.
For every visible boundary segment, the script takes the clipped segment
midpoint, offsets it outward along the halfspace normal, and checks that
outside load with two LPs:

* a compression-only feasibility LP;
* a penalty LP that minimizes total tensile normal force ``fn_minus``.

Contacts with nonzero ``fn_minus`` identify where the structure would need
tension if that outside load were imposed. The script also reports open
contacts where both ``fn_plus`` and ``fn_minus`` are zero within tolerance, so
contacts that carry no normal force are visible in the console diagnostic.
Finally, contacts with ``sqrt(fu**2 + fv**2) / (mu * fn_plus)`` close to
``1`` are reported as contacts at the friction limit, meaning they are about
to slide. The SVG marks boundary labels, edge midpoints, and outward test
points so the console report can be mapped back to the safe-load plot.

.. literalinclude:: 18_4_rbe_boundary_failure_modes_3block.py
    :language: python
