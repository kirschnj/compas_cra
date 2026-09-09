********************************************************************************
Robust RBE Force Range
********************************************************************************

Compare three approximations of the safe load set: an inner polygon from
radial sampling, inner and outer polygons from primal support solves, and an
outer polygon from dual support solves. The selected load components are
increments about gravity and any known external forces.

Radial sampling starts from a verified feasible load point. Therefore, the
safe load set may be shifted and does not have to contain the origin.
Matplotlib is optional and is imported only by the plotting helper.

.. literalinclude:: 17_rbe_robust.py
    :language: python
