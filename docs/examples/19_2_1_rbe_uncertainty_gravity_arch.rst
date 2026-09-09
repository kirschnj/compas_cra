********************************************************************************
Arch RBE Gravity/Material-Uncertainty Force Range
********************************************************************************

This example uses the same arch setup as example 19_2: block 0 is fixed, the
visible safe load is the ``Fx``-``Fz`` load on the last block, and that visible
load is realized through hidden point forces on the last block's right exposed
radial face.

Unlike example 19_2, this example models material uncertainty as gravity-related
uncertainty rather than arbitrary horizontal disturbance. Every non-support and
non-loaded block is uncertain.

Two models are compared:

* **Weight only:** each uncertain block has vertical load variation
  ``Delta Fz_i in [-r Wi, r Wi]``.
* **Weight + CoG moment:** each uncertain block has the same vertical load
  variation plus an equivalent ``My`` moment from a shifted gravity line of
  action. The placeholder offset range is ``dx_i = r * block_x_span_i``, so
  ``Delta My_i in [-Wi dx_i, Wi dx_i]``.

For each nonzero ratio, the script checks coherent sign scenarios and adds
``RANDOM_UNCERTAINTY_SAMPLES`` reproducible random samples inside the
independent uncertainty box. This avoids enumerating the full exponential vertex
set over all uncertain blocks. Empty safe-load cases are reported in the console
and omitted from the plotted curves.

.. literalinclude:: 19_2_1_rbe_uncertainty_gravity_arch.py
    :language: python
