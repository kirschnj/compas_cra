********************************************************************************
Four-Block Arch RBE Disturbance-Uncertainty Force Range
********************************************************************************

This example uses the example-06 arch geometry. Block 0 is fixed, every
non-support and non-loaded block carries uncertain external disturbance, and
the visible safe load is the ``Fx``-``Fz`` load on the last block. The default
setup uses four left-to-right blocks, but the load node follows
``NUM_BLOCKS - 1`` if the block count is changed.

The disturbance on each interior block is bounded by that block's own weight:

``Fx_i, Fz_i in [-r Wi, r Wi]``

where ``Wi`` is the corresponding block weight and ``r`` is selected from
``DISTURBANCE_RATIOS``. A ``0%`` case is included as the baseline. For each
nonzero ratio, the script checks four coherent global disturbance-direction
scenarios ``(+/-Fx, +/-Fz)`` applied to all disturbed blocks at once, then adds
``RANDOM_DISTURBANCE_SAMPLES`` reproducible random samples inside the fully
independent interior-block disturbance box. This gives a tractable diagnostic
without enumerating the full exponential vertex set. This is a fixed full-arch
analysis, not a construction-process simulation. If a ratio makes the robust
safe-load set empty, the script reports that case in the console and omits it
from the plotted curves.

Runtime scales with ``NUM_DIRECTIONS`` times the number of disturbance
scenarios. With the default random setting, each nonzero ratio checks ``64``
scenarios: four coherent cases plus sixty random samples.

The last-block visible load is realized through hidden point forces at the four
vertices of the last block's right exposed radial face. The hidden force bound
remains the large ``1e6`` placeholder used by the robust examples.

The Matplotlib plot uses automatic view limits by default, computed from the
solved safe-region points. Set ``AUTO_VIEW_LIMITS = False`` in the script to
use the manual ``VIEW_XLIM`` and ``VIEW_YLIM`` values for a fixed zoom.

When run with an interactive Matplotlib backend, the script also opens a
``compas_view2`` assembly view. The red arrow marks positive ``Fx`` and the
blue arrow marks positive ``Fz`` at the last-block load face. This viewer is
disabled under ``MPLBACKEND=Agg`` so automated SVG generation remains headless.

.. literalinclude:: 19_2_rbe_uncertainty_disturb_arch.py
    :language: python
