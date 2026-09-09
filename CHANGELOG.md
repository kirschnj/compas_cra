# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

* Add robust RBE radial sampling, primal and dual support-function analyses, and optional Matplotlib visualization for two-dimensional safe load ranges.
* Add a three-block robust RBE example comparing radial, primal-support, and dual-support safe-load approximations.
* Add a three-block RBE disturbance-uncertainty example comparing block-1 disturbance levels.
* Add a robust RBE arch-construction example showing stage-by-stage safe-load regions.
* Add a four-block arch disturbance-uncertainty example comparing block-1 and block-2 disturbance levels.
* Add robust RBE safe-load analysis for polyhedral external-load uncertainty.
* Add robust RBE foundation-tilt uncertainty analysis and a four-block arch tilt example.
* Add finite-scenario RBE geometry-uncertainty safe-load analysis and a three-block comparison example.
* Add full-arch robust RBE boundary failure-mode and discretization comparison examples.
* Add a full-arch graphic-statics example linking a maximal joint-admissible safe-load contour to a thrust-line
  family, with supplied-case pressure-point and friction diagnostics that distinguish CoG concurrency points from
  physical joint resultants.
* Add a four-geometry thrust-line comparison for circular, symmetric-catenary, and mirrored asymmetric-catenary
  arches, with separate self-weight stability and one-sided anchor-family checks.
* Add a circular/catenary joint-opening comparison with certified right-load boundaries, verified initial
  rigid-block mechanisms, and an offline viewer distinguishing pressure-limit candidates from opening edges.

### Changed

* Upgrade the joint-opening viewer with linked thrust-line family and failure tabs, bidirectional load/path
  selection, optional construction lines, and shared boundary-case and mechanism-branch selections.
* Enlarge the joint-opening example's motion preview to a collision-checked ten-degree range, with a
  rotation readout and automatic framing so opening gaps are easier to inspect.
* Fix Pyomo force-objective weighting for three- and four-component contact force layouts.
* Correct the full-arch thrust examples to use mesh-volume weights and to reconstruct one-sided asymmetric
  pressure paths from the same right side as the applied RBE load.
* Allow robust RBE analyses to use shifted feasible load regions that do not contain the load-increment origin.
* Allow robust RBE analyses to project visible loads from bounded hidden point forces at candidate application points.
* Allow robust RBE plots to clip unbounded regions to explicit visualization limits.
* Extend the three-block robust RBE example to print and annotate governing visible boundary equations.
* Extend the robust RBE arch-construction example to save thickness-specific SVG plots with zoomed stage views.
* Report RBE foundation-tilt safe-load regions in world ``Fx``-``Fz`` coordinates by default.

### Removed


## [0.4.0] 2024-03-02

### Added

* Add delete block and blocks methods in CRA_Assembly class. 
* A script to export mesh to json in Rhino. 

### Changed

### Removed


## [0.3.0] 2022-11-06

### Added

### Changed

### Removed


## [0.2.2] 2022-09-29

### Added

* Add example folder directory to tutorial docs for easy access. 

### Changed

* Fix some typos and wrong url links. 
* Change ipopt installation guide using conda.

### Removed


## [0.2.1] 2022-09-02

### Added

### Changed

### Removed


## [0.2.0] 2022-09-02

### Added

### Changed

### Removed

