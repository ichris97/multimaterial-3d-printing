"""
Multi-Material 3D Printing Post-Processing Toolkit
===================================================

A comprehensive Python toolkit for advanced multi-material FDM/FFF 3D printing
post-processing. Designed primarily for Bambu Studio 3MF files, this toolkit
provides tools for:

- Layer-by-layer material assignment with custom repeating patterns
- Mechanical property prediction using Classical Laminate Theory (CLT)
- Interlocking perimeter generation for improved inter-layer adhesion
- Topology-optimized variable-density infill based on stress analysis
- Wall-infill mechanical interlocking via complementary teeth geometries
- Thermal stress and warping prediction for multi-material interfaces
- Gradient material transitions for smooth property changes
- Print time and cost estimation

Modules
-------
postprocessors : G-code and 3MF post-processing tools
analysis       : Mechanical, thermal, and structural analysis
core           : Shared data structures, material database, and I/O utilities
utils          : Helper functions for geometry, G-code parsing, and visualization
"""

__version__ = "1.0.0"
__author__ = "Ioannis Christodoulou"
