"""
Main GUI application for the Multi-Material 3D Printing Toolkit.

Provides a tabbed interface with:
- 3D viewer with before/after comparison
- Layer Pattern tool with material analysis
- Interlocking Perimeters tool
- Topology-Optimized Infill tool
- Wall-Infill Interlocking tool
- Analysis dashboard (mechanical + thermal)
"""

import sys
import os
import io
import zipfile
import traceback
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QGroupBox, QLabel, QPushButton, QComboBox,
    QLineEdit, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
    QPlainTextEdit, QFileDialog, QStatusBar, QMenuBar, QMenu,
    QProgressBar, QFrame, QSizePolicy, QMessageBox, QGridLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QFont

from .theme import (
    DARK_STYLESHEET, COLORS, MATERIAL_COLORS,
    FEATURE_NAMES, FEATURE_COLORS_HEX,
)
from .viewer_3d import (
    load_mesh_from_3mf, color_mesh_by_layers, parse_gcode_paths
)


class WorkerThread(QThread):
    """Background thread for running post-processing operations."""
    finished = Signal(str)   # result message
    error = Signal(str)      # error message
    log = Signal(str)        # log output

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.func(*self.args, **self.kwargs)
            self.log.emit(buf.getvalue())
            self.finished.emit("Operation completed successfully.")
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ViewerPanel(QWidget):
    """Side-by-side 3D viewer with before/after comparison and legends."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()
        self.label_before = QLabel("Original")
        self.label_before.setObjectName("headerLabel")
        self.label_after = QLabel("Modified")
        self.label_after.setObjectName("headerLabel")
        header.addWidget(self.label_before)
        header.addWidget(self.label_after)
        layout.addLayout(header)

        # 3D viewports
        splitter = QSplitter(Qt.Horizontal)

        self.plotter_before = QtInteractor(splitter)
        self.plotter_after = QtInteractor(splitter)

        splitter.addWidget(self.plotter_before)
        splitter.addWidget(self.plotter_after)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # Legend area (below viewports, above slider)
        self.legend_widget = QWidget()
        self.legend_layout = QHBoxLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(4, 2, 4, 2)
        self.legend_layout.setSpacing(12)
        self.legend_widget.setVisible(False)
        layout.addWidget(self.legend_widget)

        # Layer slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Layer:"))
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setRange(0, 100)
        self.layer_slider.setValue(100)
        slider_layout.addWidget(self.layer_slider, stretch=1)
        self.layer_label = QLabel("All")
        self.layer_label.setMinimumWidth(60)
        slider_layout.addWidget(self.layer_label)
        layout.addLayout(slider_layout)

        self._mesh_before = None
        self._mesh_after = None
        self._gcode_mesh = None
        self._max_layer = 100

        self.layer_slider.valueChanged.connect(self._on_layer_change)

        # Initialize plotters
        for p in [self.plotter_before, self.plotter_after]:
            p.set_background('#181825')
            p.add_text("Import a 3MF file to begin",
                       position='upper_left', font_size=10, color='#6c7086')

    def _set_legend(self, legend_type: str, items: dict = None):
        """Update the legend bar below the viewports.

        legend_type: 'layer', 'material', 'gcode_feature', 'none'
        items: dict of {label: color_hex} for material legend
        """
        # Clear existing legend items
        while self.legend_layout.count():
            child = self.legend_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if legend_type == 'none':
            self.legend_widget.setVisible(False)
            return

        self.legend_widget.setVisible(True)

        if legend_type == 'layer':
            title = QLabel("Color: Layer height (blue=bottom, red=top)")
            title.setStyleSheet("color: #a6adc8; font-size: 11px; background: transparent;")
            self.legend_layout.addWidget(title)
            self.legend_layout.addStretch()

        elif legend_type == 'material' and items:
            title = QLabel("Materials:")
            title.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold; background: transparent;")
            self.legend_layout.addWidget(title)
            for label, color in items.items():
                swatch = QLabel("  ")
                swatch.setFixedSize(14, 14)
                swatch.setStyleSheet(
                    f"background-color: {color}; border-radius: 3px; border: 1px solid #585b70;")
                name = QLabel(label)
                name.setStyleSheet("color: #cdd6f4; font-size: 11px; background: transparent;")
                self.legend_layout.addWidget(swatch)
                self.legend_layout.addWidget(name)
            self.legend_layout.addStretch()

        elif legend_type == 'gcode_feature':
            title = QLabel("G-code Features:")
            title.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold; background: transparent;")
            self.legend_layout.addWidget(title)
            # Only show features that are present in current gcode
            present = set()
            if self._gcode_mesh is not None and 'feature' in self._gcode_mesh.cell_data:
                present = set(self._gcode_mesh.cell_data['feature'])
            for idx in sorted(present):
                if idx < len(FEATURE_NAMES):
                    swatch = QLabel("  ")
                    swatch.setFixedSize(14, 14)
                    swatch.setStyleSheet(
                        f"background-color: {FEATURE_COLORS_HEX[idx]}; "
                        f"border-radius: 3px; border: 1px solid #585b70;")
                    name = QLabel(FEATURE_NAMES[idx])
                    name.setStyleSheet("color: #cdd6f4; font-size: 11px; background: transparent;")
                    self.legend_layout.addWidget(swatch)
                    self.legend_layout.addWidget(name)
            self.legend_layout.addStretch()

    def show_mesh(self, mesh: pv.PolyData, target='before', scalars=None,
                  cmap=None, title=None, material_legend: dict = None):
        """Display a mesh in the before or after viewport."""
        plotter = self.plotter_before if target == 'before' else self.plotter_after
        plotter.clear()
        plotter.set_background('#181825')

        if mesh is None:
            return

        kwargs = {'show_edges': False, 'smooth_shading': True}
        if scalars and scalars in mesh.cell_data:
            kwargs['scalars'] = scalars
            kwargs['cmap'] = cmap or 'turbo'
            kwargs['show_scalar_bar'] = False  # We use our own legend instead
        else:
            kwargs['color'] = '#b4befe'

        plotter.add_mesh(mesh, **kwargs)
        if title:
            plotter.add_text(title, position='upper_left', font_size=10,
                             color='#cdd6f4')
        plotter.reset_camera()

        if target == 'before':
            self._mesh_before = mesh
        else:
            self._mesh_after = mesh

        # Update layer slider range
        if mesh.n_cells > 0 and 'layer' in mesh.cell_data:
            self._max_layer = int(mesh.cell_data['layer'].max())
            self.layer_slider.setRange(0, self._max_layer)
            self.layer_slider.setValue(self._max_layer)

        # Set legend
        if material_legend:
            self._set_legend('material', material_legend)
        elif scalars == 'layer':
            self._set_legend('layer')

    def show_gcode(self, gcode_mesh: pv.PolyData, target='after', title=None):
        """Display parsed G-code paths in a viewport with feature legend."""
        plotter = self.plotter_after if target == 'after' else self.plotter_before
        plotter.clear()
        plotter.set_background('#181825')

        if gcode_mesh is None:
            return

        self._gcode_mesh = gcode_mesh
        if target == 'after':
            self._mesh_after = None  # Clear mesh reference so slider uses gcode
        else:
            self._mesh_before = None

        if 'layer' in gcode_mesh.cell_data:
            self._max_layer = int(gcode_mesh.cell_data['layer'].max())
            self.layer_slider.setRange(0, self._max_layer)
            self.layer_slider.setValue(self._max_layer)

        plotter.add_mesh(gcode_mesh, scalars='feature', cmap=FEATURE_COLORS_HEX,
                         line_width=1.5, render_lines_as_tubes=False,
                         show_scalar_bar=False)
        plotter.add_text(title or "G-code Preview", position='upper_left',
                         font_size=10, color='#cdd6f4')
        plotter.reset_camera()

        # Show feature legend
        self._set_legend('gcode_feature')

    def _on_layer_change(self, value):
        """Filter displayed geometry by layer."""
        if value >= self._max_layer:
            self.layer_label.setText("All")
        else:
            self.layer_label.setText(str(value))

        # Re-render meshes with layer filter
        for mesh, plotter, label in [
            (self._mesh_before, self.plotter_before, "Original"),
            (self._mesh_after, self.plotter_after, "Modified"),
        ]:
            if mesh is not None and 'layer' in mesh.cell_data:
                plotter.clear()
                plotter.set_background('#181825')
                mask = mesh.cell_data['layer'] <= value
                if mask.any():
                    filtered = mesh.extract_cells(np.where(mask)[0])
                    kwargs = {'smooth_shading': True}
                    if 'material' in filtered.cell_data:
                        kwargs['scalars'] = 'material'
                        kwargs['cmap'] = MATERIAL_COLORS[:8]
                        kwargs['show_scalar_bar'] = False
                    elif 'layer' in filtered.cell_data:
                        kwargs['scalars'] = 'layer'
                        kwargs['cmap'] = 'turbo'
                        kwargs['show_scalar_bar'] = False
                    else:
                        kwargs['color'] = '#b4befe'
                    plotter.add_mesh(filtered, **kwargs)
                    plotter.add_text(label, position='upper_left',
                                     font_size=10, color='#cdd6f4')

        # Re-render G-code with layer filter
        if self._gcode_mesh is not None and 'layer' in self._gcode_mesh.cell_data:
            # Determine which plotter has the gcode
            if self._mesh_after is None:
                plotter = self.plotter_after
                label = "G-code"
            elif self._mesh_before is None:
                plotter = self.plotter_before
                label = "G-code"
            else:
                return  # Both have meshes, gcode not shown

            plotter.clear()
            plotter.set_background('#181825')
            mask = self._gcode_mesh.cell_data['layer'] <= value
            if mask.any():
                filtered = self._gcode_mesh.extract_cells(np.where(mask)[0])
                plotter.add_mesh(filtered, scalars='feature', cmap=FEATURE_COLORS_HEX,
                                 line_width=1.5, show_scalar_bar=False)
                plotter.add_text(label, position='upper_left',
                                 font_size=10, color='#cdd6f4')

    def close(self):
        self.plotter_before.close()
        self.plotter_after.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tool panels — one per post-processing operation
# ─────────────────────────────────────────────────────────────────────────────

class LayerPatternPanel(QWidget):
    """Controls for the layer pattern assignment tool."""
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Layer Pattern Settings")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Pattern:"), 0, 0)
        self.pattern_edit = QLineEdit("F1:3,F2:2")
        self.pattern_edit.setPlaceholderText("e.g., F1:3,F2:2 or 1,2,2,1")
        g.addWidget(self.pattern_edit, 0, 1, 1, 2)

        g.addWidget(QLabel("Layer height (mm):"), 1, 0)
        self.layer_height = QDoubleSpinBox()
        self.layer_height.setRange(0.04, 0.6)
        self.layer_height.setValue(0.20)
        self.layer_height.setSingleStep(0.02)
        self.layer_height.setDecimals(2)
        g.addWidget(self.layer_height, 1, 1)

        g.addWidget(QLabel("Total height (mm):"), 2, 0)
        self.total_height = QDoubleSpinBox()
        self.total_height.setRange(0.1, 500)
        self.total_height.setValue(20.0)
        self.total_height.setSingleStep(1.0)
        g.addWidget(self.total_height, 2, 1)

        g.addWidget(QLabel("Material 1:"), 3, 0)
        self.mat1_combo = self._material_combo()
        g.addWidget(self.mat1_combo, 3, 1)

        g.addWidget(QLabel("Material 2:"), 4, 0)
        self.mat2_combo = self._material_combo()
        self.mat2_combo.setCurrentIndex(3)  # TPU
        g.addWidget(self.mat2_combo, 4, 1)

        self.analyze_check = QCheckBox("Run mechanical + thermal analysis")
        self.analyze_check.setChecked(True)
        g.addWidget(self.analyze_check, 5, 0, 1, 3)

        layout.addWidget(grp)

        btn = QPushButton("Apply Pattern")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _material_combo(self):
        combo = QComboBox()
        from ..core.materials import MATERIAL_DB
        for key in sorted(MATERIAL_DB.keys()):
            combo.addItem(f"{key} - {MATERIAL_DB[key].name}", key)
        return combo

    def _on_run(self):
        self.run_requested.emit({
            'pattern': self.pattern_edit.text(),
            'layer_height': self.layer_height.value(),
            'total_height': self.total_height.value(),
            'mat1': self.mat1_combo.currentData(),
            'mat2': self.mat2_combo.currentData(),
            'analyze': self.analyze_check.isChecked(),
        })


class InterlockPerimPanel(QWidget):
    """Controls for interlocking perimeters."""
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Interlocking Perimeters Settings")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Z offset (fraction of layer height):"), 0, 0)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(0.1, 0.9)
        self.offset.setValue(0.5)
        self.offset.setSingleStep(0.1)
        g.addWidget(self.offset, 0, 1)

        g.addWidget(QLabel("Walls to offset:"), 1, 0)
        self.walls_edit = QLineEdit("2")
        self.walls_edit.setPlaceholderText("e.g., 2 or 1,3")
        g.addWidget(self.walls_edit, 1, 1)

        g.addWidget(QLabel("Start from layer:"), 2, 0)
        self.min_layer = QSpinBox()
        self.min_layer.setRange(1, 50)
        self.min_layer.setValue(3)
        g.addWidget(self.min_layer, 2, 1)

        layout.addWidget(grp)

        btn = QPushButton("Apply Interlocking")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit({
            'offset': self.offset.value(),
            'walls': self.walls_edit.text(),
            'min_layer': self.min_layer.value(),
        })


class TopologyInfillPanel(QWidget):
    """Controls for topology-optimized infill."""
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Topology-Optimized Infill Settings")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Min density (%):"), 0, 0)
        self.min_density = QSpinBox()
        self.min_density.setRange(5, 100)
        self.min_density.setValue(15)
        g.addWidget(self.min_density, 0, 1)

        g.addWidget(QLabel("Max density (%):"), 1, 0)
        self.max_density = QSpinBox()
        self.max_density.setRange(10, 100)
        self.max_density.setValue(80)
        g.addWidget(self.max_density, 1, 1)

        g.addWidget(QLabel("Sensitivity (0-1):"), 2, 0)
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.0, 1.0)
        self.sensitivity.setValue(0.5)
        self.sensitivity.setSingleStep(0.1)
        g.addWidget(self.sensitivity, 2, 1)

        g.addWidget(QLabel("Corner radius (mm):"), 3, 0)
        self.corner_radius = QDoubleSpinBox()
        self.corner_radius.setRange(1, 30)
        self.corner_radius.setValue(5.0)
        g.addWidget(self.corner_radius, 3, 1)

        layout.addWidget(grp)

        btn = QPushButton("Optimize Infill")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit({
            'min_density': self.min_density.value(),
            'max_density': self.max_density.value(),
            'sensitivity': self.sensitivity.value(),
            'corner_radius': self.corner_radius.value(),
        })


class WallInfillInterlockPanel(QWidget):
    """Controls for wall-infill interlocking."""
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Wall-Infill Interlocking Settings")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Teeth depth (mm):"), 0, 0)
        self.teeth_depth = QDoubleSpinBox()
        self.teeth_depth.setRange(0.1, 1.0)
        self.teeth_depth.setValue(0.4)
        self.teeth_depth.setSingleStep(0.05)
        g.addWidget(self.teeth_depth, 0, 1)

        g.addWidget(QLabel("Teeth pitch (mm):"), 1, 0)
        self.teeth_pitch = QDoubleSpinBox()
        self.teeth_pitch.setRange(0.5, 5.0)
        self.teeth_pitch.setValue(1.5)
        self.teeth_pitch.setSingleStep(0.1)
        g.addWidget(self.teeth_pitch, 1, 1)

        g.addWidget(QLabel("Interlock zone (mm):"), 2, 0)
        self.interlock_length = QDoubleSpinBox()
        self.interlock_length.setRange(1.0, 10.0)
        self.interlock_length.setValue(3.0)
        g.addWidget(self.interlock_length, 2, 1)

        layout.addWidget(grp)

        btn = QPushButton("Apply Interlocking")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit({
            'teeth_depth': self.teeth_depth.value(),
            'teeth_pitch': self.teeth_pitch.value(),
            'interlock_length': self.interlock_length.value(),
        })


class AnalysisPanel(QWidget):
    """Controls for standalone mechanical + thermal analysis."""
    run_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        grp = QGroupBox("Layup Analysis (no file needed)")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Pattern:"), 0, 0)
        self.pattern_edit = QLineEdit("F1:5,F2:10,F1:5")
        g.addWidget(self.pattern_edit, 0, 1, 1, 2)

        g.addWidget(QLabel("Layer height (mm):"), 1, 0)
        self.layer_height = QDoubleSpinBox()
        self.layer_height.setRange(0.04, 0.6)
        self.layer_height.setValue(0.20)
        self.layer_height.setSingleStep(0.02)
        g.addWidget(self.layer_height, 1, 1)

        g.addWidget(QLabel("Total height (mm):"), 2, 0)
        self.total_height = QDoubleSpinBox()
        self.total_height.setRange(0.1, 500)
        self.total_height.setValue(4.0)
        g.addWidget(self.total_height, 2, 1)

        g.addWidget(QLabel("Material 1:"), 3, 0)
        self.mat1_combo = LayerPatternPanel._material_combo(self)
        g.addWidget(self.mat1_combo, 3, 1)

        g.addWidget(QLabel("Material 2:"), 4, 0)
        self.mat2_combo = LayerPatternPanel._material_combo(self)
        self.mat2_combo.setCurrentIndex(3)
        g.addWidget(self.mat2_combo, 4, 1)

        g.addWidget(QLabel("Part length (mm):"), 5, 0)
        self.part_length = QDoubleSpinBox()
        self.part_length.setRange(10, 500)
        self.part_length.setValue(100.0)
        g.addWidget(self.part_length, 5, 1)

        layout.addWidget(grp)

        btn = QPushButton("Run Analysis")
        btn.clicked.connect(self._on_run)
        layout.addWidget(btn)
        layout.addStretch()

    def _on_run(self):
        self.run_requested.emit({
            'pattern': self.pattern_edit.text(),
            'layer_height': self.layer_height.value(),
            'total_height': self.total_height.value(),
            'mat1': self.mat1_combo.currentData(),
            'mat2': self.mat2_combo.currentData(),
            'part_length': self.part_length.value(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Material 3D Printing Toolkit")
        self.resize(1500, 900)

        self._input_path = None
        self._output_path = None
        self._worker = None

        self._build_menubar()
        self._build_ui()
        self._build_statusbar()

    def _build_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        import_action = QAction("Import 3MF...", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        export_action = QAction("Export Result...", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Left: control panel with tabs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Import/Export buttons at top
        btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("Import 3MF")
        self.import_btn.clicked.connect(self._on_import)
        btn_layout.addWidget(self.import_btn)

        self.export_btn = QPushButton("Export Result")
        self.export_btn.setObjectName("successBtn")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)
        left_layout.addLayout(btn_layout)

        # File info
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #6c7086; padding: 4px;")
        left_layout.addWidget(self.file_label)

        # Tool tabs
        self.tool_tabs = QTabWidget()
        self.tool_tabs.setTabPosition(QTabWidget.North)

        self.layer_panel = LayerPatternPanel()
        self.layer_panel.run_requested.connect(self._run_layer_pattern)
        self.tool_tabs.addTab(self.layer_panel, "Layers")

        self.interlock_panel = InterlockPerimPanel()
        self.interlock_panel.run_requested.connect(self._run_interlock_perimeters)
        self.tool_tabs.addTab(self.interlock_panel, "Interlock")

        self.topology_panel = TopologyInfillPanel()
        self.topology_panel.run_requested.connect(self._run_topology_infill)
        self.tool_tabs.addTab(self.topology_panel, "Topology")

        self.wall_panel = WallInfillInterlockPanel()
        self.wall_panel.run_requested.connect(self._run_wall_infill)
        self.tool_tabs.addTab(self.wall_panel, "Teeth")

        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.run_requested.connect(self._run_analysis)
        self.tool_tabs.addTab(self.analysis_panel, "Analysis")

        left_layout.addWidget(self.tool_tabs)

        # Log console
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console.setPlaceholderText("Output log...")
        left_layout.addWidget(self.console)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(450)

        # Right: 3D viewer
        self.viewer = ViewerPanel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.viewer, stretch=1)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Import a 3MF file to begin.")

    def _log(self, text: str):
        self.console.appendPlainText(text.rstrip())
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())

    # ── File operations ──────────────────────────────────────────────────

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import 3MF File", "",
            "3MF Files (*.3mf);;All Files (*)")
        if not path:
            return

        self._input_path = path
        self.file_label.setText(f"{Path(path).name}")
        self.status.showMessage(f"Loading {Path(path).name}...")
        self._log(f"Loading: {path}")

        try:
            # Try to load mesh geometry
            mesh = None
            try:
                mesh = load_mesh_from_3mf(path)
                self._log(f"  Mesh: {mesh.n_points} vertices, {mesh.n_cells} faces")
            except Exception:
                self._log("  No mesh geometry in file (sliced-only 3MF).")

            # Check for G-code
            gcode_content = None
            gcode_path_in_zip = None
            with zipfile.ZipFile(path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.gcode'):
                        gcode_content = zf.read(name).decode('utf-8')
                        gcode_path_in_zip = name
                        self._log(f"  G-code found: {name}")
                        break

            lh = self.layer_panel.layer_height.value()

            if mesh is not None and mesh.n_cells > 0:
                # Auto-detect height
                z_range = mesh.bounds[5] - mesh.bounds[4]
                self.layer_panel.total_height.setValue(round(z_range, 1))
                self.analysis_panel.total_height.setValue(round(z_range, 1))

                # Color by layer and show in left viewport
                mesh = color_mesh_by_layers(mesh, lh)
                self.viewer.show_mesh(mesh, target='before', scalars='layer',
                                      cmap='turbo', title='Original Model')

                if gcode_content:
                    # Show G-code in right viewport
                    gcode_mesh = parse_gcode_paths(gcode_content, max_layers=150)
                    if gcode_mesh:
                        self.viewer.show_gcode(gcode_mesh, target='after',
                                               title='Sliced G-code')
                        self._log(f"  G-code: {gcode_mesh.n_cells} path segments")
                else:
                    # No G-code: mirror mesh in right viewport
                    self._log("  No G-code (unsliced). Layer Pattern and Analysis available.")
                    self.viewer.show_mesh(mesh.copy(), target='after', scalars='layer',
                                          cmap='turbo', title='Preview')

                self.status.showMessage(
                    f"Loaded: {Path(path).name} ({mesh.n_cells} faces)")

            elif gcode_content:
                # Gcode-only 3MF (no mesh geometry) — show G-code in BOTH viewports
                self._log("  Sliced-only file: showing G-code paths.")
                gcode_mesh = parse_gcode_paths(gcode_content, max_layers=150)
                if gcode_mesh:
                    # Detect height from G-code Z range
                    if 'layer' in gcode_mesh.cell_data:
                        z_vals = gcode_mesh.points[:, 2]
                        z_range = z_vals.max() - z_vals.min()
                        self.layer_panel.total_height.setValue(round(z_range, 1))
                        self.analysis_panel.total_height.setValue(round(z_range, 1))

                    self.viewer.show_gcode(gcode_mesh, target='before',
                                           title='Original G-code')
                    self.viewer.show_gcode(gcode_mesh, target='after',
                                           title='G-code (apply a tool to modify)')
                    self._log(f"  G-code: {gcode_mesh.n_cells} path segments, "
                              f"{int(gcode_mesh.cell_data['layer'].max())} layers")
                    self.status.showMessage(
                        f"Loaded: {Path(path).name} (sliced, "
                        f"{gcode_mesh.n_cells} G-code segments)")
                else:
                    self._log("  ERROR: Could not parse G-code paths.")
                    self.status.showMessage("Error: No parseable content in file.")
            else:
                self._log("  ERROR: File contains neither mesh nor G-code.")
                self.status.showMessage("Error: Empty 3MF file.")
                QMessageBox.warning(self, "Import Error",
                    "This 3MF file contains no mesh geometry and no G-code.\n"
                    "Please use a valid model or sliced file.")

        except Exception as e:
            self._log(f"Error loading file: {e}")
            self.status.showMessage(f"Error: {e}")
            QMessageBox.warning(self, "Import Error", str(e))

    def _on_export(self):
        if not self._output_path or not Path(self._output_path).exists():
            QMessageBox.information(self, "Export", "No processed file to export. Run a tool first.")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Result", str(Path(self._output_path).name),
            "3MF Files (*.3mf);;All Files (*)")
        if not dest:
            return

        import shutil
        shutil.copy2(self._output_path, dest)
        self._log(f"Exported to: {dest}")
        self.status.showMessage(f"Saved: {Path(dest).name}")

    def _on_about(self):
        QMessageBox.about(self, "About",
            "Multi-Material 3D Printing Toolkit v1.2\n\n"
            "Post-processing and analysis for Bambu Studio 3MF files.\n\n"
            "Features:\n"
            "  - Layer-by-layer material assignment\n"
            "  - Interlocking perimeters\n"
            "  - Topology-optimized infill\n"
            "  - Wall-infill interlocking\n"
            "  - CLT mechanical analysis\n"
            "  - Thermal stress prediction\n\n"
            "Built with PySide6 + PyVista")

    # ── Tool runners ─────────────────────────────────────────────────────

    def _ensure_input(self, need_gcode=False):
        if not self._input_path:
            QMessageBox.warning(self, "No File", "Please import a 3MF file first.")
            return False
        if need_gcode:
            with zipfile.ZipFile(self._input_path, 'r') as zf:
                if not any(n.endswith('.gcode') for n in zf.namelist()):
                    QMessageBox.warning(self, "No G-code",
                        "This tool requires a sliced 3MF (with G-code).\n"
                        "Slice your model in Bambu Studio first.")
                    return False
        return True

    def _make_output_path(self, suffix: str) -> str:
        import tempfile
        base = Path(self._input_path).stem
        self._output_path = str(Path(tempfile.gettempdir()) / f"{base}_{suffix}.3mf")
        return self._output_path

    def _start_progress(self, msg: str):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.status.showMessage(msg)
        self.console.appendPlainText(f"\n--- {msg} ---")

    def _finish_operation(self, msg: str):
        self.progress.setVisible(False)
        self.export_btn.setEnabled(True)
        self.status.showMessage(msg)
        self._log(msg)

        # Reload and display result
        if self._output_path and Path(self._output_path).exists():
            try:
                # First try to show G-code visualization (for tools that
                # modify G-code, the output 3MF may not have mesh changes
                # but will have modified G-code paths)
                shown = False
                with zipfile.ZipFile(self._output_path, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.gcode'):
                            gcode_content = zf.read(name).decode('utf-8')
                            gcode_mesh = parse_gcode_paths(
                                gcode_content, max_layers=100
                            )
                            if gcode_mesh:
                                self.viewer.show_gcode(
                                    gcode_mesh, target='after',
                                    title='Modified G-code'
                                )
                                shown = True
                            break

                if not shown:
                    # Fall back to mesh visualization
                    result_mesh = load_mesh_from_3mf(self._output_path)
                    lh = self.layer_panel.layer_height.value()
                    result_mesh = color_mesh_by_layers(result_mesh, lh)
                    self.viewer.show_mesh(result_mesh, target='after',
                                          scalars='layer', cmap='turbo',
                                          title='Result')
            except Exception as e:
                self._log(f"  Could not preview result: {e}")

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        self.status.showMessage("Error!")
        self._log(f"ERROR: {msg}")

    def _run_layer_pattern(self, params):
        if not self._ensure_input():
            return

        self._start_progress("Applying layer pattern...")
        output = self._make_output_path("layered")

        from ..postprocessors.layer_pattern import parse_pattern, generate_layer_ranges_xml
        from ..core.file_io import inject_layer_config

        try:
            pattern = parse_pattern(params['pattern'])
            self._log(f"Pattern: {pattern}")

            xml = generate_layer_ranges_xml(1, params['layer_height'],
                                            params['total_height'], pattern)
            inject_layer_config(self._input_path, output, xml)
            self._log(f"Created: {output}")

            # Show colored mesh with material legend
            try:
                mesh = load_mesh_from_3mf(self._input_path)
            except Exception:
                mesh = None

            material_map = {1: params['mat1'], 2: params['mat2']}

            if mesh is not None:
                mesh = color_mesh_by_layers(mesh, params['layer_height'],
                                            pattern=pattern, material_map=material_map)
                # Build legend: {display_name: color_hex}
                from ..core.materials import get_material
                legend_items = {}
                unique_mats = sorted(set(pattern))
                for i, f in enumerate(unique_mats):
                    mat_key = material_map.get(f, 'PLA')
                    mat = get_material(mat_key)
                    color = MATERIAL_COLORS[i % len(MATERIAL_COLORS)]
                    legend_items[f"F{f}: {mat.name}"] = color

                self.viewer.show_mesh(mesh, target='after', scalars='material',
                                      cmap=MATERIAL_COLORS[:len(unique_mats)],
                                      title=f"Pattern: {params['pattern']}",
                                      material_legend=legend_items)
            else:
                self._log("  No mesh to visualize (sliced-only file).")

            if params['analyze']:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    from ..analysis.mechanical import analyze_layup
                    from ..analysis.thermal import thermal_stress_analysis
                    analyze_layup(pattern, params['total_height'],
                                  params['layer_height'], material_map)
                    thermal_stress_analysis(pattern, params['layer_height'],
                                            params['total_height'], material_map)
                self._log(buf.getvalue())

            self._output_path = output
            self._finish_operation("Layer pattern applied successfully.")

        except Exception as e:
            self._on_error(f"{e}\n{traceback.format_exc()}")

    def _run_interlock_perimeters(self, params):
        if not self._ensure_input(need_gcode=True):
            return

        self._start_progress("Applying interlocking perimeters...")
        output = self._make_output_path("interlocked")

        try:
            from ..core.file_io import extract_gcode_from_3mf, repack_3mf
            from ..postprocessors.interlocking_perimeters import (
                parse_gcode_content, generate_output, detect_layer_height
            )

            gcode, gcode_path = extract_gcode_from_3mf(self._input_path)
            lines = gcode.splitlines(keepends=True)
            header, layers, footer = parse_gcode_content(lines)
            lh = detect_layer_height(gcode)
            walls = [int(w.strip()) for w in params['walls'].split(',')]

            buf = io.StringIO()
            with redirect_stdout(buf):
                out_lines = generate_output(header, layers, footer, walls,
                                            params['offset'], lh,
                                            params['min_layer'], True)
            self._log(buf.getvalue())

            new_gcode = ''.join(out_lines)
            repack_3mf(self._input_path, output, new_gcode, gcode_path)

            self._output_path = output
            self._finish_operation("Interlocking perimeters applied.")

        except Exception as e:
            self._on_error(f"{e}\n{traceback.format_exc()}")

    def _run_topology_infill(self, params):
        if not self._ensure_input(need_gcode=True):
            return

        self._start_progress("Optimizing infill density...")
        output = self._make_output_path("topology")

        try:
            from ..core.file_io import extract_gcode_from_3mf, repack_3mf
            from ..postprocessors.topology_infill import (
                extract_geometry_from_gcode, calculate_stress_map,
                modify_infill_density
            )

            gcode, gcode_path = extract_gcode_from_3mf(self._input_path)
            xy_points, z_heights = extract_geometry_from_gcode(gcode)

            buf = io.StringIO()
            with redirect_stdout(buf):
                stress_map, metadata = calculate_stress_map(
                    xy_points, corner_radius=params['corner_radius'],
                    sensitivity=params['sensitivity'], verbose=True)

                gcode_lines = gcode.splitlines(keepends=True)
                modified = modify_infill_density(
                    gcode_lines, stress_map, metadata,
                    params['min_density'], params['max_density'], verbose=True)
            self._log(buf.getvalue())

            new_gcode = ''.join(modified)
            repack_3mf(self._input_path, output, new_gcode, gcode_path)

            self._output_path = output
            self._finish_operation("Topology-optimized infill applied.")

        except Exception as e:
            self._on_error(f"{e}\n{traceback.format_exc()}")

    def _run_wall_infill(self, params):
        if not self._ensure_input(need_gcode=True):
            return

        self._start_progress("Applying wall-infill interlocking...")
        output = self._make_output_path("teeth")

        try:
            from ..core.file_io import extract_gcode_from_3mf, repack_3mf
            from ..postprocessors.wall_infill_interlock import process_gcode

            gcode, gcode_path = extract_gcode_from_3mf(self._input_path)

            buf = io.StringIO()
            with redirect_stdout(buf):
                new_gcode, stats = process_gcode(
                    gcode, params['teeth_depth'], params['teeth_pitch'],
                    params['interlock_length'], 3, 8.0, True)
            self._log(buf.getvalue())
            self._log(f"Wall segments: {stats.wall_segments_modified}, "
                      f"Infill lines: {stats.infill_lines_modified}")

            repack_3mf(self._input_path, output, new_gcode, gcode_path)

            self._output_path = output
            self._finish_operation("Wall-infill interlocking applied.")

        except Exception as e:
            self._on_error(f"{e}\n{traceback.format_exc()}")

    def _run_analysis(self, params):
        self._start_progress("Running analysis...")

        try:
            from ..postprocessors.layer_pattern import parse_pattern
            from ..analysis.mechanical import analyze_layup
            from ..analysis.thermal import thermal_stress_analysis, predict_warping

            pattern = parse_pattern(params['pattern'])
            material_map = {1: params['mat1'], 2: params['mat2']}

            buf = io.StringIO()
            with redirect_stdout(buf):
                analyze_layup(pattern, params['total_height'],
                              params['layer_height'], material_map)
                thermal_stress_analysis(pattern, params['layer_height'],
                                        params['total_height'], material_map)
                warp = predict_warping(pattern, params['layer_height'],
                                       params['total_height'], material_map,
                                       part_length=params['part_length'])
                print(f"\n  Warping for {params['part_length']}mm part: "
                      f"{warp['deflection']:.3f} mm")
            self._log(buf.getvalue())

            self.progress.setVisible(False)
            self.status.showMessage("Analysis complete.")

        except Exception as e:
            self._on_error(f"{e}\n{traceback.format_exc()}")

    def closeEvent(self, event):
        self.viewer.close()
        super().closeEvent(event)


def main():
    """Launch the GUI application."""
    # PyVista settings for Qt integration
    pv.global_theme.allow_empty_mesh = True

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
