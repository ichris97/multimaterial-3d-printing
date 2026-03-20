"""
Dark theme stylesheet and color palette for the GUI.
"""

DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #313244;
    color: #a6adc8;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-width: 100px;
}
QTabBar::tab:selected {
    background-color: #45475a;
    color: #cdd6f4;
    font-weight: bold;
}
QTabBar::tab:hover {
    background-color: #585b70;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    padding: 8px 18px;
    border-radius: 6px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #74c7ec;
}
QPushButton:pressed {
    background-color: #89dceb;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#dangerBtn {
    background-color: #f38ba8;
}
QPushButton#dangerBtn:hover {
    background-color: #eba0ac;
}
QPushButton#successBtn {
    background-color: #a6e3a1;
}
QPushButton#successBtn:hover {
    background-color: #94e2d5;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}
QLabel {
    color: #cdd6f4;
    background: transparent;
}
QLabel#headerLabel {
    font-size: 16px;
    font-weight: bold;
    color: #89b4fa;
}
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 20px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #585b70;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #89b4fa;
}
QSlider::groove:horizontal {
    background: #45475a;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}
QPlainTextEdit, QTextEdit {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", "Fira Code", monospace;
    font-size: 12px;
    padding: 8px;
}
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 4px;
}
QSplitter::handle {
    background-color: #45475a;
    width: 3px;
    height: 3px;
}
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
}
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}
QMenuBar::item:selected {
    background-color: #45475a;
}
QMenu {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected {
    background-color: #45475a;
}
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
    background: transparent;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #89b4fa;
    border-color: #89b4fa;
}
QScrollBar:vertical {
    background: #1e1e2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

# Named colors for consistent use across the app
COLORS = {
    'bg': '#1e1e2e',
    'surface': '#313244',
    'overlay': '#45475a',
    'text': '#cdd6f4',
    'subtext': '#a6adc8',
    'blue': '#89b4fa',
    'teal': '#94e2d5',
    'green': '#a6e3a1',
    'yellow': '#f9e2af',
    'peach': '#fab387',
    'red': '#f38ba8',
    'mauve': '#cba6f7',
    'lavender': '#b4befe',
}

# Material colors for visualization (distinct, colorblind-friendly)
MATERIAL_COLORS = [
    '#89b4fa',  # blue — material 1
    '#f38ba8',  # red — material 2
    '#a6e3a1',  # green — material 3
    '#f9e2af',  # yellow — material 4
    '#cba6f7',  # mauve — material 5
    '#94e2d5',  # teal — material 6
    '#fab387',  # peach — material 7
    '#b4befe',  # lavender — material 8
]

# Feature names and colors for G-code visualization (index-matched)
FEATURE_NAMES = [
    'Outer Wall',      # 0
    'Inner Wall',      # 1
    'Sparse Infill',   # 2
    'Solid Infill',    # 3
    'Bridge',          # 4
    'Support',         # 5
    'Overhang Wall',   # 6
    'Top Surface',     # 7
    'Bottom Surface',  # 8
    'Prime Tower',     # 9
    'Custom',          # 10
    'Skirt',           # 11
]

FEATURE_COLORS = [
    [0.94, 0.55, 0.55],  # outer wall — red
    [0.55, 0.75, 0.94],  # inner wall — blue
    [0.55, 0.94, 0.55],  # sparse infill — green
    [0.94, 0.94, 0.55],  # solid infill — yellow
    [0.94, 0.55, 0.94],  # bridge — magenta
    [0.7,  0.7,  0.7 ],  # support — gray
    [0.94, 0.75, 0.55],  # overhang wall — orange
    [0.55, 0.94, 0.94],  # top surface — cyan
    [0.75, 0.55, 0.94],  # bottom surface — purple
    [0.85, 0.85, 0.55],  # prime tower — dark yellow
    [0.60, 0.60, 0.60],  # custom — dim gray
    [0.80, 0.55, 0.70],  # skirt — pink
]

# Hex versions for Qt legend labels
FEATURE_COLORS_HEX = [
    '#f08c8c', '#8cbff0', '#8cf08c', '#f0f08c', '#f08cf0', '#b3b3b3',
    '#f0bf8c', '#8cf0f0', '#bf8cf0', '#d9d98c', '#999999', '#cc8cb3',
]
