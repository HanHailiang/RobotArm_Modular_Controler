APP_STYLE = """
QMainWindow { background-color: #f4f6f8; }
QGroupBox {
    font-weight: bold;
    border: 1px solid #c6cbd1;
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px;
    background-color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QLabel { font-size: 13px; }
QDoubleSpinBox, QSpinBox {
    min-height: 28px;
    font-size: 13px;
    min-width: 90px;
}
QPushButton {
    min-height: 30px;
    padding: 4px 10px;
    border-radius: 6px;
    background-color: #e8edf3;
}
QPushButton:hover { background-color: #dce6f2; }
QCheckBox { font-size: 13px; }
"""
