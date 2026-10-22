import sys

from PySide6.QtWidgets import QApplication

from ui.app_window import MainWindow, load_stylesheet


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()