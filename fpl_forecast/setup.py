from distutils.core import setup

setup(
    name="fpl_forecast",
    version="1.0",
    py_modules=["fpl_forecast"],
    install_requires=["scikit_learn", "xgboost", "pandas"],
)
