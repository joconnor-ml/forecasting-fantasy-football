from setuptools import setup, find_packages

setup(
    name="fpl_forecast",
    version="1.0",
    install_requires=["scikit_learn", "xgboost", "pandas"],
    packages=find_packages(),
)
