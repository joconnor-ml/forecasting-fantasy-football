from setuptools import setup, find_packages

setup(
    name="fpl_opt",
    version="1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pulp"],
)
