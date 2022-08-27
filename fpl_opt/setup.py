from distutils.core import setup

setup(
    name="fpl_opt",
    version="1.0",
    py_modules=["fpl_opt"],
    install_requires=["numpy", "pandas", "pulp"],
)
