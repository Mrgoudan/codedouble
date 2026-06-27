from setuptools import setup

setup(
    name="codedouble",
    version="0.1.0",
    description="Self-Learning Code Double — prototype, logger, CLI",
    packages=["codedouble"],
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    extras_require={"real": ["sentence-transformers>=2.2"]},
    entry_points={"console_scripts": ["codedouble=codedouble.cli:main"]},
)
