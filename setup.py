import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytrends-aggregate",
    version="0.0.1",
    author="Kleber Noel",
    author_email="klebnoel@gmail.com",
    description="Aggregate calls to pytrends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klebster2/pytrends-aggregate",
    project_urls={
        "Bug Tracker": "https://github.com/klebster2/pytrends-aggregate/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License version 2",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src/*"),
    python_requires=">=3.5",
    install_requires=['pytrends'],
    entry_points={
        'console_scripts':['pytrends_aggregate=pytrends_aggregate.cli:main']
    }
)
