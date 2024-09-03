from setuptools import setup, find_packages

setup(
    name="ao_arch",
    version="0.1.2",
    description="architecture class (config) of ao_core by aolabs.ai",
    long_description="docs.aolabs.ai",
    url="https://github.com/aolabsai/ao_arch",
    author="AO Labs",
    author_email="engineering@aolabs.ai",
    include_package_data=True,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["tests", "tests.*"]),   # maybe change this to match https://stackoverflow.com/questions/14417236/setup-py-renaming-src-package-to-project-name
    install_requires=[
        "numpy>=1.23.3",
    ],
    zip_safe=False,
    keywords=[""],
    classifiers=[
        "Private :: Do Not Upload",
        "Development Status :: 3 - Aplha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
)