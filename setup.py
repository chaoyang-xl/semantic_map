from setuptools import find_packages, setup

package_name = "semantic_map"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="semantic_map maintainers",
    maintainer_email="maintainers@example.com",
    description="RGB-D detection projection and fusion utilities for 2D semantic maps",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_map_json_demo = semantic_map.json_demo:main",
        ],
    },
)
