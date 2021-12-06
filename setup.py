import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class post_install(install):
    def run(self):
        install.run(self)
        import os
        try:
            import pyspark
            vspark = str(pyspark.__version__)
        except:
            vspark = "3.2.0"
        import shutil
        scala_jar = "recdp-scala-extensions-0.1.0-jar-with-dependencies-30-spark.jar" if vspark.startswith(
            "3.0"
        ) else "recdp-scala-extensions-0.1.0-jar-with-dependencies-latest-spark.jar"
        print(
            f"cp {self.build_lib}/ScalaProcessUtils/built/{scala_jar} to {self.install_lib}/pyrecdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        )
        os.makedirs(os.path.dirname(
            f"{self.install_lib}/pyrecdp/ScalaProcessUtils/target/"),
                    exist_ok=True)
        shutil.copy(
            f"{self.build_lib}/ScalaProcessUtils/built/{scala_jar}",
            f"{self.install_lib}/pyrecdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        )


setuptools.setup(
    name="pyrecdp",
    version="0.1.1",
    author="INTEL AIA BDF",
    author_email="chendi.xue@intel.com",
    description=
    "A data processing bundle for spark based recommender system operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oap-project/recdp",
    project_urls={
        "Bug Tracker": "https://github.com/oap-project/recdp",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_dir={},
    packages=["pyrecdp", "ScalaProcessUtils"],
    package_data={"": ["*.jar"]},
    python_requires=">=3.6",
    cmdclass={'install': post_install},
    zip_safe=False,
    install_requires=['pyspark'])
