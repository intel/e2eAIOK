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
            print(f"Detect system pyspark, version is {vspark}")
        except:
            vspark = "3.2.0"
            print(f"Didn't find system pyspark, use default version {vspark}, other version can be found {self.install_lib}/pyrecdp/ScalaProcessUtils/")
        import shutil
        scala_jar = "/30/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar" if vspark.startswith(
            "3.0"
        ) else "/31/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        print(f"mkdir {self.install_lib}/pyrecdp/ScalaProcessUtils/target/")
        os.makedirs(os.path.dirname(
            f"{self.install_lib}/pyrecdp/ScalaProcessUtils/target/"),
                    exist_ok=True)
        print(
            f"cp {self.build_lib}/ScalaProcessUtils/built/{scala_jar} to {self.install_lib}/pyrecdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        )
        shutil.copy(
            f"{self.build_lib}/ScalaProcessUtils/built/{scala_jar}",
            f"{self.install_lib}/pyrecdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
        )
        print(f"cp {self.build_lib}/ScalaProcessUtils/built/ {self.install_lib}/pyrecdp/ScalaProcessUtils/")
        shutil.copytree(f"{self.build_lib}/ScalaProcessUtils/built/",
                        f"{self.install_lib}/pyrecdp/ScalaProcessUtils/built")


setuptools.setup(
    name="pyrecdp",
    version="0.1.2",
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
    install_requires=[])
