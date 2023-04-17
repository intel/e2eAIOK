import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class post_install(install):
    def run(self):
        install.run(self)
        import os
        try:
            os.system("DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre")
        except:
            print("failed to install openjdk, need manual setup, cmdline is DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre")

        import shutil
        scala_jar = "/31/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
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
    version="0.1.5",
    author="INTEL AIA BDF",
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
    install_requires=[
        "pyspark==3.3.1",
        "pyarrow",
        "psutil"
    ])
