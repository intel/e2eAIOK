Coding Style
============

* use pep8 python code style
`<https://www.python.org/dev/peps/pep-0008/>`_


* use yapf to do code format

.. code-block:: bash
   
   cd e2eAIOK/SDA
   yapf -i -r ./

   cd e2eAIOK/utils
   yapf -i -r ./


* Use pylint to check code Style

.. code-block:: bash
   
   cd e2eAIOK/SDA
   ./dev/lint-python

   cd e2eAIOK/utils
   ./dev/lint-python