Coding Style
============

* use pep8 python code style
`<https://www.python.org/dev/peps/pep-0008/>`_


* use yapf to do code format

.. code-block:: bash
   
   cd SDA
   yapf -i -r ./

   cd hydroai
   yapf -i -r ./


* Use pylint to check code Style

.. code-block:: bash
   
   cd SDA
   ./dev/lint-python

   cd hydroai
   ./dev/lint-python
