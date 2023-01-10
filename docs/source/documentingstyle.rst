Documenting Style
=================

In order to pass pep8, please keep comments line-width below 72

* Class

.. code-block:: python

  class BaseModelAdvisor:
      """
      Model Advisor Base, Model Advisor is used to create w/wo sigopt
      parameter advise based on model type

      ...

      Attributes
      ----------
      params : dict
          params include dataset_path, save_path, global_configs,
          model_parameters, passed by arguments or e2eaiok_defaults.conf
      assignment_model_tracker : dict
          a tracker map of assigned_parameters and its corresponding
          model path

      """

* Method

.. code-block:: python

    def add(num1, num2):
        """
        Add up two integer numbers.

        This function simply wraps the ``+`` operator, and does not
        do anything interesting, except for illustrating what
        the docstring of a very simple function looks like.

        Parameters
        ----------
        num1 : int
            First number to add.
        num2 : int
            Second number to add.

        Returns
        -------
        int
            The sum of ``num1`` and ``num2``.

        See Also
        --------
        subtract : Subtract one integer from another.

        Examples
        --------
        >>> add(2, 2)
        4
        >>> add(25, 0)
        25
        >>> add(10, -10)
        0
        """
        return num1 + num2

`<https://pandas.pydata.org/docs/development/contributing_docstring.html>`_