import torch
import inspect
import logging

logger = logging.getLogger(__name__)


class Sequential(torch.nn.ModuleDict):
    """A sequence of modules with potentially inferring shape on construction.

    If layers are passed with names, these can be referenced with dot notation.

    Arguments
    ---------
    input_shape : iterable
        A list or tuple of ints or None, representing the expected shape of an
        input tensor. None represents a variable-length dimension. If no
        ``input_shape`` is passed, no shape inference will be performed.
    *layers, **named_layers
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer. If a tuple is returned,
        only the shape of the first element is used to determine input
        shape of the next layer (e.g. RNN returns output, hidden).
    """

    def __init__(self, *layers, input_shape=None, **named_layers):
        super().__init__()

        # Make sure either layers or input_shape is passed
        if not layers and input_shape is None and not named_layers:
            raise ValueError("Must pass either layers or input shape")

        # Keep track of what layers need "lengths" passed
        self.length_layers = []

        # Replace None dimensions with arbitrary value
        self.input_shape = input_shape
        if input_shape and None in input_shape:
            self.input_shape = list(input_shape)
            for i, dim in enumerate(self.input_shape):

                # To reduce size of dummy tensors, use 1 for batch dim
                if i == 0 and dim is None:
                    dim = 1

                # Use 64 as nice round arbitrary value, big enough that
                # halving this dimension a few times doesn't reach 1
                self.input_shape[i] = dim or 256

        # Append non-named layers
        for layer in layers:
            self.append(layer)

        # Append named layers
        for name, layer in named_layers.items():
            self.append(layer, layer_name=name)

    def append(self, layer, *args, layer_name=None, **kwargs):
        """Add a layer to the list of layers, inferring shape if necessary.

        Arguments
        ---------
        layer : A torch.nn.Module class or object
            If the layer is a class, it should accept an argument called
            ``input_shape`` which will be inferred and passed. If the layer
            is a module object, it is added as-is.
        layer_name : str
            The name of the layer, for reference. If the name is in use,
            ``_{count}`` will be appended.
        *args, **kwargs
            These are passed to the layer if it is constructed.
        """

        # Compute layer_name
        if layer_name is None:
            layer_name = str(len(self))
        elif layer_name in self:
            index = 0
            while f"{layer_name}_{index}" in self:
                index += 1
            layer_name = f"{layer_name}_{index}"

        # Check if it needs to be constructed with input shape
        if self.input_shape:
            argspec = inspect.getfullargspec(layer)
            if "input_shape" in argspec.args + argspec.kwonlyargs:
                input_shape = self.get_output_shape()
                layer = layer(*args, input_shape=input_shape, **kwargs)

        # Finally, append the layer.
        try:
            self.add_module(layer_name, layer)
        except TypeError:
            raise ValueError(
                "Must pass `input_shape` at initialization and use "
                "modules that take `input_shape` to infer shape when "
                "using `append()`."
            )

    def get_output_shape(self):
        """Returns expected shape of the output.

        Computed by passing dummy input constructed with the
        ``self.input_shape`` attribute.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            dummy_output = self(dummy_input)
        return dummy_output.shape

    def forward(self, x):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.
        """
        for layer in self.values():
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x


class ModuleList(torch.nn.Module):
    """This class implements a wrapper to torch.nn.ModuleList with a forward()
    method to forward all the layers sequentially.

    Arguments
    ---------
    *layers : torch class
        Torch objects to be put in a ModuleList.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        """Applies the computation pipeline."""
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def append(self, module):
        """Appends module to the layers list."""
        self.layers.append(module)

    def extend(self, modules):
        """Appends module to the layers list."""
        self.layers.extend(modules)

    def insert(self, index, module):
        """Inserts module to the layers list."""
        self.layers.insert(module)
