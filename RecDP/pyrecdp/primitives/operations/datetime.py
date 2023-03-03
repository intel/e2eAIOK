from .featuretools_adaptor import FeaturetoolsOperation

class DatetimeOperation(FeaturetoolsOperation):
    def __init__(self, op_base):
        super().__init__(op_base)