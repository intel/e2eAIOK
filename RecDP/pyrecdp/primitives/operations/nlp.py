from .featuretools_adaptor import FeaturetoolsOperation

class BertDecodeOperation(FeaturetoolsOperation):
    def __init__(self, op_base):        
        super().__init__(op_base)

class TextFeatureGenerator(FeaturetoolsOperation):
    def __init__(self, op_base):        
        super().__init__(op_base)