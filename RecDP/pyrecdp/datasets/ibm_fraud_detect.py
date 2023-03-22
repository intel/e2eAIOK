from .base_api import base_api

class ibm_fraud_detect(base_api):
    def __init__(self, scale = 'full'):
        super().__init__()
        if scale == 'test':
            raise NotImplementedError("ibm_fraud_detect test dataset is not created yet")
        else:
            url = "https://dl2.boxcloud.com/d/1/b1!L_cviq5iPrNN9x2OlVJEvb1Y8i_Gn_orlD36V8uBoZXoGOG_Kve3QE1YrIIUX6RaAU7j7Ng67lp4JG80zUqP2J9Cfy8fWJ0JFkTdE7ArFW6wDpzi4F2qNa5l0SFDW4PYvruXs_CHaJwLXfwrs7jO2s5EIieKsR-b_mFLZoTVEmmaqjKWTUVNycGZxvDIGJnyMRSR8eV-g7O6JDDGgws51SVJhxlq_kfrGPp1f6p6bTo4XIq9hdanhH0Q7h32s_0pD-znzoN2RFpv8omQRLDvSMNZz2S7HkURrR20edhiCInMBM6oUt3ImtwFeYE8CFqgR9_owOz9LuN7R0e1ic6IP6H6MvBgpkLYB5UV4OnAz7XwKikUSkTdFpqcuoqmxqFtK6jjPoLNw6Q_5kjfMkymIzGArbv6W80S4VOoe7-6mbKWNAP8570lAI_Ci38NFKImLBFSQS5-j5GHiPRm5jD2W_afVuJZ9rxW4gJYNu0DUhFDoe6qPgkvoqVWtW-3i0J_QjjxP35KQZVP-b7bJgaOMJOaIr5GxosdHa286lLJ_VVFsvTD7BwsrGx9hQ-hcem6LByF8X_3b2X4-OqS5FdVWlvs8BpieCCDkwWVujNclqQ8Zks-U2VsRldYIHe6t5tovBnhIOXebnK4RDUemfwZKSgkKiycfXBcOgnqYfRwC4YeZjhN6sX9vKmJfYEoAL3-L4wgwb57R6FD1ntCqDk4fJ2IxvNgSm3Rk7UI6ChdyWZrWiOSBV9alU2K0L8NTtccXq_-ztXbFbFqWq0tJEur4ziGvtAEDILcKPDuFyZcXg2a-7sYGJXp434Vp6bbMW-XxnfNdGgtsflLvHR7OzWUjmibouNWbMikY7nthhUGpJrbIql878CNZM-GRGZcqcOCMe0QllHofSsw3QHepUaf5GatfQk87qs8osCTviyj2wHoxxlsiQmkgPb8yHokzQodcU83hZ4D1ky78t2xCZHJOVNmHKoH3ZrXflb_LpBPUneuPtpBCL0HiZmOtsCxdBjLlZ_xh73Jxz-gfL7ZGLuUiBKJ-hKOXgHg7voSMJOam59ny3DHwZlEM4lSLPcQPqhQKzCW_Hyeho-lBF_yr7NoxBLnOWEc3cqxZHdcAJ0KYhqiw3p06nNgcnPNwNJTUgIzO4cNASG7PinUudeHRI4az5_WirkJTFV4wU8d6tRvLdb61CJuxeYpnnAxNQdqNPILHXiKxxYl0W669T4HEDdGs5l750Cg1mIQ7JXNRhipOU9-0Pfaaep4iLN7A4SZbIiV1TKbIGwtKAqk-rjgU2GXk0JOjmukv3eIWTwRwlccn3AEKfujMnZqIkn4P9l1DEqO490pN2R0Wu-_Hg../download"         
            self.saved_path = self.download_url("card_transaction.v1.csv", url, unzip = True)

    def to_pandas(self, scale = 'full'):
        import pandas as pd
        if scale == 'test':
            return pd.read_csv(self.saved_path, nrows = 100000)
        return pd.read_csv(self.saved_path)
         