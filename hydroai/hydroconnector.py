class HydroConnector:
    '''
    Hydro Service is a long run service for client to submit learning
    queries and resume or fetch old result

    :param url(str): example as http://127.0.0.1:9090

    '''
    def __init__(self, url):
        self.url = url

    @classmethod
    def launch_or_get_service(cls, url):
        return cls(url)
