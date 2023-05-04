import configparser

class Conf:
    def __init__(self, config_path) -> None:
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        try:
            config.read(config_path)
        except:
            raise IOError('Something is wrong with the Config file path!')
        
        self.paths = config._sections['paths']
        self.diffusion = config._sections['diffusion']
        self.summarization = config._sections['summarization']
        
        # Function to convert certain variables to appropriate data types
        self.convert_types()

    def convert_types(self):
        self.diffusion['inf_steps'] = int(self.diffusion['inf_steps'])
        self.summarization['min_length'] = int(self.summarization['min_length'])
        self.summarization['max_length'] = int(self.summarization['max_length'])
        