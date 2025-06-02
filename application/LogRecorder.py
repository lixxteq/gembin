import logging
import os
import sys

class CLogRecoder:
    def __init__(self, logfile='log.txt', level=logging.INFO):
        self.logger = logging.getLogger('LogRecorder')
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # create log directory if it doesn't exist
        log_dir = os.path.dirname(logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # file handler
        self.fileHandler = logging.FileHandler(logfile, encoding='utf-8')
        self.fileHandler.setLevel(level)
        self.fileHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.fileHandler)
    
    def addStreamHandler(self):
        # stream handler for console output
        self.streamHandler = logging.StreamHandler(sys.stdout)
        self.streamHandler.setLevel(logging.INFO)
        self.streamHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.streamHandler)
    
    def DEBUG(self, message):
        self.logger.debug(message)
    
    def INFO(self, message):
        self.logger.info(message)
    
    def WARNING(self, message):
        self.logger.warning(message)
    
    def ERROR(self, message):
        self.logger.error(message)
    
    def CRITICAL(self, message):
        self.logger.critical(message)