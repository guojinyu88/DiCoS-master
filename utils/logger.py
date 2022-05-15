import logging

def get_logger(loggerName:str, loggerPath:str) -> logging.Logger:

    FILE_MODE = 'a'
    LOG_FORMAT = "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
    
    myLogger = logging.getLogger(loggerName)
    myLogger.setLevel(logging.DEBUG)

    printHandler = logging.StreamHandler()
    printHandler.setFormatter(logging.Formatter(LOG_FORMAT))
    printHandler.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(filename=loggerPath, mode=FILE_MODE, encoding='utf-8')
    fileHandler.setFormatter(logging.Formatter(LOG_FORMAT))
    fileHandler.setLevel(logging.DEBUG)

    myLogger.addHandler(printHandler)
    myLogger.addHandler(fileHandler)

    return myLogger