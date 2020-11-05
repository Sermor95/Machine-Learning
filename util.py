import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def logging_info(function,info):
    logging.info(f'//---{function}--->{info}---//')