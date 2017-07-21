class BaseConfig:
    DataBase = ''
    celery_ = ''
# 1:data_process.py, 2:config.py
# 1:fatal error, 2:error, 3:output, 4-6:important information, 7-A:normal information, B-F:extra information
log_level = {
    0x01, 0x02,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
    0x21, 0x22, 0x26,
    0x31, 0x32,
    0x41, 0x42
}


def log(level, *messages, **kwargs):
    if (level & 0x0F) < 0x03:
        if not kwargs.get('file'):
            from sys import stderr
            kwargs['file'] = stderr
    if level in log_level:
        print(*messages, **kwargs)
