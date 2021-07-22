import os
import logging
import time
import threading

s_oLock = None
s_oLog = None

# 初始化log配置
def init_log(astrLogName, LOG_PATH, DEBUG_MODE):
    global s_oLog
    global s_oLock

    if s_oLog is None:
        # FORMAT = ('%(asctime)-15s %(threadName)-15s'
        #        ' %(levelname)-8s %(module)-15s:%(lineno)-8s %(message)s')
        FORMAT = ('%(asctime)-15s %(threadName)-15s %(levelname)-8s %(message)s')
        useLevel = logging.DEBUG if DEBUG_MODE else logging.INFO

        s_oLog = logging.getLogger(astrLogName)
        s_oLog.setLevel(useLevel)
        #s_oLog.basicConfig(level=useLevel, format=FORMAT)

        # release生成当前日志文件名
        timeArray = time.localtime(time.time())
        strTime = time.strftime("%Y-%m-%d", timeArray)
        strLogFilePath = os.path.join(LOG_PATH, astrLogName + "_" + strTime + ".log")
        print("strLogFilePath = " + strLogFilePath)
        fmt = logging.Formatter(FORMAT)
        # 调试时设置输出控制台，release直接输出到日志
        handler = logging.StreamHandler() if DEBUG_MODE else logging.FileHandler(strLogFilePath)
        handler.setLevel(useLevel)
        handler.setFormatter(fmt)
        s_oLog.addHandler(handler)

        s_oLock = threading.RLock()

        # done
    # end if


# end def

"""
Log a message with severity 'CRITICAL' on the root logger
"""


def critical(msg, *args, **kwargs):
    global s_oLock
    global s_oLog
    if s_oLog is None: return
    with s_oLock:
        s_oLog.critical(msg, *args, **kwargs)
    # end with


# end def

"""
Log a message with severity 'ERROR' on the root logger
"""


def error(msg, *args, **kwargs):
    global s_oLock
    global s_oLog
    if s_oLog is None: return
    with s_oLock:
        s_oLog.error(msg, *args, **kwargs)
    # end with


# end def

"""
Log a message with severity 'WARNING' on the root logger
"""


def warning(msg, *args, **kwargs):
    global s_oLock
    global s_oLog
    if s_oLog is None: return
    with s_oLock:
        s_oLog.warning(msg, *args, **kwargs)
    # end with


# end def

"""
Log a message with severity 'INFO' on the root logger
"""


def info(msg, *args, **kwargs):
    global s_oLock
    global s_oLog
    if s_oLog is None: return
    with s_oLock:
        s_oLog.info(msg, *args, **kwargs)
    # end with


# end def

"""
Log a message with severity 'DEBUG' on the root logger
"""


def debug(msg, *args, **kwargs):
    global s_oLock
    global s_oLog
    if s_oLog is None: return
    with s_oLock:
        s_oLog.debug(msg, *args, **kwargs)
    # end with
# end def
