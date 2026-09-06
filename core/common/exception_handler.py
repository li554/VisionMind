"""
全局异常处理模块 - 捕获所有未处理的异常，包括 Qt 信号槽中的异常
"""
import sys
import os
import traceback
import logging
import threading
from datetime import datetime
from PySide6.QtCore import Qt, QObject, Signal, Slot, QTimer
from PySide6.QtWidgets import QMessageBox, QApplication

# 配置日志
log_file = None

def setup_exception_logging(log_path=None):
    """设置异常日志记录"""
    global log_file
    if log_path is None:
        from .config import Config
        log_dir = os.path.join(Config.ROOT_DIR, "logs")
        log_path = f"{log_dir}/crash_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
    
    # 使用更详细的格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(threadName)s:%(thread)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器 - 使用延迟写入
    file_handler = logging.FileHandler(log_path, encoding='utf-8', delay=True)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    log_file = log_path
    return log_path

def log_exception(exc_type, exc_value, exc_traceback, context=""):
    """记录异常到日志文件
    
    Args:
        exc_type: 异常类型
        exc_value: 异常值
        exc_traceback: 异常追溯
        context: 额外的上下文信息
    """
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    thread_info = f"[Thread: {threading.current_thread().name} (ID: {threading.get_ident()})]"
    context_info = f" [Context: {context}]" if context else ""
    
    # 记录到日志文件
    logging.error("=" * 80)
    logging.error(f"Uncaught Exception {thread_info}{context_info}")
    logging.error(f"Exception Type: {exc_type.__name__}")
    logging.error(f"Exception Value: {exc_value}")
    logging.error("Traceback:")
    logging.error(error_msg)
    logging.error("=" * 80)
    
    # 同时打印到控制台
    print("\n" + "=" * 80, file=sys.stderr)
    print(f"[CRITICAL ERROR] {thread_info}{context_info}", file=sys.stderr)
    print(f"Exception Type: {exc_type.__name__}", file=sys.stderr)
    print(f"Exception Value: {exc_value}", file=sys.stderr)
    print("Traceback:", file=sys.stderr)
    print(error_msg, file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    return error_msg

def show_error_dialog(exc_type, exc_value, exc_traceback):
    """显示错误对话框"""
    try:
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # 使用 QTimer 确保在主线程中显示对话框
        def show_dialog():
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("程序错误")
                msg_box.setText(f"发生未处理的异常: {exc_type.__name__}")
                msg_box.setInformativeText(str(exc_value))
                msg_box.setDetailedText(error_msg)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #ffffff;
                    }
                    QMessageBox QLabel {
                        color: #000000;
                    }
                    QPushButton {
                        background-color: #0078d4;
                        color: white;
                        border: none;
                        padding: 6px 20px;
                        border-radius: 4px;
                        min-width: 80px;
                    }
                    QPushButton:hover {
                        background-color: #106ebe;
                    }
                """)
                msg_box.exec()
            except Exception as e:
                print(f"[ExceptionHandler] Failed to show error dialog: {e}", file=sys.stderr)
        
        # 延迟执行以确保在 Qt 事件循环中
        QTimer.singleShot(0, show_dialog)
    except Exception as e:
        print(f"[ExceptionHandler] Error in show_error_dialog: {e}", file=sys.stderr)

class ExceptionHandler(QObject):
    """Qt 异常处理器 - 捕获信号槽中的异常"""
    exception_occurred = Signal(str, str, str)  # context, exc_type, error_msg
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exception_occurred.connect(self._handle_exception)
        self._original_excepthook = sys.excepthook
        self._qt_message_handler = None
        self._main_thread_id = threading.get_ident()
        
    def install(self):
        """安装全局异常处理器"""
        # 保存原始的异常钩子
        self._original_excepthook = sys.excepthook
        
        # 设置自定义异常钩子
        sys.excepthook = self._custom_excepthook
        
        # 安装 Qt 消息处理器
        from PySide6.QtCore import qInstallMessageHandler
        self._qt_message_handler = self._qt_message_handler_func
        qInstallMessageHandler(self._qt_message_handler)
        
        # 安装线程异常处理
        self._original_thread_excepthook = getattr(threading, 'excepthook', None)
        threading.excepthook = self._thread_excepthook
        
        # 安装 atexit 处理器确保退出时记录
        import atexit
        atexit.register(self._atexit_handler)
        
        print("[ExceptionHandler] Global exception handler installed")
        logging.info(f"[ExceptionHandler] Main thread ID: {self._main_thread_id}")
        
    def uninstall(self):
        """卸载全局异常处理器"""
        sys.excepthook = self._original_excepthook
        if self._original_thread_excepthook:
            threading.excepthook = self._original_thread_excepthook
        print("[ExceptionHandler] Global exception handler uninstalled")
        
    def _atexit_handler(self):
        """ateexit 处理器 - 确保异常信息被记录"""
        logging.info("[ExceptionHandler] Application exiting normally")
        
    def _thread_excepthook(self, args):
        """处理线程中的未捕获异常"""
        exc_type, exc_value, exc_traceback = args
        # 使用原始的 excepthook
        if self._original_thread_excepthook:
            self._original_thread_excepthook(args)
        else:
            log_exception(exc_type, exc_value, exc_traceback, 
                         context=f"Thread: {threading.current_thread().name}")
            
    def _custom_excepthook(self, exc_type, exc_value, exc_traceback):
        """自定义异常钩子"""
        # 获取上下文信息
        context = f"MainThread:{threading.current_thread().name}"
        
        # 记录异常
        error_msg = log_exception(exc_type, exc_value, exc_traceback, context=context)
        
        # 显示错误对话框（只在主线程）
        if threading.get_ident() == self._main_thread_id:
            show_error_dialog(exc_type, exc_value, exc_traceback)
        
        # 调用原始的异常钩子
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _qt_message_handler_func(self, mode, context, message):
        """Qt 消息处理器 - 捕获 Qt 内部错误"""
        # 过滤掉常见的非错误消息
        suppressed_keywords = [
            'does not have a property named',
            'normalColor',
            'hoverColor',
            'pressedColor',
            'normalBackgroundColor',
            'hoverBackgroundColor',
            'pressedBackgroundColor',
            'QFont::setPointSize: Point size',
            'QWindowsWindow:',  # Windows-specific window messages
            'QWidget::repaint:',  # Repaint warnings during shutdown
        ]
        
        for keyword in suppressed_keywords:
            if keyword in message:
                return
        
        # 错误关键词 - 这些需要被记录
        error_keywords = ['error', 'fail', 'crash', 'fatal', 'abort', 'assert']
        is_error = any(kw in message.lower() for kw in error_keywords)
        
        if mode == 0:  # QtDebugMsg
            if is_error:
                logging.debug(f"[Qt Debug] {message}")
        elif mode == 1:  # QtWarningMsg
            if is_error:
                logging.warning(f"[Qt Warning] {message}")
            print(f"[Qt Warning] {message}")
        elif mode == 2:  # QtCriticalMsg
            print(f"[Qt Critical] {message}", file=sys.stderr)
            logging.error(f"[Qt Critical] {message}")
        elif mode == 3:  # QtFatalMsg
            print(f"[Qt Fatal] {message}", file=sys.stderr)
            logging.critical(f"[Qt Fatal] {message}")
            # Qt Fatal 是严重错误，尝试记录更多信息
            import platform
            logging.critical(f"Platform: {platform.platform()}")
            logging.critical(f"Python: {platform.python_version()}")
            try:
                import PySide6
                logging.critical(f"PySide6: {PySide6.__version__}")
            except:
                pass
        elif mode == 4:  # QtInfoMsg
            print(f"[Qt Info] {message}")
    
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """处理异常信号"""
        log_exception(exc_type, exc_value, exc_traceback)
        show_error_dialog(exc_type, exc_value, exc_traceback)

def safe_slot(func):
    """装饰器 - 包装槽函数以捕获异常"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            log_exception(exc_type, exc_value, exc_traceback)
            show_error_dialog(exc_type, exc_value, exc_traceback)
            raise  # 重新抛出以便调试模式可以捕获
    return wrapper

class SafeQObject(QObject):
    """安全的 QObject 基类 - 自动包装所有槽函数"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def connect_safe(self, signal, slot):
        """安全地连接信号和槽"""
        @safe_slot
        def safe_wrapper(*args, **kwargs):
            return slot(*args, **kwargs)
        signal.connect(safe_wrapper)

# 全局异常处理器实例
_global_handler = None

def install_global_exception_handler():
    """安装全局异常处理器"""
    global _global_handler
    if _global_handler is None:
        _global_handler = ExceptionHandler()
        _global_handler.install()
        setup_exception_logging()
    return _global_handler

def uninstall_global_exception_handler():
    """卸载全局异常处理器"""
    global _global_handler
    if _global_handler is not None:
        _global_handler.uninstall()
        _global_handler = None


class PyTorchErrorCapture:
    """PyTorch/CUDA 错误捕获器 - 专门处理深度学习相关的错误"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._setup_handlers()
        
    def _setup_handlers(self):
        """设置 PyTorch/CUDA 相关的错误处理"""
        try:
            import torch
            # 检查 CUDA 是否可用
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                logging.info(f"[PyTorch] CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logging.info("[PyTorch] CUDA not available, using CPU")
                
            # 设置警告处理
            torch.set_warn_always(True)
            
        except ImportError:
            logging.info("[PyTorch] PyTorch not installed")
        except Exception as e:
            logging.warning(f"[PyTorch] Error during setup: {e}")
    
    def wrap_function(self, func, context=""):
        """包装函数以捕获 PyTorch/CUDA 错误
        
        Args:
            func: 要包装的函数
            context: 上下文描述
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 检查是否是 PyTorch/CUDA 相关的错误
                error_str = str(e).lower()
                pytorch_keywords = ['cuda', 'gpu', 'torch', 'tensor', 'cudnn', 'out of memory', 'device']
                
                if any(kw in error_str for kw in pytorch_keywords):
                    logging.error(f"[PyTorch Error] {context}: {e}")
                    logging.error(traceback.format_exc())
                    
                    # 尝试清理 CUDA 缓存
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logging.info("[PyTorch] CUDA cache cleared after error")
                    except:
                        pass
                
                # 重新抛出原始异常以便上层处理
                raise
        return wrapper
    
    def safe_segment(self, segment_func, *args, **kwargs):
        """安全地调用分割函数
        
        专门用于 SAM 模型调用，防止 CUDA 错误导致闪退
        """
        context = kwargs.pop('_context', 'segment')
        try:
            result = segment_func(*args, **kwargs)
            return result
        except Exception as e:
            error_str = str(e).lower()
            pytorch_keywords = ['cuda', 'gpu', 'torch', 'tensor', 'cudnn', 'out of memory', 
                               'runtimeerror', 'attributeerror', 'segmentation fault']
            
            is_pytorch_error = any(kw in error_str for kw in pytorch_keywords)
            
            logging.error(f"[PyTorch Error] In {context}: {e}")
            logging.error(traceback.format_exc())
            
            if is_pytorch_error:
                # 尝试恢复
                self._try_recovery()
                return {'status': 'error', 'message': f'PyTorch/CUDA error: {e}'}
            else:
                # 非 PyTorch 错误，重新抛出
                raise
    
    def _try_recovery(self):
        """尝试从 PyTorch/CUDA 错误中恢复"""
        try:
            import torch
            if torch.cuda.is_available():
                logging.info("[PyTorch] Attempting CUDA recovery...")
                
                # 同步 CUDA 流
                torch.cuda.synchronize()
                
                # 清空缓存
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                logging.info("[PyTorch] CUDA recovery completed")
        except Exception as e:
            logging.error(f"[PyTorch] Recovery failed: {e}")


def wrap_pytorch_call(context=""):
    """装饰器 - 包装 PyTorch 函数调用以捕获错误
    
    用法:
        @wrap_pytorch_call("bbox_to_polygon")
        def my_function(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            capturer = PyTorchErrorCapture()
            return capturer.wrap_function(func, context)(*args, **kwargs)
        return wrapper
    return decorator


# 创建全局 PyTorch 错误捕获器
_pytorch_capturer = None

def get_pytorch_capturer():
    """获取 PyTorch 错误捕获器单例"""
    global _pytorch_capturer
    if _pytorch_capturer is None:
        _pytorch_capturer = PyTorchErrorCapture()
    return _pytorch_capturer
