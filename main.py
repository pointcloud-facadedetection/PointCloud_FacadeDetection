import os
import sys

# 将项目根目录加入路径，确保 backend 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.config import Config
from backend.api.routes import register_routes
from flask import Flask
from flask_cors import CORS
import platform


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, 'templates'),
        static_folder=os.path.join(PROJECT_ROOT, 'templates')
    )
    CORS(app, resources={r"/*": {"origins": "*"}})

    # 确保上传目录存在
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # 注册路由
    register_routes(app)

    return app


def run_server():
    """启动服务器"""
    app = create_app()

    if platform.system() == 'Windows':
        try:
            from waitress import serve
            print("[INFO] 启动 Waitress 生产服务器 (Windows) | 线程: 4 | 端口: 5000")
            serve(app, host='0.0.0.0', port=5000, threads=4, channel_timeout=600)
        except ImportError:
            print("[WARN] Waitress 未安装，回退到 Flask 多线程模式")
            print("[HINT] pip install waitress")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("[INFO] 启动 Flask 开发服务器 | 端口: 5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    run_server()
