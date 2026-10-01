# -*- coding: utf-8 -*-
"""
Flask Web 应用 - 图像分类预测服务
该应用使用预训练的深度学习模型对上传的图片进行分类预测
核心交互流程：
1. 前端(index.html)访问首页 → 后端返回渲染后的页面
2. 前端上传图片 → 后端接收文件并调用模型预测 → 返回JSON格式的预测结果
3. 前端根据返回结果展示图片和预测信息/错误提示
"""

import os
# 解决KMP库重复加载的问题，避免在某些环境下出现警告或错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入Flask核心模块：
# Flask: 应用实例创建
# render_template: 渲染HTML模板（给前端返回页面）
# request: 接收前端请求数据（如上传的文件、表单参数）
# jsonify: 将Python字典转为JSON响应（给前端返回数据）
from flask import Flask, render_template, request, jsonify
from model import load_model, predict  # 导入模型加载和预测核心函数
from werkzeug.utils import secure_filename  # 安全处理文件名，防止路径攻击

# ==================== Flask 应用初始化 ====================
# 创建Flask应用实例，指定模板和静态文件目录
# 模板目录默认: templates/ (存放index.html)
# 静态文件目录默认: static/ (存放上传的图片、CSS/JS等)
app = Flask(__name__)

# ==================== 应用配置 ====================
# 上传文件保存目录（静态文件夹下），供index.html前端访问上传的图片
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# 限制上传文件最大为16MB，防止大文件攻击（前端也应做对应大小限制）
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# 允许上传的图片文件扩展名集合（需与index.html前端文件选择器的accept属性保持一致）
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ==================== 初始化 ====================
# 确保上传目录存在，如果不存在则自动创建
# 保证前端上传的图片能正常保存，后续index.html能访问到
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 在应用启动时预加载深度学习模型，避免每次预测都重新加载
# 这样可以提高预测的响应速度，减少前端等待时间
model = load_model()


# ==================== 辅助函数 ====================
def allowed_file(filename):
    """
    检查文件扩展名是否在允许的列表中
    用于验证前端(index.html)上传的文件类型是否合法
    
    参数:
        filename (str): 前端上传的文件名
        
    返回:
        bool: 如果文件扩展名合法返回True，否则返回False
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== 路由（视图函数）- 与index.html核心交互入口 ====================
#Flask 匹配 / 路由，执行 index() 函数；
@app.route('/')
def index():
    """
    首页路由 - 与index.html的核心交互入口
    前端浏览器访问根路径(/)时，后端返回渲染后的index.html页面
    交互逻辑：
    1. 前端发起GET请求 → http://服务器IP:5000/
    2. 后端渲染templates/index.html模板并返回给前端
    3. 前端展示包含"图片上传控件+预测按钮"的页面
    
    返回:
        rendered template: 渲染后的index.html页面（包含HTML/CSS/JS）
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image():
    """
    预测路由（仅接受POST请求）- 处理index.html的图片上传和预测请求
    与index.html的交互约定：
    前端请求：
        - 请求方式: POST
        - 请求体: FormData格式，包含名为 'file' 的文件字段（需与index.html的input标签name属性一致）
        - 请求地址: /predict
    后端响应（JSON格式）：
        - 成功: {'success': True, 'image_url': 图片访问路径, 'predictions': 预测结果列表}
        - 失败: {'error': 错误信息（供前端展示）}
    
    返回:
        JSON: 包含预测结果或错误信息，供index.html前端解析展示
    """
    # 1. 校验前端请求是否包含文件字段（index.html的input[type=file]的name需为'file'）
    if 'file' not in request.files:
        # 前端未上传文件时，返回错误JSON，供index.html展示"没有文件部分"提示
        return jsonify({'error': '没有文件部分'})
    
    # 2. 获取前端上传的文件对象
    file = request.files['file']
    
    # 3. 校验前端是否选择了文件（文件名空表示未选择）
    if file.filename == '':
        # 返回错误JSON，供index.html展示"未选择文件"提示
        return jsonify({'error': '未选择文件'})
    
    # 4. 校验文件类型是否合法（与ALLOWED_EXTENSIONS匹配）
    if file and allowed_file(file.filename):
        # 4.1 安全处理文件名，防止路径遍历攻击（前端可能传入恶意文件名）
        filename = secure_filename(file.filename)
        # 4.2 构建文件保存路径（保存在静态文件夹，供index.html访问）
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # 4.3 将前端上传的文件保存到服务器静态目录
        file.save(filepath)
        
        try:
            # 5. 调用模型预测函数，处理前端上传的图片
            # predict函数返回格式: [(类别名, 置信度), ...]
            predictions = predict(filepath, model)
            
            # 6. 格式化预测结果（适配index.html前端展示格式）
            # 将置信度转为百分比字符串，方便前端直接展示
            result = [{'class': pred[0], 'confidence': f"{pred[1]:.2f}%"} for pred in predictions]
            
            # 7. 返回成功响应（供index.html解析）：
            # - image_url: 上传图片的访问路径（前端通过该URL展示图片）
            # - predictions: 格式化后的预测结果（前端遍历展示类别+置信度）
            return jsonify({
                'success': True,
                'image_url': filepath,      # 前端访问路径示例: static/uploads/xxx.jpg
                'predictions': result        # 示例: [{'class': 'cat', 'confidence': '98.50%'}, ...]
            })
        except Exception as e:
            # 预测过程异常，返回错误信息，供index.html展示"预测出错"提示
            return jsonify({'error': f'预测出错: {str(e)}'})
    
    # 8. 文件类型不支持时，返回错误JSON，供index.html展示"不支持的文件类型"提示
    return jsonify({'error': '不支持的文件类型'})


# ==================== 启动应用 ====================
if __name__ == '__main__':
    # 启动Flask开发服务器
    # debug=True: 开启调试模式，代码修改后自动重启（开发环境用）
    # host='0.0.0.0': 监听所有网络接口，允许前端（如浏览器）从外部访问
    # port=5000: 服务端口号（前端访问地址: http://服务器IP:5000/）
    # 注意：生产环境需关闭debug，改用WSGI服务器部署
    app.run(debug=True, host='0.0.0.0', port=5000)