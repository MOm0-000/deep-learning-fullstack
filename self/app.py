# -*- coding: utf-8 -*-
"""
Flask 后端服务模块 —— 图像识别 Web 应用
提供单张图片识别（文件上传 / URL）、批量识别并导出结果（CSV/JSON/Markdown）。
集成 AlexNet 和蒸馏 ConvNeXt-Nano 模型，通过 model.py 中的 predict_image 接口调用。
"""

import io
import csv
import json
import imghdr
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
from model import predict_image   # 直接导入推理函数，无需关心模型加载细节

# --------------------------------------------------------------
# 1. Flask 应用实例化及基础配置
# --------------------------------------------------------------
app = Flask(__name__)
# 限制整个请求体的最大大小为 16 MB（防止大文件上传耗尽内存）
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --------------------------------------------------------------
# 2. 全局配置常量（可根据需要调整）
# --------------------------------------------------------------
MAX_SINGLE_FILE_SIZE = 10 * 1024 * 1024      # 单张图片最大 10 MB
MAX_BATCH_SIZE = 20                          # 批量最多 20 张
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}  # 允许的图片扩展名（白名单）
MAX_IMAGE_DIMENSION = 1024                   # 图片最大边长（超过则等比缩放）
IMAGE_COMPRESS_QUALITY = 85                  # JPEG 重压缩质量（1-100）
WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / "distll" / "distill_output" / "best_student_distilled.pth")

# --------------------------------------------------------------
# 3. 辅助函数
# --------------------------------------------------------------

def allowed_file(filename):
    """
    检查文件名是否具有允许的图片扩展名（后缀名白名单）。
    参数:
        filename: 上传文件的原始名称
    返回:
        bool: 若扩展名在 ALLOWED_EXT 中返回 True，否则 False
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def process_image_bytes(content_bytes):
    """
    将图片字节流转换为 PIL Image 对象，并进行尺寸缩放（若超过 MAX_IMAGE_DIMENSION）。
    参数:
        content_bytes: 原始图片的字节数据
    返回:
        PIL.Image.Image: 处理后的图片对象（可能被缩放）
    """
    img = Image.open(io.BytesIO(content_bytes))   # 从字节流打开图片
    width, height = img.size
    # 若宽或高超过阈值，则按比例缩放至最大边长不超过 MAX_IMAGE_DIMENSION
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
        # 使用 LANCZOS 重采样算法（高质量）
        img = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
    return img

def validate_and_process(file=None, content_bytes=None):
    """
    统一验证和压缩逻辑（核心安全与容错函数）。
    支持两种输入方式：
      - 通过 file 参数传入 Flask 文件对象（用于表单上传）
      - 通过 content_bytes 参数传入已读取的字节数据（用于 URL 下载）
    执行三步校验：
      1. 扩展名白名单（若通过 file 传入）
      2. 文件大小限制
      3. 魔数（Magic Number）检测，确认是否为真实图片
    然后进行图片清洗：强制转 RGB 并重新压缩为 JPEG，丢弃潜在恶意元数据。
    若压缩失败，则降级使用原始字节（容错策略）。

    参数:
        file: Flask 文件对象（可选）
        content_bytes: 图片字节数据（可选）
    返回:
        (valid, err_msg, processed_bytes)
        valid: bool，是否验证通过
        err_msg: 错误信息（若 valid=False）
        processed_bytes: 处理后的字节数据（清洗后或原始降级）
    """
    # 若传入了 file 对象，则优先处理文件上传逻辑
    if file:
        # 检查文件对象有效性、文件名是否为空、扩展名是否允许
        if not file or file.filename == '' or not allowed_file(file.filename):
            return False, '无效的文件或类型', None
        # 读取文件全部内容到内存
        content_bytes = file.read()
        # 将文件指针重置到开头（防止后续操作受到影响，虽然本函数内不再使用）
        file.seek(0)

    # 检查文件大小（无论从 file 还是 content_bytes 来源，此时 content_bytes 必须非空）
    if len(content_bytes) > MAX_SINGLE_FILE_SIZE:
        return False, f'文件超过 {MAX_SINGLE_FILE_SIZE//(1024*1024)}MB 限制', None

    # 使用 imghdr.what 检测图片魔数（文件头签名），确认是否为有效图片格式
    if imghdr.what(None, content_bytes) is None:
        return False, '文件不是有效图片', None

    # 尝试对图片进行“清洗”：解码、缩放、重新编码为 JPEG（去除潜在恶意数据）
    try:
        img = process_image_bytes(content_bytes)           # 打开并可能缩放
        out = io.BytesIO()                                 # 内存字节流，不落盘
        # 强制转为 RGB（兼容 RGBA、P 等模式），并保存为 JPEG 格式（质量 85%）
        img.convert('RGB').save(out, format='JPEG', quality=IMAGE_COMPRESS_QUALITY)
        return True, '', out.getvalue()                    # 返回清洗后的字节数据
    except Exception as e:
        # 若压缩过程异常（如 PIL 无法处理特殊格式），则记录警告并降级使用原始字节
        app.logger.warning(f"压缩图片失败: {e}，使用原始内容")
        return True, '', content_bytes                     # 仍认为校验通过，返回原始字节

# --------------------------------------------------------------
# 4. 路由定义
# --------------------------------------------------------------

@app.route('/')
def index():
    """首页，渲染前端界面（index.html）"""
    return render_template('index.html')

# --------------------------------------------------------------
# 4.1 单张图片识别（表单上传）
# --------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_single():
    """
    处理单张图片识别请求（通过 multipart/form-data 上传文件）。
    请求参数:
        - file: 图片文件（必填）
        - model: 模型名称，可选 'alexnet' 或 'convnext'，默认 'alexnet'
    返回:
        JSON 格式，包含 success 字段和 predictions 列表（Top-5）
        或 error 字段表示错误信息
    """
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'})

    # 获取用户指定的模型名称，若不合法则回退到 'alexnet'
    model_name = request.form.get('model', 'alexnet')
    if model_name not in ('alexnet', 'convnext'):
        model_name = 'alexnet'

    # 调用验证和清洗函数（传入文件对象）
    valid, err, content = validate_and_process(file=request.files['file'])
    if not valid:
        return jsonify({'error': err})

    # 进行实际推理
    try:
        # 将清洗后的字节重新转为 PIL Image（已处理过尺寸）
        img = process_image_bytes(content).convert('RGB')
        # 调用 model.py 的预测函数，返回 Top-5 列表
        preds = predict_image(img, model_name=model_name, weights_path=WEIGHTS_PATH)
        # 构造前端需要的 JSON 结构，置信度保留两位小数
        return jsonify({
            'success': True,
            'predictions': [{'class': c, 'confidence': round(p, 2)} for c, p in preds]
        })
    except Exception as e:
        # 捕获所有预测异常，记录日志并返回友好错误信息
        app.logger.error(f"单张预测出错: {e}", exc_info=True)
        return jsonify({'error': f'预测出错: {str(e)}'})

# --------------------------------------------------------------
# 4.2 通过 URL 识别图片
# --------------------------------------------------------------
@app.route('/predict_url', methods=['POST'])
def predict_url():
    """
    处理通过 URL 下载图片并识别的请求。
    请求体为 JSON，包含:
        - url: 图片的 HTTP/HTTPS 链接（必填）
        - model: 模型名称，可选，默认 'alexnet'
    返回:
        JSON，同 /predict 接口
    增加安全限制：仅允许 http/https 协议，设置超时，限制下载大小。
    """
    data = request.get_json()
    # 校验请求数据：必须存在 url 字段，且以 http:// 或 https:// 开头
    if not data or 'url' not in data or not data['url'].startswith(('http://', 'https://')):
        return jsonify({'error': '无效的 url 或协议'}), 400

    model_name = data.get('model', 'alexnet')
    if model_name not in ('alexnet', 'convnext'):
        model_name = 'alexnet'

    try:
        # 使用 requests 流式下载，设置超时 10 秒，stream=True 便于按块读取
        resp = requests.get(data['url'], timeout=10, stream=True)
        resp.raise_for_status()   # 若状态码非 2xx，抛出 HTTPError
        # 检查响应头的 Content-Type 是否以 image/ 开头（基本图片类型检测）
        if not resp.headers.get('content-type', '').startswith('image/'):
            return jsonify({'error': 'URL 不是图片类型'}), 400

        # 将下载内容写入字节缓冲区，同时监控大小
        content = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=8192):
            if content.tell() > MAX_SINGLE_FILE_SIZE:   # 检查已写入字节数
                return jsonify({'error': '图片超过大小限制'}), 413
            content.write(chunk)

        content_bytes = content.getvalue()
        # 调用验证和清洗函数（传入字节数据）
        valid, err, content_bytes = validate_and_process(content_bytes=content_bytes)
        if not valid:
            return jsonify({'error': err}), 400

        # 推理并返回结果
        img = process_image_bytes(content_bytes).convert('RGB')
        preds = predict_image(img, model_name=model_name, weights_path=WEIGHTS_PATH)
        return jsonify({
            'success': True,
            'predictions': [{'class': c, 'confidence': round(p, 2)} for c, p in preds]
        })

    except requests.exceptions.Timeout:
        # 超时异常返回 408 状态码
        return jsonify({'error': '下载超时'}), 408
    except Exception as e:
        app.logger.error(f"URL 识别出错: {e}", exc_info=True)
        return jsonify({'error': f'识别出错: {str(e)}'}), 500

# --------------------------------------------------------------
# 4.3 批量识别并导出结果
# --------------------------------------------------------------
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量识别多张图片，并根据指定的格式导出结果文件（CSV/JSON/Markdown）。
    请求参数:
        - files[]: 多个图片文件（表单项）
        - format: 输出格式，'csv'、'json' 或 'md'，默认 'csv'
        - model: 模型名称，默认 'alexnet'
    返回:
        一个附件响应，文件名包含时间戳，内容为结果数据。
    每张图片独立处理，单张失败不影响其他图片。
    """
    # 获取输出格式，并校验合法性
    fmt = request.form.get('format', 'csv').lower()
    if fmt not in ('csv', 'json', 'md'):
        return jsonify({'error': '不支持的输出格式'}), 400

    # 获取模型名称
    model_name = request.form.get('model', 'alexnet')
    if model_name not in ('alexnet', 'convnext'):
        model_name = 'alexnet'

    # 获取所有上传的文件（name='files[]'）
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': '未选择任何文件'}), 400
    if len(files) > MAX_BATCH_SIZE:
        return jsonify({'error': f'最多支持 {MAX_BATCH_SIZE} 张图片'}), 413

    results = []   # 存储每张图片的处理结果

    # 逐张处理
    for f in files:
        # 初始化结果字典，状态默认为 failed
        res = {'filename': f.filename, 'status': 'failed', 'error': '', 'top_predictions': []}
        # 验证和清洗
        valid, err, content = validate_and_process(file=f)
        if not valid:
            res['error'] = err
        else:
            try:
                img = process_image_bytes(content).convert('RGB')
                preds = predict_image(img, model_name=model_name, weights_path=WEIGHTS_PATH)
                if preds:
                    res['status'] = 'success'
                    # 只取 Top-3 用于报表（减少数据量）
                    res['top_predictions'] = [{'class': c, 'confidence': round(p, 2)} for c, p in preds[:3]]
                else:
                    res['error'] = '模型未返回有效结果'
            except Exception as e:
                res['error'] = f'推理失败: {str(e)}'
        results.append(res)

    # 生成时间戳用于文件名
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- 根据格式生成数据 ----------
    if fmt == 'csv':
        # CSV 格式：加入 BOM（\ufeff）以便 Excel 正确识别 UTF-8
        out = io.StringIO('\ufeff')
        w = csv.writer(out)
        # 构建表头：文件名、状态、错误信息，然后 Top-1 到 Top-3 的类别和置信度
        headers = ['文件名', '状态', '错误信息']
        for i in range(1, 4):
            headers.extend([f'Top-{i} 类别', f'Top-{i} 置信度(%)'])
        w.writerow(headers)

        for r in results:
            row = [r['filename'], r['status'], r['error']]
            # 添加预测结果（最多3个）
            for top in r['top_predictions']:
                row.extend([top['class'], f"{top['confidence']:.2f}"])
            # 补齐空列（若预测不足3个）
            row.extend(['', ''] * (3 - len(r['top_predictions'])))
            w.writerow(row)

        data = out.getvalue().encode('utf-8-sig')
        mime, ext = 'text/csv; charset=utf-8', 'csv'

    elif fmt == 'json':
        # JSON 格式：包含 total 和 results 数组
        data = json.dumps({'total': len(results), 'results': results}, ensure_ascii=False, indent=2).encode('utf-8')
        mime, ext = 'application/json', 'json'

    else:  # markdown
        # Markdown 格式：生成表格报告
        lines = [
            '# 批量识别结果报告',
            f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'总图片数：{len(results)}',
            '',
            '| 文件名 | 状态 | 错误信息 | Top-1 | Top-2 | Top-3 |',
            '|---|---|---|---|---|---|'
        ]
        for r in results:
            fname = r['filename'].replace('|', '\\|')   # 转义表格分隔符
            status = '✅ 成功' if r['status'] == 'success' else '❌ 失败'
            tops = [f"{t['class']} ({t['confidence']:.2f}%)" for t in r['top_predictions']]
            tops += [''] * (3 - len(tops))   # 补全空单元格
            lines.append(f'| {fname} | {status} | {r["error"]} | {tops[0]} | {tops[1]} | {tops[2]} |')
        data = '\n'.join(lines).encode('utf-8')
        mime, ext = 'text/markdown', 'md'

    # 构建响应，设置 Content-Disposition 为附件下载
    resp = Response(data, mimetype=mime)
    resp.headers['Content-Disposition'] = f'attachment; filename="results_{ts}.{ext}"'
    return resp

# --------------------------------------------------------------
# 5. 启动应用
# --------------------------------------------------------------
if __name__ == '__main__':
    # debug=True 开启调试模式，host='0.0.0.0' 监听所有网络接口，port=5000
    app.run(debug=True, host='0.0.0.0', port=5000)
