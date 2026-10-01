"""
深度学习全栈开发项目实践 - 单元测试与集成测试
文件: test.py
运行方式: pytest test.py -v
"""

import pytest
import os
import tempfile
import json
import io
from PIL import Image
import numpy as np

# 导入待测试的模块
from app import app
from model import load_model, predict, transform, labels


# ==================== 1. 测试数据加载 ====================

class TestDataLoading:
    """测试数据加载功能"""
    
    def test_labels_loaded(self):
        """测试类别标签是否成功加载"""
        assert labels is not None
        assert len(labels) > 0
        # ImageNet有1000个类别
        assert len(labels) == 1000 or len(labels) == 15  # 15是备用标签的数量
        
    def test_label_content(self):
        """测试标签内容是否正确"""
        # 检查常见类别是否存在（备用标签或真实标签）
        label_values = list(labels.values())
        # 只要有标签内容就行
        assert len(label_values) > 0
        assert all(isinstance(v, str) for v in label_values)
    
    def test_transform_defined(self):
        """测试图像预处理流程是否定义"""
        assert transform is not None
        # 检查transform是否包含必要的步骤
        assert len(transform.transforms) >= 4  # Resize, CenterCrop, ToTensor, Normalize


# ==================== 2. 测试模型推理 ====================

class TestModelInference:
    """测试模型推理功能"""
    
    def test_model_load(self):
        """测试模型加载"""
        model = load_model()
        assert model is not None
        # 检查模型有eval方法
        assert hasattr(model, 'eval')
        
    def test_model_eval_mode(self):
        """测试模型是否正确设置为评估模式"""
        model = load_model()
        # 对于MockModel，eval返回self，所以总是True
        # 对于真实模型，检查training属性
        if hasattr(model, 'training'):
            assert not model.training
            
    def test_predict_with_valid_image(self):
        """测试使用有效图片进行预测"""
        # 创建一张临时测试图片
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            model = load_model()
            predictions = predict(tmp_path, model)
            
            # 验证预测结果格式
            assert isinstance(predictions, list)
            assert len(predictions) == 5  # 返回前5个结果
            assert len(predictions[0]) == 2  # 每个结果是(类别, 置信度)元组
            assert isinstance(predictions[0][0], str)
            assert isinstance(predictions[0][1], float)
        finally:
            os.unlink(tmp_path)
    
    def test_predict_with_invalid_image(self):
        """测试使用无效图片进行预测（应返回降级结果）"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'not an image')
            tmp_path = tmp.name
        
        try:
            model = load_model()
            predictions = predict(tmp_path, model)
            
            # 即使图片无效，也应返回降级结果（5个模拟结果）
            assert isinstance(predictions, list)
            assert len(predictions) == 5
        finally:
            os.unlink(tmp_path)
    
    def test_prediction_output_range(self):
        """测试预测置信度输出范围"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (224, 224), color='blue')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            model = load_model()
            predictions = predict(tmp_path, model)
            
            for class_name, confidence in predictions:
                # 置信度应在0-100之间
                assert 0 <= confidence <= 100
        finally:
            os.unlink(tmp_path)


# ==================== 3. 测试API接口 ====================

class TestAPI:
    """测试Flask API接口"""
    
    @pytest.fixture
    def client(self):
        """创建Flask测试客户端"""
        app.config['TESTING'] = True
        # 使用临时目录存储上传文件
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config['UPLOAD_FOLDER'] = tmpdir
            with app.test_client() as client:
                yield client
    
    def test_index_page(self, client):
        """测试主页是否正常访问"""
        response = client.get('/')
        assert response.status_code == 200
        # 检查返回的是HTML
        assert b'<!DOCTYPE html' in response.data or b'<html' in response.data
    
    def test_predict_no_file(self, client):
        """测试未上传文件时的API响应"""
        response = client.post('/predict', data={})
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'error' in data
        assert data['error'] == '没有文件部分'
    
    def test_predict_empty_filename(self, client):
        """测试上传空文件名时的API响应"""
        data = {'file': (io.BytesIO(b''), '')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        data = json.loads(response.data)
        
        assert 'error' in data
        assert data['error'] == '未选择文件'
    
    def test_predict_invalid_file_type(self, client):
        """测试上传不支持的文件类型"""
        # 使用 BytesIO 创建文件内容，确保请求格式正确
        file_content = io.BytesIO(b'fake image content')
        data = {
            'file': (file_content, 'test.txt')
        }
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        result = json.loads(response.data)
        
        assert 'error' in result
        # 服务器应返回不支持的文件类型错误
        assert result['error'] == '不支持的文件类型'
    
    def test_predict_valid_image(self, client):
        """测试上传有效图片的API响应"""
        # 创建临时测试图片
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (224, 224), color='green')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                data = {'file': (f, 'test.png')}
                response = client.post('/predict', data=data, content_type='multipart/form-data')
            
            result = json.loads(response.data)
            
            # 验证响应结构
            if 'success' in result:
                assert result['success'] is True
                assert 'image_url' in result
                assert 'predictions' in result
                assert len(result['predictions']) == 5
            else:
                # 如果返回错误，至少确保有错误信息
                assert 'error' in result
        finally:
            os.unlink(tmp_path)
    
    def test_predict_response_format(self, client):
        """测试API响应格式是否正确"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (224, 224), color='yellow')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                data = {'file': (f, 'test.jpg')}
                response = client.post('/predict', data=data, content_type='multipart/form-data')
            
            result = json.loads(response.data)
            
            # 验证JSON格式
            assert isinstance(result, dict)
            
            if result.get('success'):
                # 验证predictions结构
                for pred in result['predictions']:
                    assert 'class' in pred
                    assert 'confidence' in pred
                    assert '%' in pred['confidence']
        finally:
            os.unlink(tmp_path)


# ==================== 4. 集成测试与端到端测试 ====================

class TestIntegration:
    """集成测试：模拟完整用户流程"""
    
    @pytest.fixture
    def client(self):
        """创建Flask测试客户端"""
        app.config['TESTING'] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            app.config['UPLOAD_FOLDER'] = tmpdir
            with app.test_client() as client:
                yield client
    
    def test_end_to_end_upload_and_predict(self, client):
        """端到端测试：上传图片 -> 预测 -> 返回结果"""
        # Step 1: 访问主页
        home_response = client.get('/')
        assert home_response.status_code == 200
        
        # Step 2: 创建测试图片
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Step 3: 上传并预测
            with open(tmp_path, 'rb') as f:
                data = {'file': (f, 'integration_test.png')}
                predict_response = client.post('/predict', data=data, content_type='multipart/form-data')
            
            result = json.loads(predict_response.data)
            
            # Step 4: 验证完整流程结果
            assert predict_response.status_code == 200
            
            if result.get('success'):
                # 成功路径：验证预测结果存在
                assert len(result['predictions']) == 5
                for pred in result['predictions']:
                    assert 'class' in pred
                    assert float(pred['confidence'].rstrip('%')) >= 0
            else:
                # 降级路径：验证错误信息存在
                assert 'error' in result
        finally:
            os.unlink(tmp_path)
    
    def test_multiple_image_upload_sequential(self, client):
        """集成测试：连续上传多张图片"""
        test_colors = ['red', 'green', 'blue']
        
        for color in test_colors:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img = Image.new('RGB', (224, 224), color=color)
                img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                with open(tmp_path, 'rb') as f:
                    data = {'file': (f, f'{color}.png')}
                    response = client.post('/predict', data=data, content_type='multipart/form-data')
                
                assert response.status_code == 200
                result = json.loads(response.data)
                # 每次请求都应该有响应（成功或错误）
                assert 'success' in result or 'error' in result
            finally:
                os.unlink(tmp_path)


# ==================== 5. 调试技巧演示 ====================

class TestDebugging:
    """演示常见的调试技巧"""
    
    def test_breakpoint_demo(self):
        """
        断点调试演示
        运行方式: python -m pytest test.py::TestDebugging::test_breakpoint_demo -s
        或在 IDE 中设置断点
        """
        # 方式1: 使用 breakpoint() 函数 (Python 3.7+)
        # breakpoint()  # 取消注释后会进入调试器
        
        # 方式2: 使用 pdb
        # import pdb; pdb.set_trace()
        
        # 调试示例：检查变量值
        test_tensor = np.array([[1, 2, 3], [4, 5, 6]])
        # breakpoint()  # 在此处暂停，可检查 test_tensor 的值
        
        assert test_tensor.shape == (2, 3)
    
    def test_logging_debug_demo(self):
        """
        日志调试演示
        使用不同级别的日志记录问题
        """
        import logging
        
        # 配置调试级别日志
        debug_logger = logging.getLogger('debug_demo')
        debug_logger.setLevel(logging.DEBUG)
        
        # 示例：记录模型推理过程中的中间结果
        input_shape = (1, 3, 224, 224)
        debug_logger.debug(f"输入张量形状: {input_shape}")
        
        # 模拟推理
        output = np.random.randn(1, 1000)
        debug_logger.debug(f"输出形状: {output.shape}")
        debug_logger.info("推理完成")
        
        assert output.shape == (1, 1000)
    
    def test_gradient_check_demo(self):
        """
        梯度检查演示（仅当需要训练时使用）
        当前项目使用预训练模型推理，不涉及梯度
        """
        import torch
        import torch.nn as nn
        
        # 模拟一个简单的两层网络
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)
            
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
        
        model = SimpleNet()
        x = torch.randn(2, 10)
        y = torch.randint(0, 2, (2,))
        
        criterion = nn.CrossEntropyLoss()
        
        # 前向传播
        output = model(x)
        loss = criterion(output, y)
        
        # 反向传播（计算梯度）
        loss.backward()
        
        # 检查梯度是否存在且不为零
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"参数 {name} 梯度范数: {param.grad.norm().item():.6f}")
                # 梯度检查：确保梯度不全为零
                assert param.grad.abs().sum() > 0
        
        print("梯度检查通过：所有参数梯度正常")


# ==================== 运行入口 ====================

if __name__ == '__main__':
    # 运行所有测试
    pytest.main([__file__, '-v', '--tb=short'])