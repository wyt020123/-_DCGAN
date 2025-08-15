# -*- coding: utf-8 -*-
from flask import (
    Flask, render_template, request, jsonify, redirect, 
    url_for, send_from_directory, flash, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
import os
import sys
import shutil
from werkzeug.utils import secure_filename
import bcrypt
from datetime import datetime, timedelta
import imghdr
import json
import logging
from functools import wraps
import traceback
import uuid
import requests
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 确保instance目录存在
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(instance_path, exist_ok=True)

app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(instance_path, "users.db")}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SECRET_KEY=os.urandom(24),
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    UPLOAD_EXTENSIONS=['.jpg', '.jpeg', '.png', '.gif'],
    TEMP_FOLDER=os.path.join('static', 'temp'),
    DEFAULT_AVATAR="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'%3E%3Crect width='200' height='200' fill='%23f3f4f6'/%3E%3Ccircle cx='100' cy='80' r='40' fill='%239ca3af'/%3E%3Cpath d='M160 200H40c0-40 30-60 60-60s60 20 60 60z' fill='%239ca3af'/%3E%3C/svg%3E",
    # LaoZhang AI API配置
    LAOZHANG_API_TOKEN=os.getenv('LAOZHANG_API_TOKEN', '')
)

# 创建临时文件夹
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录'
login_manager.login_message_category = 'info'

class User(UserMixin, db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    nickname = db.Column(db.String(150))
    avatar = db.Column(db.String(255))
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    generated_images = db.relationship('GeneratedImage', backref='user', lazy=True)
    favorite_images = db.relationship('FavoriteImage', backref='user', lazy=True)
    posts = db.relationship('Post', backref='author', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)

class GeneratedImage(db.Model):
    """生成的图片模型"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    style = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'style': self.style,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'user_id': self.user_id
        }

class FavoriteImage(db.Model):
    """收藏的图片模型"""
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('generated_image.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'image_id': self.image_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class GenerationTask(db.Model):
    """生成任务状态跟踪"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    progress = db.Column(db.Float, default=0.0)
    image_filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)

    def __init__(self, user_id):
        self.user_id = user_id
        self.status = 'pending'
        self.progress = 0.0

@login_manager.user_loader
def load_user(user_id):
    """加载用户"""
    return db.session.get(User, int(user_id))

def allowed_file(filename):
    """检查文件是否是允许的类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def validate_image(stream):
    """验证图片文件的有效性"""
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + format

def create_user_dirs(user_id):
    """创建用户所需的所有目录"""
    dirs = ['generated', 'favorites', 'uploads', 'avatars']
    for dir_name in dirs:
        path = os.path.join('static', dir_name, str(user_id))
        os.makedirs(path, exist_ok=True)
    return True

@app.route('/')
@login_required
def index():
    """首页路由"""
    try:
        # 确保用户目录存在
        create_user_dirs(current_user.id)
        
        # 获取用户的生成历史和收藏
        generated_images = GeneratedImage.query.filter_by(
            user_id=current_user.id
        ).order_by(GeneratedImage.created_at.desc()).all()
        
        favorite_images = FavoriteImage.query.filter_by(
            user_id=current_user.id
        ).order_by(FavoriteImage.created_at.desc()).all()
        
        # 确保生成的图片文件存在并转换为可序列化的字典
        valid_images = []
        for image in generated_images:
            image_path = os.path.join('static', 'generated', str(current_user.id), image.filename)
            if os.path.exists(image_path):
                valid_images.append(image.to_dict())
            else:
                logger.warning(f"图片文件不存在: {image_path}")
        
        # 转换收藏为可序列化的字典
        favorite_images_dict = [fav.to_dict() for fav in favorite_images]
        
        return render_template(
            'index.html',
            images=valid_images,
            favorite_images=favorite_images_dict,
            default_avatar=app.config['DEFAULT_AVATAR']
        )
            
    except Exception as e:
        logger.error(f"加载首页时发生错误: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return render_template(
            'error.html', 
            message="加载页面时发生错误",
            error_details=str(e) if app.debug else None
        ), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录路由"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        if not username or not password:
            flash('请输入用户名和密码', 'error')
            return render_template('login.html')
            
        try:
            user = User.query.filter_by(username=username).first()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                next_page = request.args.get('next')
                if not next_page or not url_has_allowed_host_and_scheme(next_page):
                    next_page = url_for('index')
                    
                return redirect(next_page)
            else:
                flash('用户名或密码错误', 'error')
        except Exception as e:
            logger.error(f"登录失败: {str(e)}")
            logger.error(traceback.format_exc())
            flash('登录时发生错误，请稍后重试', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册路由"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password or not confirm_password:
            flash('请填写所有必填字段', 'error')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('两次输入的密码不一致', 'error')
            return render_template('register.html')
            
        if len(password) < 6:
            flash('密码长度必须至少为6个字符', 'error')
            return render_template('register.html')
            
        try:
            if User.query.filter_by(username=username).first():
                flash('用户名已存在', 'error')
                return render_template('register.html')
                
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            new_user = User(
                username=username,
                password=hashed_password.decode('utf-8'),
                created_at=datetime.utcnow()
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            # 创建用户目录
            create_user_dirs(new_user.id)
            
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"注册失败: {str(e)}")
            logger.error(traceback.format_exc())
            flash('注册时发生错误，请稍后重试', 'error')
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def url_has_allowed_host_and_scheme(url):
    """检查URL是否是允许的格式"""
    if not url:
        return False
    # 只允许相对URL
    return not url.startswith(('http://', 'https://'))

@app.route('/user_center')
@login_required
def user_center():
    try:
        generated_images = GeneratedImage.query.filter_by(
            user_id=current_user.id
        ).order_by(GeneratedImage.created_at.desc()).all()
        
        favorite_images = FavoriteImage.query.filter_by(
            user_id=current_user.id
        ).order_by(FavoriteImage.created_at.desc()).all()
        
        # 让 favorite_images 变成 [{'image': <GeneratedImage对象>, 'favorite': <FavoriteImage对象>}, ...]
        fav_images = []
        for fav in favorite_images:
            img = GeneratedImage.query.get(fav.image_id)
            if img:
                fav_images.append({'image': img, 'favorite': fav})
        
        return render_template(
            'user_center.html',
            user=current_user,
            images=generated_images,
            favorite_images=fav_images,
            default_avatar=app.config['DEFAULT_AVATAR']
        )
    except Exception as e:
        logger.error(f"加载个人中心失败: {str(e)}")
        flash('加载个人中心失败', 'error')
        return render_template('error.html', message="加载个人中心失败"), 500

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '无效的请求数据'}), 400

        # 更新用户信息
        if 'nickname' in data:
            current_user.nickname = data['nickname']
        if 'bio' in data:
            current_user.bio = data['bio']

        db.session.commit()
        return jsonify({'status': 'success', 'message': '更新成功'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '更新失败'}), 500

@app.route('/update_avatar', methods=['POST'])
@login_required
def update_avatar():
    try:
        if 'avatar' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传文件'}), 400

        file = request.files['avatar']
        if not file or not file.filename:
            return jsonify({'status': 'error', 'message': '没有选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': '不支持的文件类型'}), 400

        # 保存头像
        filename = f"avatar_{current_user.id}_{int(datetime.utcnow().timestamp())}{os.path.splitext(file.filename)[1]}"
        avatar_path = os.path.join('static', 'avatars', filename)
        os.makedirs(os.path.dirname(avatar_path), exist_ok=True)
        
        file.save(avatar_path)
        
        # 更新用户头像
        current_user.avatar = url_for('static', filename=f'avatars/{filename}')
        db.session.commit()

        return jsonify({'status': 'success', 'message': '头像更新成功'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '头像更新失败'}), 500

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '无效的请求数据'}), 400

        current_password = data.get('current_password')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        if not current_password or not new_password or not confirm_password:
            return jsonify({'status': 'error', 'message': '请填写所有必填字段'}), 400

        if new_password != confirm_password:
            return jsonify({'status': 'error', 'message': '两次输入的新密码不一致'}), 400

        # 验证当前密码
        if not bcrypt.checkpw(current_password.encode('utf-8'), current_user.password.encode('utf-8')):
            return jsonify({'status': 'error', 'message': '当前密码错误'}), 400

        # 更新密码
        current_user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.session.commit()

        return jsonify({'status': 'success', 'message': '密码修改成功'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '密码修改失败'}), 500

@app.route('/generate_from_text', methods=['POST'])
@login_required
def generate_from_text():
    """根据文本描述生成头像"""
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'status': 'error', 'message': '请提供描述文本'}), 400

        prompt = data['prompt']
        
        # 使用 LaoZhang AI API 生成图片
        API_URL = "https://api.laozhang.ai/v1/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app.config['LAOZHANG_API_TOKEN']}"
        }
        
        payload = {
            "model": "dall-e-3",
            "prompt": f"anime style, simple and clean, minimalist, {prompt}, anime character, soft colors, solid white background, no grid background, no sketch lines, clean lines",
            "n": 1,
            "size": "1024x1024"
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"LaoZhang AI API 错误: {response.text}")
            return jsonify({
                'status': 'error',
                'message': '生成图片失败，请稍后重试'
            }), 500

        api_response_data = response.json()
        if not api_response_data or 'data' not in api_response_data or not api_response_data['data']:
            logger.error("LaoZhang AI API 返回结果无效")
            return jsonify({
                'status': 'error',
                'message': '生成图片失败，请稍后重试'
            }), 500
            
        image_url = api_response_data['data'][0]['url']

        # 确保用户目录存在
        user_dir = os.path.join('static', 'generated', str(current_user.id))
        os.makedirs(user_dir, exist_ok=True)

        # 保存生成的图片
        filename = f"avatar_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(user_dir, filename)
        
        # 下载图片
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            logger.error("下载生成的图片失败")
            return jsonify({
                'status': 'error',
                'message': '保存图片失败，请重试'
            }), 500

        # 保存图片
        with open(filepath, 'wb') as f:
            f.write(image_response.content)

        # 创建数据库记录
        image = GeneratedImage(
            filename=filename,
            user_id=current_user.id,
            style='text_to_image'
        )
        db.session.add(image)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'id': image.id,
            'filename': filename,
            'url': url_for('static', filename=f'generated/{current_user.id}/{filename}')
        })

    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': '生成图片时发生错误，请重试'
        }), 500

@app.route('/generate_from_image', methods=['POST'])
@login_required
def generate_from_image():
    """
    上传图片+描述生成头像（Image to Image）
    """
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': '请上传图片'}), 400

        image_file = request.files['image']
        prompt = request.form.get('prompt', '')

        if not image_file or not image_file.filename:
            return jsonify({'status': 'error', 'message': '没有选择文件'}), 400

        # 将图片文件转换为Base64编码
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 使用 LaoZhang AI API 生成图片
        API_URL = "https://api.laozhang.ai/v1/images/generations"  # 修改为正确的API端点
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app.config['LAOZHANG_API_TOKEN']}"
        }
        
        # 修改请求格式
        payload = {
            "model": "dall-e-3",
            "prompt": f"anime style, simple and clean, minimalist, {prompt}, anime character, soft colors, solid white background, no grid background, no sketch lines, clean lines",
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
            "image_data": image_base64
        }
        
        # 添加重试机制
        max_retries = 3
        retry_delay = 1  # 初始重试延迟（秒）
        
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试连接到 API (尝试 {attempt + 1}/{max_retries})")
                logger.info(f"请求URL: {API_URL}")
                logger.info(f"请求头: {headers}")
                logger.info(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
                
                response = requests.post(
                    API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=60,  # 增加超时时间
                    verify=True  # 验证SSL证书
                )
                
                # 记录响应信息
                logger.info(f"API响应状态码: {response.status_code}")
                logger.info(f"API响应头: {dict(response.headers)}")
                logger.info(f"API响应内容: {response.text[:1000]}")  # 只记录前1000个字符
                
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"API 请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        if response.status_code != 200:
            logger.error(f"LaoZhang AI API 错误: {response.text}")
            return jsonify({
                'status': 'error',
                'message': '生成图片失败，请稍后重试'
            }), 500

        api_response_data = response.json()
        
        # 检查响应格式
        if not api_response_data or 'data' not in api_response_data or not api_response_data['data']:
            logger.error(f"LaoZhang AI API 返回结果无效: {api_response_data}")
            return jsonify({
                'status': 'error',
                'message': '生成图片失败，API返回结果无效'
            }), 500
            
        image_url = api_response_data['data'][0]['url']

        # 确保用户目录存在
        user_dir = os.path.join('static', 'generated', str(current_user.id))
        os.makedirs(user_dir, exist_ok=True)

        # 保存生成的图片
        filename = f"avatar_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(user_dir, filename)
        
        # 下载图片
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(image_response.content)
        except Exception as img_save_error:
            logger.error(f"保存生成的图片失败: {str(img_save_error)}")
            return jsonify({
                'status': 'error',
                'message': '保存图片失败，请重试'
            }), 500

        # 创建数据库记录
        image = GeneratedImage(
            filename=filename,
            user_id=current_user.id,
            style='image_to_image'
        )
        db.session.add(image)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'id': image.id,
            'filename': filename,
            'url': url_for('static', filename=f'generated/{current_user.id}/{filename}')
        })

    except requests.exceptions.ConnectionError as e:
        logger.error(f"连接 API 失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '无法连接到图片生成服务，请检查网络连接后重试'
        }), 500
    except requests.exceptions.Timeout as e:
        logger.error(f"API 请求超时: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': '请求超时，请稍后重试'
        }), 500
    except Exception as e:
        logger.error(f"生成图片失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': '生成图片时发生错误，请重试'
        }), 500

@app.route('/favorite', methods=['POST'])
@login_required
def favorite():
    try:
        data = request.get_json()
        if not data or 'image_id' not in data:
            return jsonify({'status': 'error', 'message': '无效的请求数据'}), 400

        image_id = data['image_id']
        image = GeneratedImage.query.get(image_id)
        
        if not image:
            return jsonify({'status': 'error', 'message': '图片不存在'}), 404
            
        if image.user_id != current_user.id:
            return jsonify({'status': 'error', 'message': '无权操作此图片'}), 403

        # 检查是否已经收藏
        existing_favorite = FavoriteImage.query.filter_by(
            user_id=current_user.id,
            image_id=image_id
        ).first()

        if existing_favorite:
            db.session.delete(existing_favorite)
            message = '取消收藏成功'
        else:
            favorite = FavoriteImage(
                user_id=current_user.id,
                image_id=image_id
            )
            db.session.add(favorite)
            message = '收藏成功'

        db.session.commit()
        return jsonify({'status': 'success', 'message': message})

    except Exception as e:
        print(f"Error in favorite: {e}")
        return jsonify({'status': 'error', 'message': '操作失败'}), 500

@app.route('/delete_image/<int:image_id>', methods=['DELETE'])
@login_required
def delete_image(image_id):
    try:
        image = GeneratedImage.query.get_or_404(image_id)
        
        if image.user_id != current_user.id:
            return jsonify({'status': 'error', 'message': '无权删除此图片'}), 403

        # 删除收藏记录
        FavoriteImage.query.filter_by(image_id=image_id).delete()
        
        # 删除文件
        file_path = os.path.join('static', 'generated', str(current_user.id), image.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # 删除数据库记录
        db.session.delete(image)
        db.session.commit()
        
        return jsonify({'status': 'success', 'message': '删除成功'})
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting image: {e}")
        return jsonify({'status': 'error', 'message': '删除失败'}), 500

@app.route('/generation_status/<task_id>')
@login_required
def generation_status(task_id):
    try:
        task = GenerationTask.query.get(task_id)
        if task:
            return jsonify({
                'status': task.status,
                'progress': task.progress,
                'image_url': url_for('static', filename=f'generated/{task.image_filename}') if task.status == 'completed' else None
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generation_history')
@login_required
def generation_history():
    """生成历史页面"""
    try:
        generated_images = GeneratedImage.query.filter_by(
            user_id=current_user.id
        ).order_by(GeneratedImage.created_at.desc()).all()
        
        # 获取用户收藏的图片ID列表
        favorite_ids = [fav.image_id for fav in FavoriteImage.query.filter_by(
            user_id=current_user.id
        ).all()]
        
        return render_template(
            'generation_history.html',
            images=generated_images,
            favorite_ids=favorite_ids
        )
    except Exception as e:
        logger.error(f"加载生成历史失败: {str(e)}")
        flash('加载生成历史失败', 'error')
        return render_template('error.html', message="加载生成历史失败"), 500

@app.route('/favorites')
@login_required
def favorites():
    """收藏页面"""
    try:
        favorite_images = FavoriteImage.query.filter_by(
            user_id=current_user.id
        ).order_by(FavoriteImage.created_at.desc()).all()
        
        # 让 favorite_images 变成 [{'image': <GeneratedImage对象>, 'favorite': <FavoriteImage对象>}, ...]
        fav_images = []
        for fav in favorite_images:
            img = GeneratedImage.query.get(fav.image_id)
            if img:
                fav_images.append({'image': img, 'favorite': fav})
        
        return render_template(
            'favorites.html',
            favorite_images=fav_images
        )
    except Exception as e:
        logger.error(f"加载收藏失败: {str(e)}")
        flash('加载收藏失败', 'error')
        return render_template('error.html', message="加载收藏失败"), 500

@app.route('/download_image/<int:image_id>')
@login_required
def download_image(image_id):
    """下载生成的图片"""
    try:
        image = GeneratedImage.query.get_or_404(image_id)
        
        if image.user_id != current_user.id:
            return jsonify({'status': 'error', 'message': '无权下载此图片'}), 403

        # 构建文件路径
        file_path = os.path.join('static', 'generated', str(current_user.id), image.filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404

        # 获取文件扩展名
        _, ext = os.path.splitext(image.filename)
        
        # 设置下载文件名
        download_filename = f"avatar_{image.id}{ext}"
        
        return send_from_directory(
            os.path.dirname(file_path),
            os.path.basename(file_path),
            as_attachment=True,
            download_name=download_filename
        )

    except Exception as e:
        logger.error(f"下载图片失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': '下载失败'}), 500

@app.route('/batch_delete_images', methods=['POST'])
@login_required
def batch_delete_images():
    """批量删除图片"""
    try:
        data = request.get_json()
        if not data or 'image_ids' not in data:
            return jsonify({'status': 'error', 'message': '无效的请求数据'}), 400

        image_ids = data['image_ids']
        if not isinstance(image_ids, list):
            return jsonify({'status': 'error', 'message': '无效的图片ID列表'}), 400

        # 获取所有要删除的图片
        images = GeneratedImage.query.filter(
            GeneratedImage.id.in_(image_ids),
            GeneratedImage.user_id == current_user.id
        ).all()

        if not images:
            return jsonify({'status': 'error', 'message': '未找到要删除的图片'}), 404

        # 删除收藏记录
        FavoriteImage.query.filter(
            FavoriteImage.image_id.in_(image_ids)
        ).delete(synchronize_session=False)

        # 删除文件
        for image in images:
            file_path = os.path.join('static', 'generated', str(current_user.id), image.filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {str(e)}")

        # 删除数据库记录
        for image in images:
            db.session.delete(image)

        db.session.commit()
        return jsonify({'status': 'success', 'message': '删除成功'})

    except Exception as e:
        db.session.rollback()
        logger.error(f"批量删除图片失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': '删除失败'}), 500

# 错误处理器
@app.errorhandler(Exception)
def handle_error(error):
    """全局错误处理器"""
    error_message = str(error)
    logger.error(f"发生错误: {error_message}")
    return jsonify({
        'success': False,
        'error': error_message
    }), 500

@app.errorhandler(500)
def internal_error(error):
    """500错误处理器"""
    logger.error(f"500错误: {str(error)}")
    logger.error(f"错误详情: {traceback.format_exc()}")
    return render_template(
        'error.html',
        message="服务器内部错误",
        error_details=str(error) if app.debug else None
    ), 500

@app.errorhandler(404)
def not_found_error(error):
    """404错误处理器"""
    logger.error(f"404错误: {str(error)}")
    return render_template(
        'error.html',
        message="页面未找到",
        error_details=str(error) if app.debug else None
    ), 404

class Post(db.Model):
    """用户分享的图片帖子模型"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    likes_count = db.Column(db.Integer, default=0)
    comments_count = db.Column(db.Integer, default=0)
    
    # Relationships
    comments = db.relationship('Comment', backref='post', lazy=True, cascade='all, delete-orphan')
    likes = db.relationship('PostLike', backref='post', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'image_path': self.image_path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id,
            'author': {
                'id': self.author.id,
                'username': self.author.username,
                'nickname': self.author.nickname,
                'avatar': self.author.avatar
            },
            'likes_count': self.likes_count,
            'comments_count': self.comments_count
        }

class Comment(db.Model):
    """评论模型"""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'))
    
    # Relationships
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id,
            'post_id': self.post_id,
            'parent_id': self.parent_id,
            'author': {
                'id': self.author.id,
                'username': self.author.username,
                'nickname': self.author.nickname,
                'avatar': self.author.avatar
            }
        }

class PostLike(db.Model):
    """帖子点赞模型"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'post_id', name='unique_post_like'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'post_id': self.post_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

@app.route('/square')
@login_required
def square():
    """广场首页"""
    page = request.args.get('page', 1, type=int)
    per_page = 12
    
    # 获取广场帖子
    posts = Post.query.order_by(Post.created_at.desc()).paginate(page=page, per_page=per_page)
    
    # 获取用户的收藏图片，并关联GeneratedImage
    favorite_images = db.session.query(FavoriteImage, GeneratedImage).\
        join(GeneratedImage, FavoriteImage.image_id == GeneratedImage.id).\
        filter(FavoriteImage.user_id == current_user.id).\
        all()
    
    # 获取用户的生成历史
    generated_images = GeneratedImage.query.filter_by(user_id=current_user.id).order_by(GeneratedImage.created_at.desc()).all()
    
    return render_template('square.html', 
                         posts=posts,
                         favorite_images=favorite_images,
                         generated_images=generated_images)

@app.route('/post/create', methods=['POST'])
@login_required
def create_post():
    try:
        title = request.form.get('title')
        description = request.form.get('description')
        
        if not title:
            return jsonify({'status': 'error', 'message': '标题不能为空'})
        
        # 检查是否有上传的图片或选择了已有图片
        if 'image' in request.files:
            # 处理新上传的图片
            image = request.files['image']
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                # 保存到用户目录
                user_dir = os.path.join(app.static_folder, 'generated', str(current_user.id))
                os.makedirs(user_dir, exist_ok=True)
                image_path = os.path.join(user_dir, filename)
                image.save(image_path)
                # 设置图片路径
                image_path = f'generated/{current_user.id}/{filename}'
            else:
                return jsonify({'status': 'error', 'message': '不支持的图片格式'})
        elif 'image_path' in request.form:
            # 使用已有的图片
            image_path = request.form.get('image_path')
            # 验证图片路径是否属于当前用户
            if not image_path.startswith(f'generated/{current_user.id}/'):
                return jsonify({'status': 'error', 'message': '无权使用该图片'})
        else:
            return jsonify({'status': 'error', 'message': '请选择或上传图片'})
        
        # 创建新帖子
        post = Post(
            title=title,
            description=description,
            image_path=image_path,
            user_id=current_user.id
        )
        
        db.session.add(post)
        db.session.commit()
        
        return jsonify({'status': 'success', 'message': '发布成功'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"发布帖子失败: {str(e)}")
        return jsonify({'status': 'error', 'message': '发布失败，请重试'})

@app.route('/post/<int:post_id>')
@login_required
def view_post(post_id):
    """查看帖子详情"""
    post = Post.query.get_or_404(post_id)
    return render_template('post_detail.html', post=post)

@app.route('/post/<int:post_id>/like', methods=['POST'])
@login_required
def like_post(post_id):
    """点赞/取消点赞帖子"""
    try:
        post = Post.query.get_or_404(post_id)
        existing_like = PostLike.query.filter_by(
            user_id=current_user.id,
            post_id=post_id
        ).first()
        
        if existing_like:
            # 取消点赞
            db.session.delete(existing_like)
            post.likes_count -= 1
            action = 'unliked'
        else:
            # 添加点赞
            like = PostLike(user_id=current_user.id, post_id=post_id)
            db.session.add(like)
            post.likes_count += 1
            action = 'liked'
            
        db.session.commit()
        return jsonify({
            'action': action,
            'likes_count': post.likes_count
        })
        
    except Exception as e:
        logger.error(f"点赞操作失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '操作失败'}), 500

@app.route('/post/<int:post_id>/comment', methods=['POST'])
@login_required
def comment_post(post_id):
    """评论帖子"""
    try:
        content = request.form.get('content', '').strip()
        parent_id = request.form.get('parent_id', type=int)
        
        if not content:
            return jsonify({'error': '评论内容不能为空'}), 400
            
        post = Post.query.get_or_404(post_id)
        
        comment = Comment(
            content=content,
            user_id=current_user.id,
            post_id=post_id,
            parent_id=parent_id if parent_id else None
        )
        
        post.comments_count += 1
        db.session.add(comment)
        db.session.commit()
        
        return jsonify(comment.to_dict()), 201
        
    except Exception as e:
        logger.error(f"评论失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '评论失败'}), 500

@app.route('/post/<int:post_id>/delete', methods=['DELETE'])
@login_required
def delete_post(post_id):
    """删除帖子（仅作者可删除）"""
    try:
        post = Post.query.get_or_404(post_id)
        
        if post.user_id != current_user.id:
            return jsonify({'error': '没有权限删除此帖子'}), 403
            
        # 删除帖子图片
        # try:
        #     image_path = os.path.join('static', post.image_path)
        #     if os.path.exists(image_path):
        #         os.remove(image_path)
        # except Exception as e:
        #     logger.error(f"删除帖子图片失败: {str(e)}")
            
        # 删除帖子记录
        db.session.delete(post)
        db.session.commit()
        
        return jsonify({'message': '帖子已删除'})
        
    except Exception as e:
        logger.error(f"删除帖子失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '删除帖子失败'}), 500

@app.route('/comment/<int:comment_id>/delete', methods=['DELETE'])
@login_required
def delete_comment(comment_id):
    """删除评论（仅作者可删除）"""
    try:
        comment = Comment.query.get_or_404(comment_id)
        
        if comment.user_id != current_user.id:
            return jsonify({'error': '没有权限删除此评论'}), 403
            
        post = comment.post
        post.comments_count -= 1
        
        db.session.delete(comment)
        db.session.commit()
        
        return jsonify({'message': '评论已删除'})
        
    except Exception as e:
        logger.error(f"删除评论失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': '删除评论失败'}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
            os.makedirs(db_dir, exist_ok=True)
            
            # 创建数据库表
            db.create_all()
            logger.info("数据库初始化成功！")
        except Exception as e:
            logger.error(f"数据库初始化错误: {str(e)}")
            logger.error(traceback.format_exc())
            
    app.run(debug=True)
