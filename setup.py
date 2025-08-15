import os
import sys

def setup_project():
    # 创建必要的目录
    directories = [
        'instance',
        os.path.join('static', 'temp'),
        os.path.join('static', 'generated'),
        os.path.join('static', 'uploads'),
        os.path.join('static', 'avatars')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # 创建空的数据库文件
    db_path = os.path.join('instance', 'users.db')
    if not os.path.exists(db_path):
        open(db_path, 'a').close()
        print(f"Created database file: {db_path}")
    
    print("Setup completed successfully!")

if __name__ == '__main__':
    setup_project() 