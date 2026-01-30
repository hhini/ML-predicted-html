print("Setting up project...")
import os

# 定义我们要创建的目录结构
structure = {
    "app": ["ui", "services", "utils", "models"],
    "data": [],
    "tests": [],
}

# 定义我们要创建的空文件 (作为占位符)
files = [
    "app/main.py",
    "app/ui/__init__.py",
    "app/services/__init__.py",
    "app/utils/__init__.py",
    "app/utils/logger.py",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "README.md",
    ".gitignore"
]

def create_project():
    # 1. 创建文件夹
    for root, subdirs in structure.items():
        os.makedirs(root, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(root, subdir), exist_ok=True)
            print(f"✅ 创建目录: {os.path.join(root, subdir)}")

    # 2. 创建文件
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w', encoding='utf-8') as f:
                pass # 创建空文件
            print(f"📄 创建文件: {file}")
    
    print("\n🎉 项目骨架搭建完成！现在你可以开始编写代码了。")

if __name__ == "__main__":
    create_project()