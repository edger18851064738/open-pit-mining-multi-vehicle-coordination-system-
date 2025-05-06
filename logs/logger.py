import os

def extract_last_5_percent(input_file_path, output_file_path):
    # 获取文件大小
    file_size = os.path.getsize(input_file_path)
    
    # 计算最后5%的起始位置
    start_position = int(file_size * 0.95)
    
    # 读取最后5%的内容并写入新文件
    with open(input_file_path, 'rb') as input_file:
        # 移动到起始位置
        input_file.seek(start_position)
        
        # 读取剩余内容
        content = input_file.read()
        
        # 写入新文件
        with open(output_file_path, 'wb') as output_file:
            output_file.write(content)
    
    print(f"已成功提取文件最后5%的内容到 {output_file_path}")
    print(f"原文件大小: {file_size} 字节")
    print(f"提取内容大小: {len(content)} 字节")

if __name__ == "__main__":
    input_file = r"C:\mine\reinforce\data\biz_road_conflict(8).log"
    output_file = r"C:\工作\biz_road_conflict(8)_last_5_percent.log"
    
    extract_last_5_percent(input_file, output_file)