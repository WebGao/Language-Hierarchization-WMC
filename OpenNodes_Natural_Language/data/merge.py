import os

def merge_all_txt(output_filename="merged_all.txt"):
    """
    合并当前目录下所有 .txt 文件到一个输出文件
    :param output_filename: 输出合并文件的名称，默认是 merged_all.txt
    """
    # 获取当前目录下所有 .txt 文件（排除输出文件本身，避免重复合并）
    txt_files = [f for f in os.listdir(".") if f.endswith(".txt") and f != output_filename]
    
    if not txt_files:
        print("当前目录下没有找到 .txt 文件！")
        return
    
    # 按文件名排序（可选，如需按创建时间排序可修改此处）
    txt_files.sort()
    
    # 合并文件：使用 utf-8 编码避免中文乱码
    with open(output_filename, "w", encoding="utf-8") as out_file:
        for idx, file in enumerate(txt_files, 1):
            # 写入文件名作为分隔标识（便于区分不同文件内容）
            out_file.write(f"=== 开始合并文件：{file} ===\n\n")
            try:
                # 读取单个 txt 文件（兼容不同编码，优先 utf-8，失败则用 gbk）
                with open(file, "r", encoding="utf-8") as in_file:
                    out_file.write(in_file.read())
            except UnicodeDecodeError:
                with open(file, "r", encoding="gbk") as in_file:
                    out_file.write(in_file.read())
            # 写入文件结束标识和空行分隔
            out_file.write(f"\n\n=== 文件 {file} 合并完成 ===\n{'='*50}\n\n")
    
    print(f"合并完成！共处理 {len(txt_files)} 个 .txt 文件")
    print(f"合并后的文件：{output_filename}")

# 执行合并（默认输出为 merged_all.txt，可修改参数自定义名称）
merge_all_txt()