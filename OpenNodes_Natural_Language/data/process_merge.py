import os
import re

def clean_book_text(file_path):
    """
    清理单本书，返回正文字符串
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 删除开头版权信息、书名、作者等
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", text)
    if start_match:
        text = text[start_match.end():]  # 保留 START 后的内容

    # 删除 Contents 目录（匹配 Contents 到正文章节）
    text = re.sub(r"(?i)contents.*?(Chapter\s+[IVXLCDM]+\.|CHAPTER\s+[IVXLCDM]+\.)", r"\1", text, flags=re.DOTALL)

    # 标准化章节开头
    chapter_pattern = re.compile(r"(CHAPTER\s+[IVXLCDM]+\.|Chapter\s+[IVXLCDM]+\.)", re.IGNORECASE)
    chapters = chapter_pattern.split(text)

    clean_text = ""
    for i in range(1, len(chapters), 2):
        chapter_body = chapters[i+1].strip()
        
        # 按段落分割，段落之间保留空行
        paragraphs = chapter_body.split("\n\n")
        clean_paragraphs = []
        for p in paragraphs:
            # 段落内部换行替换为空格，去掉首尾空白
            lines = [line.strip() for line in p.splitlines() if line.strip()]
            if lines:
                clean_paragraphs.append(" ".join(lines))
        clean_text += "\n\n".join(clean_paragraphs) + "\n\n"

    return clean_text

def merge_all_books(input_dir="./", output_file="merged_all.txt"):
    """
    合并所有清理后的书，保存到一个文件，并统计句子数
    """
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    txt_files.sort()

    merged_text = ""
    for txt_file in txt_files:
        path = os.path.join(input_dir, txt_file)
        print(f"Processing {txt_file} ...")
        book_text = clean_book_text(path)
        merged_text += book_text + "\n\n"

    # 删除合并标记（如 === 文件xxx 合并完成 === 等）
    # merged_text = re.sub(r"===.*?===\s*", "", merged_text)

    # 统计句子数（英文句子用 . ! ? 分隔）
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", merged_text))

    # 保存合并文件
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(merged_text)

    print(f"All books merged into {output_file}")
    print(f"Total sentences: {sentence_count}")

# 执行
merge_all_books()
