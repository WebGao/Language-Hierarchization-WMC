import json
import os
import re
from typing import List, Dict, Optional
import requests
import time

class DeepSeekTranslator:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """
        初始化DeepSeek翻译器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL，默认为DeepSeek官方API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def read_file(self, file_path: str) -> str:
        """
        读取本地文本文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 未找到")
            raise
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            raise
    
    def split_into_chapters(self, text: str) -> List[Dict[str, str]]:
        """
        将文本按照 CHAPTER I.、CHAPTER II. 等格式划分章节
        
        Args:
            text: 完整文本内容
            
        Returns:
            章节列表，每个章节包含标题和内容
        """
        chapters = []
        
        # 优化章节分割模式：匹配 CHAPTER I.、CHAPTER II. 等格式
        # 支持罗马数字 I, II, III, IV, V, VI, VII, VIII, IX, X 等
        chapter_pattern = r'(CHAPTER\s+[IVXLCDM]+\.)'
        
        # 查找所有章节标题的位置
        matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
        
        if not matches:
            print("未检测到 CHAPTER X. 格式的章节，尝试其他格式...")
            # 尝试其他可能的格式
            alternative_patterns = [
                r'(CHAPTER\s+\d+\.)',  # CHAPTER 1.
                r'(Chapter\s+[IVXLCDM]+\.)',  # Chapter I.
                r'(Chapter\s+\d+\.)',  # Chapter 1.
            ]
            
            for pattern in alternative_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    print(f"检测到格式: {pattern}")
                    break
        
        if not matches:
            print("未检测到标准章节格式，将整个文本作为一章处理")
            return [{
                'title': '完整文本',
                'content': text.strip(),
                'chapter_number': 1
            }]
        
        print(f"检测到 {len(matches)} 个章节")
        
        # 根据章节标题分割文本
        for i, match in enumerate(matches):
            chapter_title = match.group().strip()
            
            # 获取章节开始位置
            start_pos = match.start()
            
            # 确定章节内容范围
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            chapter_content = text[start_pos:end_pos].strip()
            
            # 提取罗马数字
            roman_num = re.search(r'[IVXLCDM]+', chapter_title, re.IGNORECASE)
            if roman_num:
                chapter_number = self._roman_to_int(roman_num.group())
            else:
                chapter_number = i + 1
            
            chapters.append({
                'title': chapter_title,
                'content': chapter_content,
                'chapter_number': chapter_number,
                'roman_num': roman_num.group() if roman_num else str(i + 1)
            })
            
            print(f"  章节 {i + 1}: {chapter_title} (罗马数字: {roman_num.group() if roman_num else 'N/A'})")
        
        return chapters
    
    def _roman_to_int(self, roman: str) -> int:
        """
        罗马数字转换为整数
        
        Args:
            roman: 罗马数字字符串
            
        Returns:
            对应的整数值
        """
        roman_dict = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        roman = roman.upper()
        total = 0
        prev_value = 0
        
        for char in reversed(roman):
            value = roman_dict.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
            
        return total
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        调用DeepSeek API翻译文本
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言（如：中文、法语、西班牙语等）
            
        Returns:
            翻译后的文本
        """
        # 构建API请求
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""请将以下英文文本翻译成{target_language}。

翻译要求：
1. 保持原文的格式、段落、章节标题和标点符号
2. 人名、地名等专有名词保持原样，首次出现时可在括号内注明音译
3. 确保翻译准确自然，符合{target_language}的表达习惯
4. 不要添加任何额外的解释、说明或注释
5. 保持CHAPTER标题格式不变，只翻译内容

英文文本：
{text}

{target_language}翻译："""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # 降低温度以获得更稳定的翻译
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result['choices'][0]['message']['content'].strip()
            
            # 确保翻译以CHAPTER标题开始
            if not translated_text.startswith("CHAPTER"):
                # 查找CHAPTER标题
                chapter_match = re.search(r'(CHAPTER\s+[IVXLCDM]+\.)', text)
                if chapter_match:
                    chapter_title = chapter_match.group()
                    if chapter_title not in translated_text:
                        translated_text = f"{chapter_title}\n\n{translated_text}"
            
            return translated_text
            
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"响应状态码: {e.response.status_code}")
                try:
                    error_detail = e.response.json()
                    print(f"错误详情: {error_detail}")
                except:
                    print(f"响应内容: {e.response.text[:200]}")
            raise
        except KeyError as e:
            print(f"解析API响应时出错: {e}")
            print(f"API响应: {result}")
            raise
        except Exception as e:
            print(f"翻译过程中发生错误: {e}")
            raise
    
    def translate_chapter(self, chapter: Dict, target_language: str, 
                         max_chunk_length: int = 2500) -> Dict:
        """
        翻译单个章节，处理长文本分块
        
        Args:
            chapter: 章节字典
            target_language: 目标语言
            max_chunk_length: 每个翻译块的最大长度
            
        Returns:
            包含翻译后内容的章节字典
        """
        print(f"\n正在翻译: {chapter['title']} ({len(chapter['content'])} 字符)")
        
        content = chapter['content']
        
        # 如果内容过长，分块翻译
        if len(content) > max_chunk_length:
            print(f"  章节内容较长，将分块翻译...")
            chunks = self._split_into_chunks(content, max_chunk_length)
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"  翻译块 {i + 1}/{len(chunks)}...")
                try:
                    # 添加延迟避免API限制
                    if i > 0:
                        time.sleep(1)
                    
                    translated_chunk = self.translate_text(chunk, target_language)
                    translated_chunks.append(translated_chunk)
                except Exception as e:
                    print(f"  翻译块 {i + 1} 失败: {e}")
                    # 如果翻译失败，保留原文并标记
                    translated_chunks.append(f"[翻译失败部分，保留原文]\n{chunk}")
                
            translated_content = '\n\n'.join(translated_chunks)
        else:
            translated_content = self.translate_text(content, target_language)
        
        # 提取翻译后的标题
        translated_title = self._extract_translated_title(translated_content, chapter['title'])
        
        return {
            'original_title': chapter['title'],
            'translated_title': translated_title,
            'original_content': content,
            'translated_content': translated_content,
            'chapter_number': chapter['chapter_number'],
            'roman_num': chapter.get('roman_num', str(chapter['chapter_number']))
        }
    
    def _extract_translated_title(self, translated_content: str, original_title: str) -> str:
        """
        从翻译内容中提取章节标题
        
        Args:
            translated_content: 翻译后的内容
            original_title: 原始标题
            
        Returns:
            提取的标题
        """
        # 尝试从翻译内容中提取CHAPTER标题
        title_patterns = [
            r'(CHAPTER\s+[IVXLCDM]+\.)',
            r'(第\s*[一二三四五六七八九十]+\s*章)',
            r'(Chapter\s+[IVXLCDM]+\.)',
            r'(章节\s+[IVXLCDM]+\.)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, translated_content, re.IGNORECASE)
            if match:
                return match.group()
        
        # 如果没有找到，返回原始标题
        return original_title
    
    def _split_into_chunks(self, text: str, max_length: int) -> List[str]:
        """
        将文本分割成适合翻译的块，保持段落完整
        
        Args:
            text: 原始文本
            max_length: 每个块的最大长度
            
        Returns:
            文本块列表
        """
        chunks = []
        
        # 按段落分割
        paragraphs = re.split(r'(\n\s*\n)', text)
        
        current_chunk = ""
        for i in range(0, len(paragraphs), 2):
            paragraph = paragraphs[i]
            separator = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            
            # 如果当前段落加上分隔符的长度不超过限制
            if len(current_chunk) + len(paragraph) + len(separator) <= max_length:
                current_chunk += paragraph + separator
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果单个段落就超过最大长度，按句子分割
                if len(paragraph) > max_length:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 2 <= max_length:
                            if temp_chunk:
                                temp_chunk += ' ' + sentence
                            else:
                                temp_chunk = sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    current_chunk = separator
                else:
                    current_chunk = paragraph + separator
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def save_translation(self, translated_chapters: List[Dict], 
                        output_dir: str, target_language: str, 
                        save_format: str = 'both'):
        """
        保存翻译结果
        
        Args:
            translated_chapters: 翻译后的章节列表
            output_dir: 输出目录
            target_language: 目标语言
            save_format: 保存格式 ('txt', 'json', 或 'both')
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        safe_lang = re.sub(r'[^\w\s-]', '', target_language).replace(' ', '_').lower()
        base_filename = f"alice_{safe_lang}"
        
        # 保存完整翻译文件
        if save_format in ['txt', 'both']:
            txt_path = os.path.join(output_dir, f"{base_filename}_complete.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                for chapter in translated_chapters:
                    f.write(f"{chapter['translated_title']}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(chapter['translated_content'])
                    f.write("\n\n" + "=" * 80 + "\n\n")
            
            print(f"\n完整翻译已保存为TXT文件: {txt_path}")
        
        # 保存JSON格式（包含原文和译文）
        if save_format in ['json', 'both']:
            json_path = os.path.join(output_dir, f"{base_filename}_bilingual.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(translated_chapters, f, ensure_ascii=False, indent=2)
            
            print(f"双语对照已保存为JSON文件: {json_path}")
        
        # 保存纯译文文件（无额外分隔符）
        txt_clean_path = os.path.join(output_dir, f"{base_filename}_clean.txt")
        with open(txt_clean_path, 'w', encoding='utf-8') as f:
            for chapter in translated_chapters:
                f.write(chapter['translated_content'])
                f.write("\n\n")
        
        print(f"纯净译文已保存为: {txt_clean_path}")
        
        # 同时保存单个章节文件
        chapters_dir = os.path.join(output_dir, "individual_chapters")
        os.makedirs(chapters_dir, exist_ok=True)
        
        print(f"\n正在保存单个章节文件...")
        for chapter in translated_chapters:
            chapter_num = chapter['chapter_number']
            roman_num = chapter.get('roman_num', str(chapter_num))
            
            # 保存译文
            translated_filename = f"chapter_{roman_num}_{safe_lang}.txt"
            translated_path = os.path.join(chapters_dir, translated_filename)
            
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(chapter['translated_content'])
            
            # 保存双语对照
            bilingual_filename = f"chapter_{roman_num}_bilingual.txt"
            bilingual_path = os.path.join(chapters_dir, bilingual_filename)
            
            with open(bilingual_path, 'w', encoding='utf-8') as f:
                f.write("=" * 40 + " 英文原文 " + "=" * 40 + "\n")
                f.write(chapter['original_content'])
                f.write("\n\n" + "=" * 40 + f" {target_language}译文 " + "=" * 40 + "\n")
                f.write(chapter['translated_content'])
        
        print(f"单个章节文件已保存至: {chapters_dir}")
        
        # 生成摘要报告
        self._generate_summary(translated_chapters, output_dir, target_language)
    
    def _generate_summary(self, chapters: List[Dict], output_dir: str, target_language: str):
        """
        生成翻译摘要报告
        
        Args:
            chapters: 章节列表
            output_dir: 输出目录
            target_language: 目标语言
        """
        report_path = os.path.join(output_dir, "translation_summary.txt")
        
        total_original_chars = sum(len(chapter['original_content']) for chapter in chapters)
        total_translated_chars = sum(len(chapter['translated_content']) for chapter in chapters)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"《爱丽丝梦游仙境》{target_language}翻译报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"翻译基本信息：\n")
            f.write(f"- 目标语言: {target_language}\n")
            f.write(f"- 章节数量: {len(chapters)}\n")
            f.write(f"- 原文总字符数: {total_original_chars:,}\n")
            f.write(f"- 译文总字符数: {total_translated_chars:,}\n")
            f.write(f"- 翻译比率: {total_translated_chars/total_original_chars:.2f}\n\n")
            
            f.write("章节详情：\n")
            f.write("-" * 60 + "\n")
            
            for chapter in chapters:
                orig_len = len(chapter['original_content'])
                trans_len = len(chapter['translated_content'])
                ratio = trans_len / orig_len if orig_len > 0 else 0
                
                f.write(f"第{chapter['chapter_number']}章 ({chapter.get('roman_num', 'N/A')})\n")
                f.write(f"  标题: {chapter['original_title']} → {chapter['translated_title']}\n")
                f.write(f"  原文长度: {orig_len:,} 字符\n")
                f.write(f"  译文长度: {trans_len:,} 字符\n")
                f.write(f"  长度比率: {ratio:.2f}\n")
                f.write("-" * 40 + "\n")
            
            f.write(f"\n报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"翻译摘要报告已生成: {report_path}")

def main():
    """主函数"""
    # 配置参数
    API_KEY = "sk-6fc1143a5b8c4859a8c659c0589eca9f"  # 替换为你的API密钥
    INPUT_FILE = "merge_all.txt"  # 输入文件路径
    lanList = ['Chinese', 'French', 'German', 'Russian', 'Japanese', 'Spanish', 'Italian']
    TARGET_LANGUAGE = lanList[6] # 目标语言: 0, 1, 2, 3, 4, 5, 6
    OUTPUT_DIR = "alice_translation"  # 输出目录
    
    print("《爱丽丝梦游仙境》章节翻译工具")
    print("=" * 50)
    
    # 初始化翻译器
    translator = DeepSeekTranslator(api_key=API_KEY)
    
    try:
        # 1. 读取文件
        print("步骤1: 正在读取文件...")
        text_content = translator.read_file(INPUT_FILE)
        print(f"✓ 文件读取完成，总字符数: {len(text_content):,}")
        
        # 2. 划分章节
        print("\n步骤2: 正在分析章节结构...")
        chapters = translator.split_into_chapters(text_content)
        print(f"✓ 共识别到 {len(chapters)} 个章节")
        
        # 3. 翻译每个章节
        print(f"\n步骤3: 开始翻译为 {TARGET_LANGUAGE}...")
        print("=" * 50)
        
        translated_chapters = []
        
        for i, chapter in enumerate(chapters):
            try:
                print(f"\n[{i + 1}/{len(chapters)}] ", end="")
                translated_chapter = translator.translate_chapter(
                    chapter, 
                    TARGET_LANGUAGE
                )
                translated_chapters.append(translated_chapter)
                print(f"✓ 完成: {chapter['title']}")
                
                # 添加章节间延迟
                if i < len(chapters) - 1:
                    time.sleep(2)  # 避免API频率限制
                    
            except Exception as e:
                print(f"\n✗ 章节翻译失败: {chapter['title']}")
                print(f"   错误信息: {e}")
                print("   将跳过此章节继续处理...")
                
                # 添加错误章节占位符
                translated_chapters.append({
                    'original_title': chapter['title'],
                    'translated_title': chapter['title'] + " [翻译失败]",
                    'original_content': chapter['content'],
                    'translated_content': f"【翻译失败】\n\n{chapter['content']}",
                    'chapter_number': chapter['chapter_number'],
                    'roman_num': chapter.get('roman_num', str(chapter['chapter_number']))
                })
        
        # 4. 保存结果
        print("\n" + "=" * 50)
        print("步骤4: 正在保存翻译结果...")
        translator.save_translation(
            translated_chapters,
            OUTPUT_DIR,
            TARGET_LANGUAGE,
            save_format='both'
        )
        
        print("\n" + "=" * 50)
        print("🎉 翻译任务完成！")
        print(f"所有文件已保存至: {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

def quick_translate():
    """快速翻译函数（简化版）"""
    API_KEY = input("请输入DeepSeek API密钥: ").strip()
    TARGET_LANGUAGE = input("请输入目标语言（如：中文、日语、法语）: ").strip()
    INPUT_FILE = "alice_english.txt"
    
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件 {INPUT_FILE} 不存在！")
        return
    
    translator = DeepSeekTranslator(api_key=API_KEY)
    
    # 读取并分割章节
    text = translator.read_file(INPUT_FILE)
    chapters = translator.split_into_chapters(text)
    
    print(f"\n开始翻译 {len(chapters)} 个章节...")
    
    # 只翻译前3章作为示例
    sample_chapters = chapters[:3]
    translated = []
    
    for chapter in sample_chapters:
        print(f"翻译: {chapter['title']}")
        try:
            result = translator.translate_chapter(chapter, TARGET_LANGUAGE)
            translated.append(result)
        except Exception as e:
            print(f"  失败: {e}")
    
    # 保存示例
    output_dir = "sample_translation"
    translator.save_translation(translated, output_dir, TARGET_LANGUAGE, save_format='txt')
    
    print(f"\n示例翻译已保存到: {output_dir}")

if __name__ == "__main__":
    print("请选择模式:")
    print("1. 完整翻译（所有章节）")
    print("2. 快速测试（仅翻译前3章）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        quick_translate()
    else:
        main()